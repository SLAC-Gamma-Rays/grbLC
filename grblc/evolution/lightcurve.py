# standard libs
import os
import re
import sys
from functools import reduce

# third party libs
import numpy as np
import pandas as pd
import lmfit as lf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import plotly.express as px
#from matplotlib.figure import Figure

# custom modules
from grblc.util import get_dir
from . import io


class Lightcurve:
    _name_placeholder = "unknown grb"
    _flux_fixed_inplace = False


    def __init__(
        self,
        path: str = None,
        xdata: np.float64 = None,
        ydata: np.float64 = None,
        xerr: np.float64 = None,
        yerr: np.float64 = None,
        band: str = None,
        data_space: str = "log",
        name: str = None,
    ):
        """The main module for fitting lightcurves.

        .. warning::
            Data stored in :py:class:`Lightcurve` objects are always in logarithmic
            space; the parameter ``data_space`` is only used to convert data to log space
            if it is not already in such. If your data is in linear space [i.e., your
            time data is sec, and not $log$(sec)], then you should set ``data_space``
            to ``lin``.


        Parameters
        ----------
        path : str, optional
            Name of file containing light curve data, by default None
        xdata : array_like, optional
            X values, length (n,), by default None
        ydata : array_like, optional
            Y values, by default None
        xerr : array_like, optional
            X error, by default None
        yerr : array_like, optional
            Y error, by default None
        data_space : str, {log, lin}, optional
            Whether the data inputted is in log or linear space, by default 'log'
        name : str, optional
            Name of the GRB, by default :py:class:`Model` name, or ``unknown grb`` if not
            provided.
        """
        assert bool(path) ^ (
            xdata is not None and ydata is not None
        ), "Either provide a path or xdata, ydata."

        if name:
            self.name = name
        else:
            self.name = self._name_placeholder

        if isinstance(path, str):
            self.path = path
            self.set_data(*self.read_data(path), data_space=data_space)
        else:
            self.path = reduce(
                os.path.join,
                [
                    get_dir(),
                    "lightcurves",
                    "{}_flux.txt".format(self.name.replace(" ", "_").replace(".", "p")),
                ],
            )
            self.set_data(xdata, ydata, xerr, yerr, band, data_space=data_space)


    def set_bounds(
        self, xmin=-np.inf, xmax=np.inf, ymin=-np.inf, ymax=np.inf
    ):
        """Sets the bounds on the xdata and ydata to (1) plot and (2) fit with. Either
            provide bounds or xmin, xmax, ymin, ymax. Assumes data is already in log
            space. If :py:meth:`Lightcurve.set_data` has been called, then the data
            has already been converted to log space.

        Parameters
        ----------
        bounds : array_like of length 4, optional
            Bounds on inputted x and y-data, by default None
        xmin : float, optional
            Minimum x, by default -np.inf
        xmax : float, optional
            Maximum x, by default np.inf
        ymin : float, optional
            Minimum y, by default -np.inf
        ymax : float, optional
            Maximum y, by default np.inf
        """

        xmask = (xmin <= self.xdata) & (self.xdata <= xmax)
        ymask = (ymin <= self.ydata) & (self.ydata <= ymax)
        self.mask = xmask & ymask

        self.set_data(self.xdata, self.ydata, self.xerr, self.yerr, data_space="log")


    def set_data(self, xdata, ydata, xerr=None, yerr=None, band=None, data_space="log"):
        """Set the `xdata` and `ydata`, and optionally `xerr` and `yerr` of the lightcurve.

        .. warning::
            Data stored in :py:class:`Lightcurve` objects are always in logarithmic
            space; the parameter ``data_space`` is only used to convert data to log space
            if it is not already in such. If your data is in linear space [i.e., your
            time data is sec, and not log(sec)], then you should set ``data_space``
            to ``lin``.

        Parameters
        ----------
        xdata : array_like
            X data
        ydata : array_like
            Y data
        xerr : array_like, optional
            X error, by default None
        yerr : array_like, optional
            Y error, by default None
        data_space : str, {log, lin}, optional
            Whether the data inputted is in logarithmic or linear space, by default 'log'.
        """
        if not hasattr(self, "mask"):
            self.mask = np.ones(len(xdata), dtype=bool)

        def convert_data(data):
            if data_space == "lin":
                d = np.log10(data)
            elif data_space == "log":
                d = data
            else:
                raise ValueError("data_space must be 'log' or 'lin'")

            return np.asarray(d)

        def convert_err(data, err):
            if data_space == "lin":
                eps = err / (data * np.log(10))
            elif data_space == "log":
                eps = err
            else:
                raise ValueError("data_space must be 'log' or 'lin'")
            return np.asarray(eps)

        self.orig_xdata = convert_data(xdata)
        self.orig_ydata = convert_data(ydata)
        self.xdata = self.orig_xdata[self.mask]
        self.ydata = self.orig_ydata[self.mask]
        self.orig_xerr = convert_err(xdata, xerr) if xerr is not None else None
        self.orig_yerr = convert_err(ydata, yerr) if yerr is not None else None
        self.xerr = self.orig_xerr[self.mask] if xerr is not None else None
        self.yerr = self.orig_yerr[self.mask] if yerr is not None else None
        self.band = band


    def exclude_range(self, xs=(), ys=(), data_space="log"):
        """
        `exclude_range` takes a range of x and y values and excludes them from the data
        of the current lightcurve.

        Parameters
        ----------
        xs : tuple of form (xmin, xmax), optional
            Range along the x-axis to exclude.
        ys : tuple of form (ymin, ymax), optional
            Range along the y-axis to exclude.
        data_space : str, {log, lin}, optional
            Whether you'd like to exclude in logarithmic or linear space, by default 'log'.
        """
        if xs == [] and ys == []:
            return
        if xs == ():
            xs = (-np.inf, np.inf)
        if ys == ():
            ys = (-np.inf, np.inf)
        assert len(xs) == 2 and len(ys) == 2, "xs and ys must be tuples of length 2"

        xmin, xmax = xs
        ymin, ymax = ys
        xmask = (xmin <= self.orig_xdata) & (self.orig_xdata <= xmax)
        self.mask &= ~xmask
        ymask = (ymin <= self.orig_ydata) & (self.orig_ydata <= ymax)
        self.mask &= ~ymask
        self.set_data(self.xdata, self.ydata, self.xerr, self.yerr, data_space=data_space)


    def read_data(self, path: str):
        """
            Reads in data from a file. The data must be in the correct format.
            See the :py:meth:`io.read_data` for more information.

        Parameters
        ----------
        path : str

        Returns
        ----------
        xdata, ydata, xerr, yerr : array_like
        """
        df = io.read_data(path)

        df = df[df['mag_err'] != 0]

        assert len(df)!=0, "Only limiting magnitudes present."

        xdata = df["time_sec"].to_numpy()
        ydata = df["mag"].to_numpy()
        xerr = None
        yerr = df["mag_err"].to_numpy()
        band = df["band"].to_list()

        return xdata, ydata, xerr, yerr, band

    def show_data(self, save=False, fig_kwargs={}, save_kwargs={}):
        """
            Plots the lightcurve data. If no fit has been ran, :py:meth:`Lightcurve.show` will call
            this function.

            .. note:: This doesn't plot any fit results. Use :py:meth:`Lightcurve.show_fit` to do so.

            Example:

            .. jupyter-execute::

                import numpy as np
                import grblc

                model = grblc.Model.W07(vary_t=False)
                xdata = np.linspace(0, 10, 15)
                yerr = np.random.normal(0, 0.5, len(xdata))
                ydata = model(xdata, 5, -12, 1.5, 0) + yerr
                lc = grblc.Lightcurve(xdata=xdata, ydata=ydata, yerr=yerr, model=model)
                lc.show_data()


        Parameters
        ----------
        fig_kwargs : dict, optional
            Arguments to pass to ``plt.figure()``, by default {}.
        """

        fig_dict = dict(
            figsize=[
                plt.rcParams["figure.figsize"][0],
                plt.rcParams["figure.figsize"][0],
            ]
        )
        if bool(fig_kwargs):
            fig_dict.update(fig_kwargs)
        plot_fig = plt.figure(**fig_dict)
        ax = plot_fig.add_subplot(1, 1, 1)

        xmin, xmax, ymin, ymax = -np.inf, +np.inf, np.inf, +np.inf
        logt = self.orig_xdata
        mag = self.orig_ydata
        logterr = self.orig_xerr
        magerr = self.orig_yerr

        mask = (logt >= xmin) & (logt <= xmax) & (mag >= ymin) & (mag <= ymax)

        # plot all data points inside xmin and xmax in black
        ax.errorbar(
            logt[mask],
            mag[mask],
            xerr=logterr[mask] if logterr is not None else logterr,
            yerr=magerr[mask] if magerr is not None else magerr,
            color="k",
            fmt=".",
            ms=10,
            zorder=0,
        )

        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()

        # plot all data points outside of xmin and xmax in grey
        if sum(~mask) > 0:
            ax.errorbar(
                logt[~mask],
                mag[~mask],
                xerr=logterr[~mask] if logterr is not None else logterr,
                yerr=magerr[~mask] if magerr is not None else magerr,
                color="grey",
                fmt=".",
                ms=10,
                alpha=0.2,
                zorder=0,
            )

        ax.set_xlim(ax_xlim)
        ax.set_ylim(ax_ylim)
        ax.set_xlabel("log10 Time (sec)")
        ax.set_ylabel("Magnitudes")
        ax.set_title(self.name)

        plt.show()

    def _overlap(start1, end1, start2, end2):
        #how much does the range (start1, end1) overlap with (start2, end2)
        return max(max((end2-start1), 0) - max((end2-end1), 0) - max((start2-start1), 0), 0)
        
    def displayGRB(self, save_static=False, save_static_type='.png', save_interactive=False, save_in_folder='plots/'):
        '''
        For an interactive plot
        '''

        fig = px.scatter(
                    x=self.xdata,
                    y=self.ydata,
                    error_y=self.yerr,
                    color=self.band,
                    #symbol=data['marker'],
                    #hover_data=['Source', 'Telescope'],
                )

        headpoint_list = []
        for t,m,e in zip(self.xdata,self.ydata,self.yerr):
            if e == 0:
                headpoint_list.append((t, m))

        tailpoint_list = [(i, j+1) for (i, j) in headpoint_list]

        #make a list of go.layout.Annotation() for each pair of arrow head and tail
        arrows = []
        for head, tail in zip(headpoint_list, tailpoint_list):
            arrows.append(dict(
                x= head[0], #x position of arrowhead
                y= head[1], #y position of arrowhead
                showarrow=True,
                xref = "x", #reference axis of arrow head coordinate_x
                yref = "y",#reference axis of arrow head coordinate_y
                arrowcolor="gray", #color of arrow
                arrowsize = 1.5, #size of arrow head
                arrowwidth = 2, #width of arrow line
                ax = tail[0], #arrow tail coordinate_x
                ay = tail[1], #arrow tail coordinate_y
                axref= "x", #reference axis of arrow tail coordinate_x
                ayref= "y", #reference axis of arrow tail coordinate_y
                arrowhead=3, #annotation arrow head style, from 0 to 8
                ))

        #update_layout with annotations
        fig.update_layout(annotations=arrows)

        font_dict=dict(family='arial',
                    size=18,
                    color='black'
                    )

        fig['layout']['yaxis']['autorange'] = 'reversed'
        fig.update_yaxes(title_text="<b>Magnitude<b>",
                        title_font_color='black',
                        title_font_size=18,
                        showline=True,
                        showticklabels=True,
                        showgrid=False,
                        linecolor='black', 
                        linewidth=2.4, 
                        ticks='outside', 
                        tickfont=font_dict,
                        mirror='allticks', 
                        tickwidth=2.4, 
                        tickcolor='black',  
                        )

        fig.update_xaxes(title_text="<b>log10 Time (s)<b>",
                        title_font_color='black',
                        title_font_size=18,
                        showline=True,
                        showticklabels=True,
                        showgrid=False,
                        linecolor='black',
                        linewidth=2.4,
                        ticks='outside',
                        tickfont=font_dict,
                        mirror='allticks',
                        tickwidth=2.4,
                        tickcolor='black',
                        )

        fig.update_layout(title="GRB " + self.name,
                        title_font_size=25,
                        font=font_dict,
                        plot_bgcolor='white',  
                        width=960,
                        height=540,
                        margin=dict(l=40,r=40,t=50,b=40)
                        )
        
        fig.show()

        if save_static:
            fig.write_image(save_in_folder+self.name+save_static_type)

        if save_interactive:
            fig.write_html(save_in_folder+self.name+'.html')
    
    def colorevolGRB(self, return_rescaledf=False, save_plot=False, save_in_folder=''):

        light = pd.DataFrame()
        light['time_sec'] = self.xdata
        light['mag'] = self.ydata
        light['mag_err'] = self.yerr
        light['band'] = self.band

        assert len(light)>1, "Has only one data point."

        occur = light['band'].value_counts()
        filterslist = occur.index
        #data['occur']=data['band'].map(data['band'].value_counts())
        
        
        # Identifying the most numerous filter in the GRB 
        mostcommonfilter = occur.index[0]
        mostcommonfilter_occur = occur[0]

        print('The most numerous filter of this GRB: ',mostcommonfilter,', with', mostcommonfilter_occur, 'occurrences.\n'+
              'The most numerous will be considered for rescaling')
        
        scalingfactorslist = [[mostcommonfilter, mostcommonfilter_occur, [[0,0,0]]]] ## since the most common filter is not scaled
        
        mostcommonlight=light.loc[(light['band'] == mostcommonfilter)]
        mostcommonx=mostcommonlight['time_sec'].values
        mostcommony=mostcommonlight['mag'].values  
        mostcommonyerr=mostcommonlight['mag_err'].values  
        
        for j in range(1, len(occur)):
            scalingfactorslist.append([occur.index[j],occur[j],[]])
        
        evolutionrescalingfactor=[]
        
        for j in range(1,len(occur)):
            
            sublight=light.loc[(light['band'] == occur.index[j])]
            subx=sublight['time_sec'].values
            suby=sublight['mag'].values
            suberror_y=sublight['mag_err'].values
            
            timediff = [[p1,p2] for p1 in range(len(mostcommonx)) for p2 in range(len(subx)) if np.log10(np.abs(10**mostcommonx[p1]-10**subx[p2]))<=np.log10((10**mostcommonx[p1])*2.5/100)]

            if len(timediff)!=0:
                for ll in timediff:
                    sf2=[subx[ll[1]],mostcommony[ll[0]]-suby[ll[1]],np.log10(np.abs(10**mostcommonx[ll[0]]-10**subx[ll[1]])),mostcommonyerr[ll[0]]+suberror_y[ll[1]]]
                    scalingfactorslist[j][2].append(sf2)  

        for fl in scalingfactorslist:

            times=set(el[0] for el in fl[2])
            
            for tt in times:
                suppllist=[fl[2][x] for x in range(len(fl[2])) if fl[2][x][0]==tt]
                suppllistdist=[fl[2][x][2] for x in range(len(fl[2])) if fl[2][x][0]==tt]     
                
                mindistpos=suppllistdist.index(min(suppllistdist))
                
                evolutionrescalingfactor.append([fl[0],fl[1],suppllist[mindistpos]])    
                
        finalevolutionlist=evolutionrescalingfactor 
        finalevolutionlist=sorted(finalevolutionlist, key=lambda finalevolutionlist: finalevolutionlist[2][0])
      
        filt=[jj[0] for jj in finalevolutionlist if jj[0]!=mostcommonfilter]
        filtoccur=[jj[1] for jj in finalevolutionlist if jj[0]!=mostcommonfilter]
        resctime=[jj[2][0] for jj in finalevolutionlist if jj[0]!=mostcommonfilter]
        rescfact=[jj[2][1] for jj in finalevolutionlist if jj[0]!=mostcommonfilter]
        rescfacterr=[jj[2][3] for jj in finalevolutionlist if jj[0]!=mostcommonfilter]
        rescfactweights=[(1/jj[2][3]) for jj in finalevolutionlist if jj[0]!=mostcommonfilter]
        
        rescale_df=pd.DataFrame(list(zip(filt,filtoccur,resctime,rescfact,
                                                    rescfacterr,rescfactweights)),columns=['band','Occur_band','Log10(t)','Resc_fact','Resc_fact_err','Resc_fact_weights'])

        x_all = rescale_df['Log10(t)']
        y_all = rescale_df['Resc_fact']
        yerr_all = rescale_df['Resc_fact_err']
        filters = [*set(rescale_df['band'].values)]
        rescale_df['plot_color'] = ""

        # Set the color map to match the number of filter
        cmap = plt.get_cmap('gist_ncar')
        cNorm  = colors.Normalize(vmin=0, vmax=len(filters))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        # Plot each species
        fig = plt.figure()

        for i, band in enumerate(filters):
            colour = scalarMap.to_rgba(i)
            index = rescale_df['band'] == band
            plt.scatter(x_all[index], y_all[index],
                        s=15, 
                        color=colour, 
                        label=filters[i])
            plt.errorbar(x_all[index], y_all[index], yerr_all[index],
                        fmt='o',
                        barsabove=True,
                        ls='',
                        color=colour
                        )
        for j in rescale_df.index:
            rescale_df.at[j,"plot_color"] = colour

        resc_slopes_df = pd.DataFrame()
        resc_slopes_df.index = filters
        resc_slopes_df['slope'] = ""
        resc_slopes_df['slope_err'] = ""
        resc_slopes_df['intercept'] = ""
        resc_slopes_df['inter_err'] = ""
        resc_slopes_df['acceptance'] = ""
        resc_slopes_df['red_chi2'] = ""
        resc_slopes_df['comment'] = ""
        resc_slopes_df['plot_color'] = ""

        for band in resc_slopes_df.index:
            ind = rescale_df.index[rescale_df['band'] == band][0]
            resc_slopes_df.loc[band]['plot_color'] = rescale_df.loc[ind]["plot_color"]
            resc_band_df = rescale_df[rescale_df['band'] == band]

            x = resc_band_df['Log10(t)']
            y = resc_band_df['Resc_fact']
            weights = resc_band_df['Resc_fact_weights']
            
            ## lmfit linear

            if len(x) >= 3:
                linear_model = lf.models.LinearModel(prefix='line_')
                linear_params = linear_model.make_params()
                
                linear_params['line_slope'].set(value=-1.0)
                linear_params['line_intercept'].set(value=np.max(y))

                linear_fit = linear_model.fit(y, params=linear_params, x=x, weights=weights)
                
                resc_slopes_df.loc[band]['slope'] = np.around(linear_fit.params['line_slope'].value, decimals=4)
                resc_slopes_df.loc[band]['slope_err'] = np.around(linear_fit.params['line_slope'].stderr, decimals=4)
                resc_slopes_df.loc[band]['intercept'] = np.around(linear_fit.params['line_intercept'].value, decimals=4)
                resc_slopes_df.loc[band]['inter_err'] = np.around(linear_fit.params['line_intercept'].stderr, decimals=4)
                resc_slopes_df.loc[band]['acceptance'] = np.around(np.abs(resc_slopes_df.loc[band]['slope_err']/resc_slopes_df.loc[band]['slope']), decimals=4)
                resc_slopes_df.loc[band]['red_chi2'] = np.around(linear_fit.redchi, decimals=4)
                
            else: # not enough data points
                resc_slopes_df.loc[band]['slope'] = 0
                resc_slopes_df.loc[band]['slope_err'] = 0
                resc_slopes_df.loc[band]['intercept'] = 0
                resc_slopes_df.loc[band]['inter_err'] = 0
                resc_slopes_df.loc[band]['acceptance'] = 0
                resc_slopes_df.loc[band]['comment'] = "insufficient data"
                resc_slopes_df.loc[band]['red_chi2'] = 'insufficient data'
                
            if resc_slopes_df.loc[band]['slope'] != 0:
                if resc_slopes_df.loc[band]['acceptance'] < 10000: #put ad-hoc to have all the plots

                    y_fit = resc_slopes_df.loc[band]['slope'] * x + resc_slopes_df.loc[band]['intercept']
                    y_fit_err = resc_slopes_df.loc[band]['slope_err'] * x + resc_slopes_df.loc[band]['inter_err']

                    plt.plot(x, y_fit, 
                            color=resc_slopes_df.loc[band]["plot_color"],
                            label=str(band+ ": " + str(resc_slopes_df.loc[band]["slope"])))

                    if np.abs(resc_slopes_df.loc[band]['slope']) < 0.1:
                        resc_slopes_df.loc[band]['comment'] = "no color evolution"
                    elif resc_slopes_df.loc[band]['slope']-(3*resc_slopes_df.loc[band]['slope_err'])<=0<=resc_slopes_df.loc[band]['slope']+(3*resc_slopes_df.loc[band]['slope_err']):
                        resc_slopes_df.loc[band]['comment'] = "no color evolution"
                    else:    
                        resc_slopes_df.loc[band]['comment'] = "slope >= 0.1"

                else:
                    resc_slopes_df.loc[band]['comment'] = "slope=0"  

        rescale_df.drop(labels='plot_color', axis=1, inplace=True)
        resc_slopes_df.drop(labels='plot_color', axis=1, inplace=True)
        
        plt.title('Rescaling factors for '+ str(self.name))
        plt.xlabel('log10 Time (sec)')
        plt.ylabel('Rescale factors wrt '+mostcommonfilter+' (mag)')
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        plt.tight_layout
        plt.show()

        if save_plot:
            plt.savefig(os.path.join(save_in_folder+'/'+save_in_folder+'/'+str(self.name)+'_colorevol.png'))

        if return_rescaledf:
            return rescale_df

        return fig, resc_slopes_df



major, *__ = sys.version_info
readfile_kwargs = {"encoding": "utf-8"} if major >= 3 else {}


def _readfile(path):
    with open(path, **readfile_kwargs) as fp:
        contents = fp.read()
    return contents


version_regex = re.compile('__version__ = "(.*?)"')
contents = _readfile(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "__init__.py"
    )
)
__version__ = version_regex.findall(contents)[0]

__directory__ = get_dir()
