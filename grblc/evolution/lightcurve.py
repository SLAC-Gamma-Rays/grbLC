# Standard libs
import os
import re
import warnings

# Third party libs
import numpy as np
import pandas as pd
import lmfit as lf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.ticker as mt
import plotly.express as px

# Custom modules
from . import io

# Ignore warnings
warnings.filterwarnings(action='ignore')#, category=['Warning', 'RuntimeWarning'])

class Lightcurve(object):
    _name_placeholder = "xxxxxxA"
    _flux_fixed_inplace = False



    def __init__(
        self,
        path: str = None,
        data_space: str = 'lin',
        name: str = None,
    ):
        """
        The main module for fitting lightcurves.

        Reads in data from a file. The data must be in the prescribed format: 
        Time, Magnitude or Upper limit, Magnitude Error in 1 sigma (0 when upper limit), Band, System, Telescope, Galactic extinction correction flag ('y'/'n'), Source
        See the :py:meth:`io.read_data` for more information.

        .. warning::
            Data stored in :py:class:`lightcurve` objects are always in logarithmic
            space; the parameter ``data_space`` is only used to convert data to log space
            if it is not already in such. If your data is in linear space [i.e., your
            time data is sec, and not $log$(sec)], then you should set ``data_space``
            to ``lin``.


        Parameters
        ----------
        path : str, optional
            Name of file containing light curve data, by default None
        data_space : str, {log, lin}, optional
            Whether the time data inputted is in log or linear space, by default 'log'
        set_bands: str, optional
            Whether to approximate bands for color evolution analysis. The following approximations are performed:
            - primed SDSS ugriz bands are treated same as unprimed
            - Ks = K' = K
            - Js = J
            - Mould photometric system is treated same as Johnson - Cousins
        name : str, optional
            Name of the GRB, by default :py:class:`Model` name, or ``unknown grb`` if not
            provided.
        """

        # Asserting the name of the GRB
        if name:
            self.name = name
        else:
            self.name = self._name_placeholder

        # Reading data
        if isinstance(path, str):
            self.path = path
            self.light = io.read_data(path, data_space=data_space)



    def displayGRB(
            self, 
            save_static = False, 
            save_interactive = False, 
            save_in_folder = 'plots/'
    ):

        """
        For an interactive plot
        """

        fig = px.scatter(
                    data_frame=self.light,
                    x=self.light['time_sec'],
                    y=self.light['mag'],
                    error_y=self.light['mag_err'],
                    color=self.light['band'],
                    hover_data=['telescope', 'source'],
                )

        tailpoint_list = []
        for t,m,e in zip(self.light['time_sec'],self.light['mag'],self.light['mag_err']):
            if e == 0:
                tailpoint_list.append((t, m))

        headpoint_list = [(i, j+1) for (i, j) in tailpoint_list]

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
                arrowsize = 1, #size of arrow head
                arrowwidth = 2, #width of arrow line
                ax = tail[0], #arrow tail coordinate_x
                ay = tail[1]-0.25, #arrow tail coordinate_y
                axref= "x", #reference axis of arrow tail coordinate_x
                ayref= "y", #reference axis of arrow tail coordinate_y
                arrowhead=4, #annotation arrow head style, from 0 to 8
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

        if save_static:
            fig.write_image(save_in_folder+self.name+'.png')

        if save_interactive:
            fig.write_html(save_in_folder+self.name+'.html')

        return fig



    # The function that calls io.py to read the data and performs band approximations.
    # It is not callable outside the class.
    def set_data(
        self, 
        set_bands = False
    ):
        """
        Does the approximations necessary for the analysis

        .. warning::
            Data stored in :py:class:`lightcurve` objects are always in logarithmic
            space; the parameter ``data_space`` is only used to convert data to log space
            if it is not already in such. If your data is in linear space [i.e., your
            time data is sec, and not log(sec)], then you should set ``data_space``
            to ``lin``.

        Parameters
        ----------
        path : str, optional
            Name of file containing light curve data, by default None
        data_space : str, {log, lin}, optional
            Whether the time data inputted is in log or linear space, by default 'log'
        set_bands: str, optional
            Whether to approximate bands for color evolution analysis. The following approximations are performed:
            - primed SDSS ugriz bands are treated same as unprimed
            - Ks = K' = K
            - Js = J
            - Mould photometric system is treated same as Johnson - Cousins
        """

        # Aprroximations related to the analysis are performed within the class 
        # as they are checked for validity only in the context of color evolution

        # here the code requires only magnitudes and not limiting magnitudes,
        # there are some transients observed in the optical before the
        # satellite trigger, thus they have negative times since in our
        # analysis, we consider the trigger time as start time of the LC

        assert len(self.light[self.light['mag_err'] != 0]) != 0, "Only limiting magnitudes present."
        assert len(self.light[self.light['mag_err'] != 0]) > 1, "Has only one data point."

        self.light = self.light[(self.light['mag_err'] != 0) & (self.light['time_sec']>0)]
        
        # Band approximations
        def _convert_bands(data):

            data = list(data)

            for i, band in enumerate(data):
                if band.lower() in ['clear', 'unfiltered', 'lum']:
                    band == band.lower()

            if set_bands:
                for i, band in enumerate(data):
                    if band=="u'":
                        data[i]="u"
                    if band=="g'":
                        data[i]="g"            
                    if band=="r'":
                        data[i]="r"
                    if band=="i'":
                        data[i]="i"            
                    if band=="z'":
                        data[i]="z"            
                    if band.upper()=="BJ":
                        data[i]="B"            
                    if band.upper()=="VJ":
                        data[i]="V"
                    if band.upper()=="UJ":
                        data[i]="U"            
                    if band.upper()=="RM":
                        data[i]="R"             
                    if band.upper()=="BM":
                        data[i]="B"
                    if band.upper()=="UM":
                        data[i]="U"
                    if band.upper()=="JS":
                        data[i]="J"            
                    if band.upper()=="KS":
                        data[i]="K"    
                    if band.upper()=="K'":
                        data[i]="K" 
                    if band.upper()=="KP":
                        data[i]="K" 

                bands = data
            else:
                bands = data

            return bands
        
        self.light["band_set"] = _convert_bands(self.light["band"])
    

   
    def colorevolGRB( 
            self, 
            resc_band = 'numerous', 
            print_status = True, 
            save = False, 
            save_in_folder = 'colorevol/'
    ):
        """
        This monstrosity performs the color evolution analysis.
        """

        # Initialising
        
        self.light['resc_fact'] = np.nan
        self.light['resc_fact_err'] = np.nan
        self.light['time_diff_percent'] = np.nan
        self.light['mag_overlap'] = np.nan
        self.light['resc_band_mag'] = np.nan # magnitude of the filter chosen for rescaling (either the most numerous or another one)
        self.light['resc_band_mag_err'] = np.nan # error on the magnitude of the filter chosen for rescaling

        # Counting the occurences of each band
        filters = pd.DataFrame(self.light['band_set'].value_counts())
        filters.rename(columns={'band_set':'band_occur'}, inplace=True)

        self.light["band_occur"] = self.light['band_set'].map(self.light['band_set'].value_counts())

        assert resc_band == 'numerous' or resc_band in self.light['band_set'], "Rescaling band provided is not present in data!"

        # Identifying the most numerous filter in the GRB 
        if resc_band == 'numerous':
            resc_band = filters.index[0]

        self.resc_band = resc_band

        if print_status:
            print(self.name)
            print('-------')
            print(filters, '\nThe reference filter for rescaling of this GRB: ', resc_band, 
                ', with', filters.loc[resc_band, 'band_occur'], 'occurrences.\n')
            

        # Set the color map to match the number of filter
        cmap = plt.get_cmap('gist_ncar')
        cNorm  = colors.Normalize(vmin=0, vmax=len(filters.index))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        filters['plot_color'] = ""
        for i, band in enumerate(filters.index):
            filters.at[band, 'plot_color'] = scalarMap.to_rgba(i)

        
        # In the following rows the code extracts only the datapoints with the filter chosen for rescaling (usually, the most numerous)

        resc_light = self.light.loc[(self.light['band_set'] == resc_band)] # resc_light dataframe is the one constituted of the chosen filter for rescaling,
        resc_time = resc_light['time_sec'].values                   # for simplicity it is called resc_light
        resc_mag = resc_light['mag'].values                        # time_sec is linear
        resc_magerr = resc_light['mag_err'].values

        # The following loop fills the columns of the general dataframe with the rescaling factor, its error, the time difference
        # between the filter in the dataframe and the filter chosen for rescaling, and the magnitude overlap (if overlap is zero,
        # then the magnitude values do not overlap)

        for row in self.light.index: # running on all the magnitudes
            if self.light.loc[row, 'band_set'] == resc_band:   # when the filter in the dataframe is the filter chosen for rescaling,
                continue                    # the rescaling factor is obviously not existing and the columns are filled with "-"
            else:
                compatiblerescalingfactors=[] # the compatiblerescalingfactors is a list that contains all the possible rescaling factors for a magnitude, since more of them can fall in the 2.5 percent criteria
                for pp in range(len(resc_light)): # running on the magnitudes of the filter chosen for rescaling
                    if np.abs(resc_time[pp]-self.light.loc[row, "time_sec"])<=(0.025*resc_time[pp]): # if the filter chosen for rescaling is in the 2.5 percent time condition with the magnitude of the loop
                        rescfact=resc_mag[pp]-self.light.loc[row, "mag"] # this is the rescaling factor, mag_resc_band - mag_filter
                        rescfacterr=np.sqrt(resc_magerr[pp]**2+self.light.loc[row, "mag_err"]**2) # rescaling factor error, propagation of uncertainties on both the magnitudes
                        timediff=np.abs(resc_time[pp]-self.light.loc[row, "time_sec"])/resc_time[pp] # linear time difference between the time of filter chosen for rescaling and the filter to be rescaled, divided by the chosen filter time
                        magchosenf=resc_mag[pp]
                        magchosenferr=resc_magerr[pp]
                        compatiblerescalingfactors.append([rescfact,rescfacterr,timediff,magchosenf,magchosenferr]) # all these values are appended in the compatiblerescalingfactors list, since in principle there may be more than one for a single datapoint
                    if len(compatiblerescalingfactors)==0: # if there are no rescaling factors that respect the 2.5 percent for the given filter, then once again the columns are filled with "-"
                        continue
                    else: # in the other cases
                        minimumtimediff=min([k[2] for k in compatiblerescalingfactors]) # the rescaling factor with the minimum time difference between the filter chosen for rescaling and the filter to be rescaled is taken
                        acceptedrescalingfactor=list(filter(lambda x: x[2] == minimumtimediff, compatiblerescalingfactors)) # this line locates the rescaling factor with minimum time difference
                        self.light.loc[row, "resc_fact"]=acceptedrescalingfactor[0][0]
                        self.light.loc[row, "resc_fact_err"]=acceptedrescalingfactor[0][1]
                        self.light.loc[row, "time_diff_percentage"]=acceptedrescalingfactor[0][2]
                        self.light.loc[row, "resc_band_mag"]=acceptedrescalingfactor[0][3]
                        self.light.loc[row, "resc_band_mag_err"]=acceptedrescalingfactor[0][4]

        self.light_rescalable = self.light.dropna(axis=0, subset=['resc_fact', 'resc_fact_err']) #[~np.isnan(self.light["resc_fact"])] # the self.light_rescalable selects only the datapoints with rescaling factors

        # The following command defines the dataframe of rescaling factors

        resc_df = self.light_rescalable[['time_sec', 'resc_fact', 'resc_fact_err', 'band_set', 'band_occur']]

        resc_filters = [*set(resc_df['band_set'].values)] # list of filters in the rescaling factors sample
        '''resc_df['plot_color'] = "" # empty list that will filled with the color map condition

        # Set the color map to match the number of filter
        cmap = plt.get_cmap('gist_ncar') # import the color map
        cNorm  = colors.Normalize(vmin=0, vmax=len(resc_filters)) # linear map of the colors in the colormap from data values vmin to vmax
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap) # The ScalarMappable applies data normalization before returning RGBA colors from the given colormap
'''
        # Plot each filter
        fig, axs = plt.subplots(2, 1, sharex=True)

        for band in filters.index:
            sublight=self.light.loc[self.light['band_set'] == band]
            axs[0].scatter(sublight['time_sec'], sublight['mag'], color=filters.loc[band, "plot_color"])
            axs[0].errorbar(x=sublight['time_sec'], y=sublight['mag'], yerr=sublight['mag_err'], color=filters.loc[band, "plot_color"], ls='')


        for i, band in enumerate(resc_filters): # loop on the given filter
            index = resc_df['band_set'] == band # selects the magnitudes that have the filter equal to the band on which the loop iterates
            axs[1].scatter(resc_df.loc[index, 'time_sec'], resc_df.loc[index, 'resc_fact'], # options for the plot of the central values
                        s=15,
                        color=filters.loc[band, 'plot_color']) # color-coding of the plot
            axs[1].errorbar(resc_df.loc[index, 'time_sec'], resc_df.loc[index, 'resc_fact'], resc_df.loc[index, 'resc_fact_err'], #options for the plot of the error bars, these must be added in this command
                        fmt='o', # this is the data marker, a circle
                        barsabove=True, # bars plotted above the data marker
                        ls='', # line style = None, so no lines are drawn between the points (later the fit will be done and plotted)
                        color=filters.loc[band, 'plot_color'] # color-coding
                        )

        resc_slopes_df = pd.DataFrame() # initialize of the rescaling factors fitting dataframe
        resc_slopes_df.index = filters.index[1::] # the filters are taken as index ignoring rescaling band
        resc_slopes_df['slope'] = np.nan # placeholder, default set to empty, then it will change - slope of the linear fit
        resc_slopes_df['slope_err'] = np.nan # placeholder, default set to empty, then it will change - error on slope
        resc_slopes_df['intercept'] = np.nan # placeholder, default set to empty, then it will change - intercept of linear fit
        resc_slopes_df['inter_err'] = np.nan # placeholder, default set to empty, then it will change - error on intercept
        resc_slopes_df['slope_err/slope'] = np.nan # placeholder, default set to empty, then it will change - slope_err/slope = |slope_err|/|slope|
        resc_slopes_df['red_chi2'] = np.nan # placeholder, default set to empty, then it will change - reduced chi^2
        resc_slopes_df['comment'] = "" # placeholder, default set to empty, then it will change - the comment that will say "no color evolution","color evolution"
        
        for band in resc_slopes_df.index: # in this loop, we assign the bands in the dataframe defined in line 580
            
            resc_band_df = resc_df[resc_df['band_set'] == band]

            x = resc_band_df['time_sec'].values # we here define the dataframe to fit, log10(time)
            y = resc_band_df['resc_fact'].values # the rescaling factors
            weights = 1/resc_band_df['resc_fact_err'].values # The weights are considered as 1/yerr given that the lmfit package will be used: https://lmfit.github.io/lmfit-py/model.html#the-model-class

            ## lmfit linear - lmfit is imported as "lf" -> the lmfit uses the Levenberg-Marquardt method
            # https://lmfit.github.io/lmfit-py/model.html#the-model-class
                        
            if len(x) >= 3: # the fitting will be performed if and only if, for the given filter, at least 3 rescaling factors are available
                linear_model = lf.models.LinearModel(prefix='line_') # importing linear model from lmfit
                linear_params = linear_model.make_params() # we here initialize the fitting parameters, then these will be changed

                linear_params['line_slope'].set(value=-1.0) # initializing the fitting slope
                linear_params['line_intercept'].set(value=np.max(y)) # initializing the fitting intercept

                linear_fit = linear_model.fit(y, params=linear_params, x=x, weights=weights) # the command for weighted lmfit
                resc_slopes_df.loc[band, 'slope'] = np.around(linear_fit.params['line_slope'].value, decimals=3) # slope of the fitting
                resc_slopes_df.loc[band, 'slope_err'] = np.around(linear_fit.params['line_slope'].stderr, decimals=3) # slope error
                resc_slopes_df.loc[band, 'intercept'] = np.around(linear_fit.params['line_intercept'].value, decimals=3) # intercept
                resc_slopes_df.loc[band, 'inter_err'] = np.around(linear_fit.params['line_intercept'].stderr, decimals=3) # intercept error
                resc_slopes_df.loc[band, 'slope_err/slope'] = np.around(np.abs(np.around(linear_fit.params['line_slope'].stderr, decimals=3) /np.around(linear_fit.params['line_slope'].value, decimals=3)), decimals=3) # slope_err/slope = |slope_err|/|slope|
                resc_slopes_df.loc[band, 'red_chi2'] = np.around(linear_fit.redchi, decimals=3) # reduced chi^2

            else: # not enough data points, less than 3 rescaling factors for the filter
                resc_slopes_df.loc[band, 'comment'] = "insufficient data"

            # commenting

            if np.abs(resc_slopes_df.loc[band, 'slope']) < 0.001: # in case of zero-slope, since the fittings have 3 digits precision we consider the precision 0.001 as the "smallest value different from zero"
                if resc_slopes_df.loc[band, 'slope_err/slope'] <= 10:
                    y_fit = resc_slopes_df.loc[band, 'slope'] * x + resc_slopes_df.loc[band, 'intercept']
                    axs[1].plot(x, y_fit, color=filters.loc[band, 'plot_color']) # plot of the fitting line between time_sec and resc_fact

                    resc_slopes_df.loc[band, 'comment'] = "no color evolution"


            if np.abs(resc_slopes_df.loc[band, 'slope']) >= 0.001: # in the case of non-zero slope
                if resc_slopes_df.loc[band, 'slope_err/slope'] <= 10: # this boundary of slope_err/slope is put ad-hoc to show all the plots
                                                                   # it's a large number that can be modified
                    y_fit = resc_slopes_df.loc[band, 'slope'] * x + resc_slopes_df.loc[band, 'intercept'] # fitted y-value according to linear model
                    axs[1].plot(x, y_fit, color=filters.loc[band, 'plot_color']) # plot of the fitting line between time_sec and resc_fact

                    if resc_slopes_df.loc[band, 'slope']-(3*resc_slopes_df.loc[band, 'slope_err'])<=0<=resc_slopes_df.loc[band, 'slope']+(3*resc_slopes_df.loc[band, 'slope_err']):
                        resc_slopes_df.loc[band, 'comment'] = "no color evolution" # in case it is comp. with zero in 3 sigma, there is no color evolution
                    else:
                        resc_slopes_df.loc[band, 'comment'] = "color evolution" # in case the slope is not compatible with zero in 3 sigma

                else:
                    resc_slopes_df.loc[band, 'comment'] = "slope_err/slope>10"  # when the slope_err/slope is very high

        for band in resc_slopes_df.index: # this loop defines the labels to be put in the rescaling factor plot legend

            if np.isnan(resc_slopes_df.loc[band, "slope"])==True: # in case the fitting is not done, the label will be "filter: no fitting"
                label=band+": failed fitting"
            else:
                label=band+": "+ str(resc_slopes_df.loc[band, "slope"]) + r'$\pm$' + str(resc_slopes_df.loc[band, "slope_err"])
                # when the slopes are estimated, the label is "filter: slope +/- slope_err"

            axs[1].scatter(x=[], y=[],
                        color=filters.loc[band, 'plot_color'],
                        label=label # here the labels for each filter are inserted
                        )

        fmt = '%.1f' # set the number of significant digits you want
        xyticks = mt.FormatStrFormatter(fmt)
        for ax in axs:
            ax.yaxis.set_major_formatter(xyticks)
            ax.xaxis.set_major_formatter(xyticks)

        axs[0].invert_yaxis()
        axs[0].set_xlabel('log10 Time (s)', fontsize=15)
        axs[0].set_ylabel('Magnitude', labelpad=15, fontsize=15)
        axs[0].tick_params(labelsize=15)

        axs[1].set_ylabel('Rescaling factor to '+self.resc_band, labelpad=15, fontsize=15)
        axs[1].tick_params(labelsize=15)

        #axs[1].locator_params(axis='x', nbins=5)
        #axs[1].locator_params(axis='y', nbins=5)
        #axs[0].yaxis.set_major_formatter('{x:9<5.1f}')
        
        fig.legend(title='Band: slope±err', bbox_to_anchor=(1, 0.946), loc='upper left', fontsize='large')   

        plt.rcParams['legend.title_fontsize'] = 'x-large'
        fig.suptitle("GRB "+self.name.split("/")[-1], fontsize=22)
        plt.rcParams['figure.figsize'] = [12, 8]
        fig.tight_layout()
            
        if print_status: # when this option is selected in the function it prints the following

            print("Individual point rescaling:")
            print(resc_df) # the dataframe of rescaling factors

            print("\nSlopes of rescale factors for each filter:")
            print(resc_slopes_df) # the dataframe that contains the fitting parameters of rescaling factors

        compatibilitylist=[] # here we initialize the list that contains the ranges of (slope-3sigma,slope+3sigma) for each filter

        for band in resc_slopes_df.index: # this code appends the values of (slope-3sigma,slope+3sigma) in case the slope is not a "nan"
                                          # and in case both the slope and slope_err are different from zero
            if resc_slopes_df.loc[band, 'slope']!=0 and resc_slopes_df.loc[band, 'slope_err']!=0 and np.isnan(resc_slopes_df.loc[band, 'slope'])==False and np.isnan(resc_slopes_df.loc[band, 'slope_err'])==False:
                compatibilitylist.append([band,[resc_slopes_df.loc[band, 'slope']-(3*resc_slopes_df.loc[band, 'slope_err']),
                                        resc_slopes_df.loc[band, 'slope']+(3*resc_slopes_df.loc[band, 'slope_err'])]])

        self.nocolorevolutionlist=[] # this is the list of filters with slopes that are compatible with zero in 3 sigma
        self.colorevolutionlist=[] # this is the list of filters with slopes that ARE NOT compatible with zero in 3 sigma
        for l in compatibilitylist:
            if l[1][0]<=0<=l[1][1]: # if slope-3sigma<=0<=slope+3sigma (namely, compat. with zero in 3sigma)
                    self.nocolorevolutionlist.append(l[0])                           # then for the given filter the slope is compatible with zero in 3sigma, NO COLOR EVOLUTION
            else:
                self.colorevolutionlist.append(l[0]) # in the other cases, slopes are not compatible with zero in 3 sigma, COLOR EVOLUTION

        if print_status:
            if len(self.nocolorevolutionlist)==0: # if there are no filters compatible with zero in 3 sigma
                print('No filters compatible with zero in 3sigma')

            else:
                print('Filters compatible with zero in 3sigma: ',*self.nocolorevolutionlist) # if there are filters without color evolution, namely, compatible with zero in 3 sigma

            if len(self.colorevolutionlist)==0: # if there are not filters with color evolution, namely, that are not compatible with zero in 3 sigma
                print('No filters compatible with zero in >3sigma')

            else: # otherwise
                print('Filters not compatible with zero in 3sigma: ',*self.colorevolutionlist)
            print('\n')
            print('No color evolution: ',*self.nocolorevolutionlist,' ; Color evolution: ',*self.colorevolutionlist) # print of the two lists


            string="" # this is the general printing of all the slopes
            for band in resc_slopes_df.index:
                string=string+band+":"+str(round(resc_slopes_df.loc[band, 'slope'],3))+"+/-"+str(round(resc_slopes_df.loc[band, 'slope_err'],3))+"; "

        if save:

            if not os.path.exists(save_in_folder):
                os.makedirs(save_in_folder)

            fig.savefig(os.path.join(save_in_folder+'/'+str(self.name)+'_colorevol.pdf'), dpi=300) #  , bbox_inches='tight' option to export the pdf plot of rescaling factors

            if len(self.nocolorevolutionlist)>len(self.colorevolutionlist):
                rescflag='yes'
            else:
                rescflag='no'

            #reportfile = open('report_colorevolution.txt', 'a')
            #reportfile.write(self.name.split("/")[-1]+" "+str(self.nocolorevolutionlist).replace(' ','')+" "+str(self.colorevolutionlist).replace(' ','')+" "+rescflag+" "+self.resc_band+"\n")
            #reportfile.close()


        return fig, resc_df, resc_slopes_df, #reportfile # the variables in the other case



    def rescaleGRB(
            self, 
            save = 'False',
            save_in_folder = 'rescale/'
    ): # this function makes the rescaling of the GRB

        def overlap(mag1lower,mag1upper,mag2lower,mag2upper): # this is the condition to state if two magnitude ranges overlap
            if mag1upper <mag2lower or mag1lower > mag2upper:
                return 0 # in the case of no overlap, zero is returned
            else:
                return max(mag1upper, mag2upper) # in the case of overlap, a value>0 is returned

        # here the code uses the colorevolGRB function defined above; the outputs of the function colorevolGRB will be used as input in the current function

        #output_colorevolGRB = self.colorevolGRB(print_status=False, return_rescaledf=False, save=False, chosenfilter=chosenfilter, save_in_folder=save_in_folder)
        #input = output_colorevolGRB
        #nocolorevolutionlist = input[2] # 3rd output of colorevolGRB function, this is the list of filters whose resc.fact. slopes are compatible with zero in 3sigma or are < 0.10
        #colorevolutionlist =input[3] # 4th output of colorevolGRB function, this is the list of filters whose resc.fact. slopes are incompatible with zero in 3sigma and are>0.10
        #resc_band = input[4] # 5th output of colorevolGRB function, this is the filter chosen for rescaling
        #light = input[5] # 6th output of colorevolGRB function, is the given dataframe (since the filter chosen for rescaling is present here and it is needed)

        # Before rescaling the magnitudes, the following instructions plot the magnitudes in the unrescaled case
        figunresc = px.scatter(
                x=self.light['time_sec'].values, # the time is set to log10(time) only in the plot frame
                y=self.light['mag'].values,
                error_y=self.light['mag_err'].values,
                color=self.light['band'].values,
                )

        font_dict=dict(family='arial', # font of the plot's layout
                    size=18,
                    color='black'
                    )

        figunresc['layout']['yaxis']['autorange'] = 'reversed' # here the y axis is reversed, so that higher magnitudes are down and lower are up
        figunresc.update_yaxes(title_text="<b>Magnitude<b>", # updating plot options in the y-axes
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

        figunresc.update_xaxes(title_text="<b>log10 Time (s)<b>", # updating plot options in the x-axes
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

        figunresc.update_layout(title="GRB " + self.name, # updating the layout of the plot
                        title_font_size=25,
                        font=font_dict,
                        plot_bgcolor='white',
                        width=960,
                        height=540,
                        margin=dict(l=40,r=40,t=50,b=40)
                        )

        # Two additive columns must be inserted in the light dataframe

        self.light["mag_rescaled_to_"+self.resc_band] = "" # the column with rescaled magnitudes
        self.light["mag_rescaled_err"] = "" # the column with magnitude errors, propagating the uncertainties on the magnitude itself and on the rescaling factor

        for rr in self.light.index:
            # In these cases, the rescaled magnitude is the same of the given magnitude:
            # 1) The datapoint has the filter chosen for rescaling (obviously, can't be rescaled to itself)
            # 2) If the rescaling factor is not estimated (time difference > 2.5 percent)
            # 3) If the magnitudes of the filter chosen for rescaling and the filter to be rescaled overlap (mag overlap >0)
            # 4) If the filter belongs to the list of filters that have color evolution
            if self.light.loc[rr, "resc_band_mag"] == "-" or self.light.loc[rr, "resc_band_mag_err"] =="-":
                self.light.loc[rr, "mag_rescaled_to_"+self.resc_band] = self.light.loc[rr, "mag"]
                self.light.loc[rr, "mag_rescaled_err"] = self.light.loc[rr, "mag_err"]
            else:
                magoverlapcheck=overlap(self.light.loc[rr, "mag"]-self.light.loc[rr, "mag_err"], # the overlap between the filter chosen for rescaling and the filter to be rescaled; zero if no overlap, number>0 if overlap
                                           self.light.loc[rr, "mag"]+self.light.loc[rr, "mag_err"],
                                           self.light.loc[rr, "resc_band_mag"]-self.light.loc[rr, "resc_band_mag_err"],
                                           self.light.loc[rr, "resc_band_mag"]+self.light.loc[rr, "resc_band_mag_err"])

            if (self.light.loc[rr, "band_set"]==self.resc_band) or (self.light.loc[rr, "resc_fact"] == "-") or (magoverlapcheck != 0) or (self.light.loc[rr, "band_set"] in self.colorevolutionlist):
                self.light.loc[rr, "mag_rescaled_to_"+self.resc_band] = self.light.loc[rr, "mag"]
                self.light.loc[rr, "mag_rescaled_err"] = self.light.loc[rr, "mag_err"]

            # In the following cases, magnitudes are rescaled
            # If the rescaling factor is estimated AND the time difference is smaller than 2.5 percent AND the overlap is zero AND the band is in the list of filters that have no color evolution
            elif (self.light.loc[rr, "resc_fact"] != "-") and (self.light.loc[rr, "time_diff_percentage"]<=0.025) and (magoverlapcheck==0) and (self.light.loc[rr, "band_set"] in self.nocolorevolutionlist):
                self.light.loc[rr, "mag_rescaled_to_"+self.resc_band] = self.light.loc[rr, "mag"]+self.light.loc[rr, "resc_fact"]
                self.light.loc[rr, "mag_rescaled_err"] = self.light.loc[rr, "resc_fact_err"] # the error on the rescaled magnitude is already estimated with the rescaling factor error

            # In all the other cases, the magnitudes can't be rescaled and the default options follow
            else:
                self.light.loc[rr, "mag_rescaled_to_"+self.resc_band] = self.light.loc[rr, "mag"]
                self.light.loc[rr, "mag_rescaled_err"] = self.light.loc[rr, "mag_err"]


        # The plot of the rescaled dataframe
        figresc = px.scatter(
                x=np.log10(self.light["time_sec"].values), # the time is set to log10(time) only in the plot frame
                y=self.light["mag_rescaled_to_"+self.resc_band].values,
                error_y=self.light["mag_rescaled_err"].values,
                color=self.light["band_set"],
                )

        font_dict=dict(family='arial',
                    size=18,
                    color='black'
                    )

        figresc['layout']['yaxis']['autorange'] = 'reversed'
        figresc.update_yaxes(title_text="<b>Magnitude<b>",
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

        figresc.update_xaxes(title_text="<b>log10 Time (s)<b>",
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

        figresc.update_layout(title="GRB " + self.name + " rescaled",
                        title_font_size=25,
                        font=font_dict,
                        plot_bgcolor='white',
                        width=960,
                        height=540,
                        margin=dict(l=40,r=40,t=50,b=40)
                        )

        # The definition of the rescaled dataframe
        # the list of values must be merged again in a new dataframe before exporting

        rescmagdataframe = pd.DataFrame()
        rescmagdataframe['time_sec'] = self.light['time_sec']
        rescmagdataframe['mag_rescaled_to_'+str(self.resc_band)] = self.light['mag_rescaled_to_'+self.resc_band]
        rescmagdataframe['mag_err'] = self.light['mag_rescaled_err']
        rescmagdataframe['band_given'] = self.light['band'] # the given band (not approximated is re-established) to allow a more precise photometric estimation if mags are converted into fluxes
        rescmagdataframe['system'] = self.light['system']
        rescmagdataframe['telescope'] = self.light['telescope']
        rescmagdataframe['extcorr'] = self.light['extcorr']
        rescmagdataframe['source'] = self.light['source']

        # The option for exporting the rescaled magnitudes as a dataframe
        if save:
            if not os.path.exists(save_in_folder):
                os.makedirs(save_in_folder)
            rescmagdataframe.to_csv(os.path.join(save_in_folder+'/' + str(self.name).split("/")[-1]+  '_rescaled_to_'+str(self.resc_band)+'.txt'),sep=' ',index=False)

        return figunresc, figresc, rescmagdataframe
