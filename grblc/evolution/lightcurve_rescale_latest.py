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

pd.set_option('display.max_rows', None)

# custom modules
from grblc.util import get_dir
from grblc.evolution.lightcurve_colorevo import Lightcurve as Lightcurve_colorevo
from . import io


class Lightcurve: # define the object Lightcurve
    _name_placeholder = "unknown grb" # assign the name for GRB if not provided
    _flux_fixed_inplace = False #


    def __init__(
        self,
        path: str = None,
        appx_bands: str = True, # if it is True it enables the approximation of bands, e.g. u' approximated to u,.....
        name: str = None,
    ):
        """The main module for fitting lightcurves.

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
        #assert bool(path) ^ (
        #    xdata is not None and ydata is not None
        #), "Either provide a path or xdata, ydata."


        # some default conditions for the name of GRBs and the path of the data file
        if name:
            self.name = name  # asserting the name of the GRB
        else:
            self.name = self._name_placeholder  # asserting the name of the GRB as 'Unknown GRB' if the name is not provided

        if isinstance(path, str):
            self.path = path  # asserting the path of the data file
            self.set_data(path, appx_bands=appx_bands, data_space='lin') # reading the data from a file


    def set_data(self, path: str, appx_bands=True, data_space='lin'):
        """
            Reads in data from a file. The data must be in the correct format.
            See the :py:meth:`io.read_data` for more information.

            Set the `xdata` and `ydata`, and optionally `xerr` and `yerr` of the lightcurve.

        .. warning::
            Data stored in :py:class:`lightcurve` objects are always in logarithmic
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

        df = io.read_data(path) # reads the data, sorts by time, excludes negative time

        df = df[df['mag_err'] != 0] # asserting those data points only which does not have limiting nagnitude
        assert len(df)!=0, "Only limiting magnitudes present."

        # converting the data here in the required format for color evolution analysis
        def convert_data(data):

            data = list(data) # reading the data as a list

            for i, band in enumerate(data):
                if band.lower() in ['clear', 'unfiltered', 'lum']:  # here it is checking for existence of the bands in lower case for three filters 'clear', 'unfiltered', 'lum'
                    band == band.lower()  # here it passes the lower case bands

            if appx_bands:  # here we reassigns the bands (reapproximation of the bands), e.g. u' reaasigned to u,.....
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


        self.xdata = df["time_sec"].to_numpy()  # passing the time in sec as a numpy array in the x column of the data
        self.ydata = df["mag"].to_numpy() # passing the magnitude as a numpy array in the y column of the data
        self.yerr = df["mag_err"].to_numpy()  # passing the magnitude error as an numpy array y error column of the data
        self.band_original = df["band"].to_list() # passing the original bands (befotre approximation of the bands) as a list
        self.band = df["band"] = convert_data(df["band"]) # passing the reassigned bands (after the reapproximation of the bands) as a list
        self.system = df["system"].to_list()  # passing the filter system as a list
        self.telescope = df["telescope"].to_list()  # passing the telescope name as a list
        self.extcorr = df["extcorr"].to_list()  # passing the galactic extinction correction detail (if it is corrected or not) as a list
        self.source = df["source"].to_list()  # passing the source from where the particular data point has been gathered as a list
        self.flag = df["flag"].to_list() # passing the flag for outliers from where the particular data point has been gathered as a list
        self.df = df  # passing the whole data as a data frame

    def displayGRB(self, save_static=False, save_static_type='.png', save_interactive=False, save_in_folder='plots/'):
        # This function plots the magnitudes, excluding the limiting magnitudes

        '''
        For an interactive plot
        '''

        fig = px.scatter(
                    x=self.xdata,
                    y=self.ydata,
                    error_y=self.yerr,
                    color=self.band,
                    hover_data=self.telescope,
                )

        font_dict=dict(family='arial',
                    size=26,
                    color='black'
                    )

        fig['layout']['yaxis']['autorange'] = 'reversed'
        fig.update_yaxes(title_text="<b>Magnitude<b>",
                        title_font_color='black',
                        title_font_size=25,
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

        fig.update_xaxes(title_text=r"$\log_{10} t\, (s)$",
                        title_font_color='black',
                        title_font_size=35,
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
            fig.write_image(save_in_folder+self.name+save_static_type)

        if save_interactive:
            fig.write_html(save_in_folder+self.name+'.html')

        return fig

    def rescaleGRB(self, output_colorevolGRB, chosenfilter='mostnumerous', save_rescaled_in='', duplicateremove=True): # this function makes the rescaling of the GRB

        # the global option is needed when these variables inputed in the current function are output of another function recalled, namely, colorevolGRB
        global filterforrescaling, light, overlap #, nocolorevolutionlist

        def overlap(mag1lower,mag1upper,mag2lower,mag2upper): # this is the condition to state if two magnitude ranges overlap
            if mag1upper <mag2lower or mag1lower > mag2upper:
                return 0 # in the case of no overlap, zero is returned
            else:
                return max(mag1upper, mag2upper) # in the case of overlap, a value>0 is returned

        def duplicate_remover(df):
            #freq_df = df['band_init'].value_counts()
            freq_df = pd.DataFrame(df['band_approx'].value_counts())
            freq_df.rename(columns = {'band_approx':'freq'}, inplace = True)
            time=df['time_sec']
            #df['logtime']=logtime
    
            for i,j in zip(df.index, df['band_approx']):
                df.loc[i, 'freq'] = freq_df.loc[j, 'freq']

            df = df.sort_values('freq', ascending=False).drop_duplicates(subset='time_sec', keep="first")

    
            df = df.sort_values('time_sec', ascending=True)

            #display(df)

            return df

        # here the code uses the colorevolGRB function defined above; the outputs of the function colorevolGRB will be used as input in the current function

        #output_colorevolGRB = self.colorevolGRB(print_status=False, return_rescaledf=False, save_plot=False, chosenfilter=chosenfilter, save_in_folder=save_rescaled_in)
        input = output_colorevolGRB
        nocolorevolutionlist = input[2] # 3rd output of colorevolGRB function, this is the list of filters whose resc.fact. slopes are compatible with zero in 3sigma or are < 0.10
        colorevolutionlist =input[3] # 4th output of colorevolGRB function, this is the list of filters whose resc.fact. slopes are incompatible with zero in 3sigma and are>0.10
        filterforrescaling = input[4] # 5th output of colorevolGRB function, this is the filter chosen for rescaling
        light = input[5] # 6th output of colorevolGRB function, is the original dataframe (since the filter chosen for rescaling is present here and it is needed)
        
        # Before rescaling the magnitudes, the following instructions plot the magnitudes in the unrescaled case
        
        color=light['band_approx'].tolist()
        filterslist=[[x,color.count(x)] for x in set(color)]

        freq=[]
        for k in color:
            for j in filterslist:
                if j[0]==k:
                    freq.append(str(k)+' ('+str(j[1])+')')

        figunresc = px.scatter(
                x=np.log10(light['time_sec'].values), # the time is set to log10(time) only in the plot frame
                y=light['mag'].values,
                error_y=light['mag_err'].values,
                #color=light['band'].values,
                color=freq,
                color_discrete_sequence=px.colors.qualitative.Set1,
                labels={"color": "      <b>Band<b>"}
                #hover_data=light['source'].values
                )

        figunresc.update_layout(legend=dict(font=dict(size=26)))

        font_dict=dict(family='arial', # font of the plot's layout
                    size=26,
                    color='black'
                    )

        figunresc.update_layout(legend_font_size=26, legend_font_color='black')
        figunresc.for_each_trace(lambda t: t.update(name = '<b>' + t.name +'</b>'))

        figunresc['layout']['yaxis']['autorange'] = 'reversed' # here the y axis is reversed, so that higher magnitudes are down and lower are up
        figunresc.update_yaxes(title_text="<b>Magnitude<b>", # updating plot options in the y-axes
                        title_font_color='black',
                        title_font_size=28,
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
                        tickprefix="<b>",
                        ticksuffix ="</b><br>"
                        )

        figunresc.update_xaxes(title_text="${\\bf\\huge log_{10} t\, (s)}$", # updating plot options in the x-axes
                        title_font_color='black',
                        title_font_size=35,
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
                        tickprefix="<b>",
                        ticksuffix ="</b><br>"
                        )

        #figunresc.update_layout(title="GRB " + self.name, # updating the layout of the plot
        #                title_font_size=25,
        #                font=font_dict,
        
        figunresc.update_layout(plot_bgcolor='white',
                        width=960,
                        height=540,
                        margin=dict(l=40,r=40,t=50,b=40)
                        )

        figunresc.add_annotation(
        text = ('<b>'+"GRB "+ self.name +'</b>')
        , showarrow=False
        , x = 0.08
        , y = 0.15
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'
        , xshift=-1
        , yshift=-5
        , font=dict(size=35, color="black")
        , align="left"
        ,)

        figunresc.update_traces(marker={'size': 9})

        figunresc.show()

        # Two additive columns must be inserted in the light dataframe

        light["mag_rescaled_to_"+filterforrescaling] = "" # the column with rescaled magnitudes
        light["mag_rescaled_err"] = "" # the column with magnitude errors, propagating the uncertainties on the magnitude itself and on the rescaling factor

        print(nocolorevolutionlist)

        for rr in light.index:
            # In these cases, the rescaled magnitude is the same of the original magnitude:
            # 1) The datapoint has the filter chosen for rescaling (obviously, can't be rescaled to itself)
            # 2) If the rescaling factor is not estimated (time difference > 2.5 percent)
            # 3) If the magnitudes of the filter chosen for rescaling and the filter to be rescaled overlap (mag overlap >0)
            # 4) If the filter belongs to the list of filters that have color evolution
            if light.loc[rr, "mag_chosenfilter"] == "-" or light.loc[rr, "mag_chosenfilter_err"] =="-":
                light.loc[rr, "mag_rescaled_to_"+filterforrescaling] = light.loc[rr, "mag"]
                light.loc[rr, "mag_rescaled_err"] = light.loc[rr, "mag_err"]
            else:
                magoverlapcheck=overlap(light.loc[rr, "mag"]-light.loc[rr, "mag_err"], # the overlap between the filter chosen for rescaling and the filter to be rescaled; zero if no overlap, number>0 if overlap
                                           light.loc[rr, "mag"]+light.loc[rr, "mag_err"],
                                           light.loc[rr, "mag_chosenfilter"]-light.loc[rr, "mag_chosenfilter_err"],
                                           light.loc[rr, "mag_chosenfilter"]+light.loc[rr, "mag_chosenfilter_err"])

            if (light.loc[rr, "band_approx"]==filterforrescaling) or (light.loc[rr, "resc_fact"] == "-") or (magoverlapcheck != 0) or (light.loc[rr, "band_approx"] in colorevolutionlist):
                light.loc[rr, "mag_rescaled_to_"+filterforrescaling] = light.loc[rr, "mag"]
                light.loc[rr, "mag_rescaled_err"] = light.loc[rr, "mag_err"]

            # In the following cases, magnitudes are rescaled
            # If the rescaling factor is estimated AND the time difference is smaller than 2.5 percent AND the overlap is zero AND the band is in the list of filters that have no color evolution
            elif (light.loc[rr, "resc_fact"] != "-") and (light.loc[rr, "time_difference_percentage"]<=0.025) and (magoverlapcheck==0) and (light.loc[rr, "band_approx"] in nocolorevolutionlist):
                light.loc[rr, "mag_rescaled_to_"+filterforrescaling] = light.loc[rr, "mag"]+light.loc[rr, "resc_fact"]
                light.loc[rr, "mag_rescaled_err"] = light.loc[rr, "resc_fact_err"] # the error on the rescaled magnitude is already estimated with the rescaling factor error

            # In all the other cases, the magnitudes can't be rescaled and the default options follow
            else:
                light.loc[rr, "mag_rescaled_to_"+filterforrescaling] = light.loc[rr, "mag"]
                light.loc[rr, "mag_rescaled_err"] = light.loc[rr, "mag_err"]

        if duplicateremove:

            light=duplicate_remover(light)

        color=light['band_approx'].tolist()
        filterslist=[[x,color.count(x)] for x in set(color)]

        freq=[]
        for k in color:
            for j in filterslist:
                if j[0]==k:
                    freq.append(str(k)+' ('+str(j[1])+')')

        # The plot of the rescaled dataframe
        figresc = px.scatter(
                x=np.log10(light["time_sec"].values), # the time is set to log10(time) only in the plot frame
                y=light["mag_rescaled_to_"+filterforrescaling].values,
                error_y=light["mag_rescaled_err"].values,
                #color=light["band_approx"],
                color=freq,
                color_discrete_sequence=px.colors.qualitative.Set1,
                labels={"color": "      <b>Band<b>"}
                )

        font_dict=dict(family='arial',
                    size=26,
                    color='black'
                    )

        #figresc.update_layout(legend=dict(font=dict(size=26)))

        figresc.update_layout(legend_font_size=26, legend_font_color='black')
        figresc.for_each_trace(lambda t: t.update(name = '<b>' + t.name +'</b>'))

        figresc['layout']['yaxis']['autorange'] = 'reversed'
        figresc.update_yaxes(title_text="<b>Magnitude<b>",
                        title_font_color='black',
                        title_font_size=28,
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
                        tickprefix="<b>",
                        ticksuffix ="</b><br>"
                        )

        figresc.update_xaxes(title_text="${\\bf\\huge log_{10} t\, (s)}$",
                        title_font_color='black',
                        title_font_size=35,
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
                        tickprefix="<b>",
                        ticksuffix ="</b><br>"
                        )

        #figresc.update_layout(title="GRB " + self.name + " rescaled",
        #                title_font_size=25,
        #                font=font_dict,
        
        figresc.update_layout(plot_bgcolor='white',
                        width=960,
                        height=540,
                        margin=dict(l=40,r=40,t=50,b=40)
                        )
        
        figresc.add_annotation(
        text = ('<b>'+"GRB " + self.name + "<br>    rescaled"+'</b>')
        , showarrow=False
        , x = 0.05
        , y = 0.12
        , xref='paper'
        , yref='paper' 
        , xanchor='left'
        , yanchor='bottom'
        , xshift=-1
        , yshift=-5
        , font=dict(size=35, color="black")
        , align="left"
        ,)

        figresc.update_traces(marker={'size': 9})

        figresc.show()

        # The definition of the rescaled dataframe
        # the list of values must be merged again in a new dataframe before exporting

        rescmagdataframe = pd.DataFrame()
        rescmagdataframe['time_sec'] = light['time_sec']
        rescmagdataframe['mag_rescaled_to_'+str(filterforrescaling)] = light['mag_rescaled_to_'+filterforrescaling]
        rescmagdataframe['mag_err'] = light['mag_rescaled_err']
        rescmagdataframe['band_original'] = light['band'] # the original band (not approximated is re-established) to allow a more precise photometric estimation if mags are converted into fluxes
        rescmagdataframe['system'] = light['system']
        rescmagdataframe['telescope'] = light['telescope']
        rescmagdataframe['extcorr'] = light['extcorr']
        rescmagdataframe['source'] = light['source']
        rescmagdataframe['flag'] = light['flag']

        # The option for exporting the rescaled magnitudes as a dataframe
        if not os.path.exists(save_rescaled_in):
            os.makedirs(save_rescaled_in)
        rescmagdataframe.to_csv(os.path.join(save_rescaled_in+'/' + str(self.name).split("/")[-1]+  '_rescaled_to_'+str(filterforrescaling)+'.txt'),sep=' ',index=False)

        return rescmagdataframe

major, *__ = sys.version_info # this command checks the Python version installed locally
readfile_kwargs = {"encoding": "utf-8"} if major >= 3 else {} # this option specifies the enconding of imported files in Python
                                                              # the encoding is utf-8 for Python versions superior to 3.
                                                              # otherwise it is left free to the code

def _readfile(path): # function for basic importation of text files, using the options defined in lines 1043,1044
    with open(path, **readfile_kwargs) as fp:
        contents = fp.read()
    return contents

# re.compile(): compile the regular expression specified by parenthesis to make it match
version_regex = re.compile('__version__ = "(.*?)"') #
contents = _readfile(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "__init__.py"
    )
) # this command reads __init__.py that gives the basic functions for the package, namely get_dir, set_dir
__version__ = version_regex.findall(contents)[0]

__directory__ = get_dir()