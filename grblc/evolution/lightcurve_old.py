# standard libs
import os
import re
import sys
from functools import reduce

# third party libs
import numpy as np
import pandas as pd
import lmfit as lf
from lmfit import Parameters,minimize,fit_report,Model
import math
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.ticker as mt
import matplotlib.font_manager as font_manager
import plotly.express as px
#from matplotlib.figure import Figure

pd.set_option('display.max_rows', None)

# custom modules
from grblc.util import get_dir
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

            #if appx_bands:  # here we reassigns the bands (reapproximation of the bands), e.g. u' reaasigned to u,.....
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
                if band.upper()=="CR":
                    data[i]="R"
                if band.upper()=="CLEAR":
                    data[i]="Clear"
                if band.upper()=="N":
                    data[i]="Unfiltered"
                if band.upper()=="UNFILTERED":
                    data[i]="Unfiltered"

            bands = data
            #else:
                #bands = data

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
        self.flag = df["flag"].to_list()
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
                    size=18,
                    color='black'
                    )

        fig['layout']['yaxis']['autorange'] = 'reversed'
        fig.update_yaxes(title_text="<b>Magnitude<b>",
                        title_font_color='black',
                        title_font_size=20,
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

        fig.update_xaxes(title_text=r"log_{10} t\, (s)",
                        title_font_color='black',
                        title_font_size=20,
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
                        fontdict=dict(weight='bold'),
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

    def colorevolGRB(self, print_status=True, return_rescaledf=False, save_plot=False, chosenfilter='mostnumerous', save_in_folder='', reportfill=False): #, rescaled_dfsave=False):

        # global nocolorevolutionlist, colorevolutionlist, light, lightonlyrescalable #, overlap
        # here the output variables are defined as global so that can be used in the functions
        # that recall the colorevolGRB() function

        light = pd.DataFrame() # here the original light curve dataframe is defined
        light['time_sec'] = self.xdata # time is linear
        light['mag'] = self.ydata
        light['mag_err'] = self.yerr
        light['band'] = self.band_original # here the band is the original one, not approximated
        light['band_approx'] = self.band # here the band is the approximated one, e.g., u' -> u, Ks -> K
        light['band_approx_occurrences'] = ""
        light['system'] = self.system
        light['telescope'] = self.telescope
        light['extcorr'] = self.extcorr
        light['source'] = self.source
        light['flag'] = self.flag
        light['resc_fact'] = "-"
        light['resc_fact_err'] = "-"
        light['time_difference'] = "-"
        light['mag_overlap'] = "-"
        light['mag_chosenfilter'] = "-" # magnitude of the filter chosen for rescaling (either the most numerous or another one)
        light['mag_chosenfilter_err'] = "-" # error on the magnitude of the filter chosen for rescaling


        light = light[(light['mag_err']!=0) & (light['time_sec']>0) & (light['flag']!="yes")] # here the code requires only magnitudes and not limiting magnitudes,
                                                                     # there are some transients observed in the optical before the
                                                                     # satellite trigger, thus they have negative times since in our
                                                                     # analysis, we consider the trigger time as start time of the LC
                                                                     # we furthermore exclude the datapoints that are outliers, namely, have the "yes" in the last column
                                                                     # we also exclude points with mag_err>0.5

        assert len(light)!=0, "The magnitude file has only limiting magnitudes." # assert is a command that verifies the condition written, if the condition
                                                              # doesn't hold then the error on the right is printed
        assert len(light)>1, "The magnitude file has only one data point."       # here we highlight if the dataframe has only limiting magnitudes
                                                              # or if it has only one data point

        occur = light['band_approx'].value_counts()    # this command returns a dataframe that contains in one column the
                                                # label of the filter and in another column the occurrences
                                                # Example: filter occurrences
                                                # R 18
                                                # I 6
                                                # B 5
                                                # F606W 4

        # In this loop, the column of band_approx_occurrences is filled with the countings of each filter
        # E.g. if the filter R is present 50 times in the magnitudes this will append 50 to the row where this filter is present

        for row in light.index:
            for ff in occur.index:
                if light.loc[row, "band_approx"]==ff:
                    light.loc[row, "band_approx_occurrences"]=occur[ff]

        # Identifying the most numerous filter in the GRB

        assert chosenfilter == 'mostnumerous' or chosenfilter in self.band, "Rescaling band provided as <<chosenfilter>> is not present in data. Check the filters present in the magnitudes file."

        # chosenfilter is an input of the function colorevolGRB(...)
        # This assert condition is needed to verify that the filter is present in the LC; for example, if "g" is selected but it's
        # not present then the string "Rescaling..." is printed

        if chosenfilter == 'mostnumerous':          # here I select by default the filterforrescaling as the most numerous inside the LC
                                                    # if chosenfilter input is 'mostnumerous', then I automatically take the most numerous
            filterforrescaling = occur.index[0]     # namely, the first element in the occur frame (with index 0, since in Python the counting starts from zero)
            filteroccurrences = occur[0]     # this is the number of occurrences of the filterforrescaling
        else:
            for ii in occur.index:                  # if the input chosenfilter of the function is a filter label different from 'mostnumerous'
                if ii==chosenfilter:                # with this loop, the input variable chosenfilter is matched with the list of filters called "occur"
                    filterforrescaling = ii
                    filteroccurrences = occur[ii]

        if print_status:                            # the print_status option is set to true by default, and it prints
            print(self.name)                        # the GRB name
            print('\n')
            print('-------')                        # and the details of the filter chosen for rescaling, name + occurrences
            print(occur)
            print('\n The filter chosen in this GRB: ',filterforrescaling,', with', filteroccurrences, 'occurrences.\n'+
                'This filter will be considered for rescaling')

        # In the following rows the code extracts only the datapoints with the filter chosen for rescaling (usually, the most numerous)

        mostnumerouslight=light.loc[(light['band_approx'] == filterforrescaling)] # mostnumerouslight dataframe is the one constituted of the chosen filter for rescaling,
        mostnumerousx=mostnumerouslight['time_sec'].values                   # for simplicity it is called mostnumerouslight
        mostnumerousy=mostnumerouslight['mag'].values                        # time_sec is linear
        mostnumerousyerr=mostnumerouslight['mag_err'].values

        # The following loop fills the columns of the general dataframe with the rescaling factor, its error, the time difference
        # between the filter in the dataframe and the filter chosen for rescaling, and the magnitude overlap (if overlap is zero,
        # then the magnitude values do not overlap)

        for row in light.index: # running on all the magnitudes
            if light.loc[row, "band_approx"]==filterforrescaling:   # when the filter in the dataframe is the filter chosen for rescaling,
                light.loc[row, "resc_fact"]="-"                     # the rescaling factor is obviously not existing and the columns are filled with "-"
                light.loc[row, "resc_fact_err"]="-"
                light.loc[row, "time_difference_percentage"]="-"
                light.loc[row, "mag_chosenfilter"]="-"
                light.loc[row, "mag_chosenfilter_err"]="-"
            else:
                compatiblerescalingfactors=[] # the compatiblerescalingfactors is a list that contains all the possible rescaling factors for a magnitude, since more of them can fall in the 2.5 percent criteria
                for pp in range(len(mostnumerouslight)): # running on the magnitudes of the filter chosen for rescaling
                    if np.abs(mostnumerousx[pp]-light.loc[row, "time_sec"])<=(0.025*mostnumerousx[pp]): # if the filter chosen for rescaling is in the 2.5 percent time condition with the magnitude of the loop
                        rescfact=mostnumerousy[pp]-light.loc[row, "mag"] # this is the rescaling factor, mag_filterforrescaling - mag_filter
                        rescfacterr=np.sqrt(mostnumerousyerr[pp]**2+light.loc[row, "mag_err"]**2) # rescaling factor error, propagation of uncertainties on both the magnitudes
                        timediff=np.abs(mostnumerousx[pp]-light.loc[row, "time_sec"])/mostnumerousx[pp] # linear time difference between the time of filter chosen for rescaling and the filter to be rescaled, divided by the chosen filter time
                        magchosenf=mostnumerousy[pp]
                        magchosenferr=mostnumerousyerr[pp]
                        compatiblerescalingfactors.append([rescfact,rescfacterr,timediff,magchosenf,magchosenferr]) # all these values are appended in the compatiblerescalingfactors list, since in principle there may be more than one for a single datapoint
                    if len(compatiblerescalingfactors)==0: # if there are no rescaling factors that respect the 2.5 percent for the given filter, then once again the columns are filled with "-"
                        light.loc[row, "resc_fact"]="-"
                        light.loc[row, "resc_fact_err"]="-"
                        light.loc[row, "time_difference_percentage"]="-"
                        light.loc[row, "mag_chosenfilter"]="-"
                        light.loc[row, "mag_chosenfilter_err"]="-"
                        continue
                    else: # in the other cases
                        minimumtimediff=min([k[2] for k in compatiblerescalingfactors]) # the rescaling factor with the minimum time difference between the filter chosen for rescaling and the filter to be rescaled is taken
                        acceptedrescalingfactor=list(filter(lambda x: x[2] == minimumtimediff, compatiblerescalingfactors)) # this line locates the rescaling factor with minimum time difference
                        light.loc[row, "resc_fact"]=acceptedrescalingfactor[0][0]
                        light.loc[row, "resc_fact_err"]=acceptedrescalingfactor[0][1]
                        light.loc[row, "time_difference_percentage"]=acceptedrescalingfactor[0][2]
                        light.loc[row, "mag_chosenfilter"]=acceptedrescalingfactor[0][3]
                        light.loc[row, "mag_chosenfilter_err"]=acceptedrescalingfactor[0][4]

        #display(light)

        lightonlyrescalable=light[light["resc_fact"]!='-'] # the lightonlyrescalable selects only the datapoints with rescaling factors

        filt=[] # these are the empty lists to be filled with filter
        filtoccur=[] # occurrences of the filter
        resclogtime=[] #log10(time) of the rescaling factor for the filter
        rescfact=[] # rescaling factor of the filter
        rescfacterr=[] # rescaling factor error of the filter
        rescfactweights=[] # weights of the rescaling factor
        for jj in lightonlyrescalable.index:
            filt.append(lightonlyrescalable.loc[jj, "band_approx"]) # here we have the filters that are rescaled to the selected filter for rescaling
            filtoccur.append(lightonlyrescalable.loc[jj, "band_approx_occurrences"]) # here we have the occurrences of the filters
            resclogtime.append(np.log10(lightonlyrescalable.loc[jj, "time_sec"])) # WATCH OUT! For the plot and fitting, we take the log10(time) of rescaling factor
            rescfact.append(light.loc[jj, "resc_fact"]) # The rescaling factor value
            rescfacterr.append(light.loc[jj, "resc_fact_err"]) # The rescaling factor error
            rescfactweights.append((1/light.loc[jj, "resc_fact_err"])) # The weights on the rescaling factor

            # The weights are considered as 1/yerr given that the lmfit package will be used: https://lmfit.github.io/lmfit-py/model.html#the-model-class

        # The following command defines the dataframe of rescaling factors

        rescale_df=pd.DataFrame(list(zip(filt,filtoccur,resclogtime,rescfact,
                                                    rescfacterr,rescfactweights)),columns=['band','Occur_band','Log10(t)','Resc_fact','Resc_fact_err','Resc_fact_weights'])

        x_all = rescale_df['Log10(t)']  # list of log10 times for the rescaling factors
        y_all = rescale_df['Resc_fact'] # list of the rescaling factors
        yerr_all = rescale_df['Resc_fact_err'] # list of the rescaling factors errors
        filters = [*set(rescale_df['band'].values)] # list of filters in the rescaling factors sample
        rescale_df['plot_color'] = "" # empty list that will filled with the color map condition

        # Set the color map to match the number of filter
        cmap = plt.get_cmap('gist_ncar') # import the color map
        cNorm  = colors.Normalize(vmin=0, vmax=len(filters)) # linear map of the colors in the colormap from data values vmin to vmax
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap) # The ScalarMappable applies data normalization before returning RGBA colors from the given colormap

        # Plot each filter
        fig, axs = plt.subplots(2, 1, sharex=True)
        #fig = plt.figure()

        for i, band in enumerate(filters): # loop on the given filter
            colour = scalarMap.to_rgba(i) # mapping the colour into the RGBA
            #print(colour,band)
            index = rescale_df['band'] == band # selects the magnitudes that have the filter equal to the band on which the loop iterates
            axs[1].scatter(x_all[index], y_all[index], # options for the plot of the central values
                        s=15,
                        color=colour) # color-coding of the plot
            axs[1].errorbar(x_all[index], y_all[index], yerr_all[index], #options for the plot of the error bars, these must be added in this command
                        fmt='o', # this is the data marker, a circle
                        barsabove=True, # bars plotted above the data marker
                        ls='', # line style = None, so no lines are drawn between the points (later the fit will be done and plotted)
                        color=colour # color-coding
                        )
            for j in rescale_df[index].index:
                rescale_df.at[j,"plot_color"] = colour # this loop assigns each filter to a color in the plot

        axs[0].text(0.1, 0.1, "GRB "+(self.name.split("/")[-1]).split("_")[0], fontsize=22, fontweight='bold', horizontalalignment='left', verticalalignment='bottom', transform=axs[0].transAxes)

        # 0.1, 0.1 or 0.45,s 0.65

        resc_slopes_df = pd.DataFrame() # initialize of the rescaling factors fitting dataframe
        resc_slopes_df.index = filters # the filters are taken as index            
        resc_slopes_df['slope_lin'] = "" # placeholder, default set to empty, then it will change - slope of the linear fit
        resc_slopes_df['slope_lin_err'] = "" # placeholder, default set to empty, then it will change - error on slope
        resc_slopes_df['intercept'] = "" # placeholder, default set to empty, then it will change - intercept of linear fit
        resc_slopes_df['inter_err'] = "" # placeholder, default set to empty, then it will change - error on intercept
        resc_slopes_df['slope_lin_err/slope_lin'] = "" # placeholder, default set to empty, then it will change - slope_err/slope = |slope_err|/|slope|
        resc_slopes_df['red_chi2_lin'] = "" # placeholder, default set to empty, then it will change - reduced chi^2 of linear fit
        resc_slopes_df['prob_lin'] = "" # placeholder, default set to empty, then it will change - probability of linear fit   
        resc_slopes_df['BIC_lin'] = "" # placeholder, default set to empty, then it will change - gives the BIC value for the variable a fitting
        resc_slopes_df['comment_lin'] = "" # placeholder, default set to empty, then it will change - the comment that will say "no color evolution","color evolution" considering the linear fitting
        resc_slopes_df['plot_color'] = "" # placeholder, default set to empty, then it will change - color-coding for the fitting lines
        resc_slopes_df['intercept_a0'] = "" # placeholder, default set to empty, then it will change - intercept of fit with a=0
        resc_slopes_df['inter_a0_err'] = "" # placeholder, default set to empty, then it will change - error on intercept with a=0
        resc_slopes_df['intercept_a0_err/inter_a0'] = "" # placeholder, default set to empty, then it will change - intercept_err/intercept = |intercept_err|/|intercept| with a=0
        resc_slopes_df['red_chi2_a0'] = "" # placeholder, default set to empty, then it will change - reduced chi^2 of fit with a=0
        resc_slopes_df['prob_a0'] = "" # placeholder, default set to empty, then it will change - probability of linear fit with a=0         
        resc_slopes_df['BIC_a0'] = "" # placeholder, default set to empty, then it will change - gives the BIC value for the a=0 fitting
        resc_slopes_df['comment_a0'] = "" # placeholder, default set to empty, then it will change - the comment that will say "no color evolution","color evolution" considering the a=0 fitting

        for band in resc_slopes_df.index: # in this loop, we assign the bands in the dataframe defined in line 580
            ind = rescale_df.index[rescale_df['band'] == band][0]
            resc_slopes_df.loc[band, "plot_color"] = str(rescale_df.loc[ind, "plot_color"])
            resc_band_df = rescale_df[rescale_df['band'] == band]

            x = resc_band_df['Log10(t)'] # we here define the dataframe to fit, log10(time)
            y = resc_band_df['Resc_fact'] # the rescaling factors
            weights = resc_band_df['Resc_fact_weights'] # the rescaling factors weights

            ## lmfit linear - lmfit is imported as "lf" -> the lmfit uses the Levenberg-Marquardt method
            # https://lmfit.github.io/lmfit-py/model.html#the-model-class

            if len(x) >= 3: # the fitting will be performed if and only if, for the given filter, at least 3 rescaling factors are available
                linear_model = lf.models.LinearModel(prefix='line_') # importing linear model from lmfit
                linear_params = linear_model.make_params() # we here initialize the fitting parameters, then these will be changed

                linear_params['line_slope'].set(value=-1.0) # initializing the fitting slope
                linear_params['line_intercept'].set(value=np.max(y)) # initializing the fitting intercept

                linear_fit = linear_model.fit(y, params=linear_params, x=x, weights=weights) # the command for weighted lmfit
                resc_slopes_df.loc[band, 'slope_lin'] = np.around(linear_fit.params['line_slope'].value, decimals=3) # slope of the fitting
                resc_slopes_df.loc[band, 'slope_lin_err'] = np.around(linear_fit.params['line_slope'].stderr, decimals=3) # slope error
                resc_slopes_df.loc[band, 'intercept'] = np.around(linear_fit.params['line_intercept'].value, decimals=3) # intercept
                resc_slopes_df.loc[band, 'inter_err'] = np.around(linear_fit.params['line_intercept'].stderr, decimals=3) # intercept error
                resc_slopes_df.loc[band, 'slope_lin_err/slope_lin'] = np.around(np.abs(np.around(linear_fit.params['line_slope'].stderr, decimals=3) /np.around(linear_fit.params['line_slope'].value, decimals=3)), decimals=3) # slope_err/slope = |slope_err|/|slope|
                resc_slopes_df.loc[band, 'red_chi2_lin'] = np.around(linear_fit.redchi, decimals=3) # reduced chi^2

                nulinfit=len(x)
                xxlinfit=linear_fit.redchi*nulinfit
                
                try:
                    problin=(2**(-nulinfit/2)/math.gamma(nulinfit/2))*scipy.integrate.quad(lambda x: math.exp(-x/2)*x**(-1+(nulinfit/2)),xxlinfit,np.inf)[0]
                except:
                    problin="NaN"

                resc_slopes_df.loc[band, 'prob_lin'] = problin # probability of the linear fit
                resc_slopes_df.loc[band, 'BIC_lin'] = np.around(linear_fit.bic, decimals=3) # BIC value for the linear fitting (variable a)

                def slopezero(m,c,x):
                    #m = slopezero_params['m']
                    #c = slopezero_params['c']
                    y_fit = m*x + c
                    return y_fit

                # Defining the parameters
                slopezero_params = Parameters()
                # Slope is set to zero and can't be varied
                slopezero_params.add('m', value=0.0, vary=False)
                # Intercept can be varied between very large boundaries
                slopezero_params.add('c', min=-1000, max=1000)

                modelslopezero = Model(slopezero, independent_vars=['x'])
                slopezero_fit = modelslopezero.fit(y, params=slopezero_params, x=x, weights=weights)   

                # slopezero_model = lf.models.LinearModel(prefix='line_') # importing linear model from lmfit
                # slopezero_params = slopezero_model.make_params() # we here initialize the fitting parameters, then these will be changed

                # slopezero_params['line_slope'].set(value=0, vary=False) # initializing the fitting slope
                # slopezero_params['line_intercept'].set(value=np.max(y)) # initializing the fitting intercept

                # print("TEST"+'\n')
                # print(slopezero_fit.params['line_intercept'].value, decimals=3)

                resc_slopes_df.loc[band, 'intercept_a0'] = np.around(slopezero_fit.params['c'].value, decimals=3) # intercept with a=0
                resc_slopes_df.loc[band, 'inter_a0_err'] = np.around(slopezero_fit.params['c'].stderr, decimals=3) # intercept error with a=0
                resc_slopes_df.loc[band, 'intercept_a0_err/inter_a0'] = np.around(np.abs(np.around(slopezero_fit.params['c'].stderr, decimals=3) /np.around(slopezero_fit.params['c'].value, decimals=3)), decimals=3) # intercept_err/intercept = |intercept_err|/|intercept| when slope (a) =0
                resc_slopes_df.loc[band, 'red_chi2_a0'] = np.around(slopezero_fit.redchi, decimals=3) # reduced chi^2 for a=0

                nuzeroslopefit=len(x)
                xxzeroslopefit=slopezero_fit.redchi*nuzeroslopefit
                
                try:
                    probzeroslope=(2**(-nuzeroslopefit/2)/math.gamma(nuzeroslopefit/2))*scipy.integrate.quad(lambda x: math.exp(-x/2)*x**(-1+(nuzeroslopefit/2)),xxzeroslopefit,np.inf)[0]
                except:
                    probzeroslope="NaN"

                resc_slopes_df.loc[band, 'prob_a0'] = probzeroslope # probability of the zero slope fit
                resc_slopes_df.loc[band, 'BIC_a0'] = np.around(slopezero_fit.bic, decimals=3) # BIC value for the linear fitting (a=0)

                if np.abs(resc_slopes_df.loc[band, 'slope_lin']) < 0.001: # in case of zero-slope, since the fittings have 3 digits precision we consider the precision 0.001 as the "smallest value different from zero"
                    
                    if np.abs(resc_slopes_df.loc[band, 'slope_lin_err/slope_lin']) <= 1000:
                        y_fit = resc_slopes_df.loc[band, 'slope_lin'] * x + resc_slopes_df.loc[band, 'intercept']
                        axs[1].plot(x, y_fit, color=tuple(np.array(re.split('[(),]', resc_slopes_df.loc[band, "plot_color"])[1:-1], dtype=float))) # plot of the fitting line between log10(t) and resc_fact

                        resc_slopes_df.loc[band, 'comment_lin'] = "no-color-evolution"

                    else:
                        resc_slopes_df.loc[band, 'comment_lin'] = "slope_err/slope>1000"

                if np.abs(resc_slopes_df.loc[band, 'slope_lin']) >= 0.001: # in the case of non-zero slope
                    
                    if np.abs(resc_slopes_df.loc[band, 'slope_lin_err/slope_lin']) <= 1000: # this boundary of slope_err/slope is put ad-hoc to show all the plots
                                                                    # it's a large number that can be modified
                        y_fit = resc_slopes_df.loc[band, 'slope_lin'] * x + resc_slopes_df.loc[band, 'intercept'] # fitted y-value according to linear model
                        axs[1].plot(x, y_fit, color=tuple(np.array(re.split('[(),]', resc_slopes_df.loc[band, "plot_color"])[1:-1], dtype=float))) # plot of the fitting line between log10(t) and resc_fact

                        if np.abs(3*resc_slopes_df.loc[band, 'slope_lin_err'])>=np.abs(resc_slopes_df.loc[band, 'slope_lin']):
                        #if resc_slopes_df.loc[band, 'slope_lin']-(3*resc_slopes_df.loc[band, 'slope_lin_err'])<=0<=resc_slopes_df.loc[band, 'slope_lin']+(3*resc_slopes_df.loc[band, 'slope_lin_err']):
                            resc_slopes_df.loc[band, 'comment_lin'] = "no-color-evolution" # in case it is comp. with zero in 3 sigma, there is no color evolution
                        else:
                            resc_slopes_df.loc[band, 'comment_lin'] = "color-evolution" # in case the slope is not compatible with zero in 3 sigma

                    else:
                        resc_slopes_df.loc[band, 'comment_lin'] = "slope_err/slope>10sigma"  # when the slope_err/slope is very high


                if resc_slopes_df.loc[band, 'prob_a0'] >= 0.05:
                    resc_slopes_df.loc[band, 'comment_a0'] = "no-color-evolution"
                else:
                    resc_slopes_df.loc[band, 'comment_a0'] = "color-evolution"  
                          
                if resc_slopes_df.loc[band, 'slope_lin_err/slope_lin'] > 1:
                    resc_slopes_df.loc[band, 'comment_lin'] = 'undetermined'

            else: # not enough data points, less than 3 rescaling factors for the filter
                resc_slopes_df.loc[band, 'slope_lin'] = np.nan
                resc_slopes_df.loc[band, 'slope_lin_err'] = np.nan
                resc_slopes_df.loc[band, 'intercept'] = np.nan
                resc_slopes_df.loc[band, 'inter_err'] = np.nan
                resc_slopes_df.loc[band, 'slope_lin_err/slope_lin'] = np.nan
                resc_slopes_df.loc[band, 'red_chi2_lin'] = 'insufficient_data'
                resc_slopes_df.loc[band, 'prob_lin'] = 'insufficient_data'
                resc_slopes_df.loc[band, 'BIC_lin'] = "insufficient_data"
                resc_slopes_df.loc[band, 'comment_lin'] = "insufficient_data"
                resc_slopes_df.loc[band, 'plot_color'] = np.nan 
                resc_slopes_df.loc[band, 'intercept_a0'] = np.nan
                resc_slopes_df.loc[band, 'inter_a0_err'] = np.nan
                resc_slopes_df.loc[band, 'intercept_a0_err/inter_a0'] = np.nan
                resc_slopes_df.loc[band, 'red_chi2_a0'] = 'insufficient_data'
                resc_slopes_df.loc[band, 'prob_a0'] = 'insufficient_data'  
                resc_slopes_df.loc[band, 'BIC_a0'] = 'insufficient_data'
                resc_slopes_df.loc[band, 'comment_a0'] = 'insufficient_data'

        for band in resc_slopes_df.index: # this loop defines the labels to be put in the rescaling factor plot legend

            ind = rescale_df.index[rescale_df['band'] == band][0]
            resc_slopes_df.loc[band, "plot_color"] = str(rescale_df.loc[ind, "plot_color"])
            resc_band_df = rescale_df[rescale_df['band'] == band]

            x = resc_band_df['Log10(t)'] # we here define the dataframe to fit, log10(time)

            if np.isnan(resc_slopes_df.loc[band, "slope_lin"])==True: # in case the fitting is not done, the label will be "filter: no fitting"
                if len(x)==1:
                    label=band+": "+str(len(x))+" data point"
                else:
                    label=band+": "+str(len(x))+" data points"
            else:
                label=band+": rf="+ r'('+ str(resc_slopes_df.loc[band, "slope_lin"]) + r'$\pm$' + str(resc_slopes_df.loc[band, "slope_lin_err"]) + ')' + r'$ *log_{10}(t)$' + '+ (' + str(resc_slopes_df.loc[band, "intercept"]) + r'$\pm$' + str(resc_slopes_df.loc[band, "inter_err"]) + r')'
                # when the slopes are estimated, the label is "filter: slope +/- slope_err"

            ind = rescale_df.index[rescale_df['band'] == band][0] # initializing the variables to be plotted for each filter
            color = rescale_df.loc[ind, "plot_color"]
            axs[1].scatter(x=[], y=[],
                        color=color,
                        label=label # here the labels for each filter are inserted
                        )

        axs[0].scatter(np.log10(mostnumerousx), mostnumerousy, marker='D', c='k', label=filterforrescaling+": used for rescaling")
        axs[0].errorbar(np.log10(mostnumerousx), mostnumerousy, yerr=mostnumerousyerr, c='k', ls='')

        for band in resc_slopes_df.index:
            #color = resc_slopes_df.loc[band]["plot_color"]
            color=tuple(np.array(re.split('[(),]', resc_slopes_df.loc[band, "plot_color"])[1:-1], dtype=float))
            #print(color)
            sublight=light.loc[(light['band_approx'] == band)]
            subx=np.log10(sublight['time_sec'].values)
            suby=sublight['mag'].values
            suberror_y=sublight['mag_err'].values
            axs[0].scatter(subx, suby, color=color)
            axs[0].errorbar(x=subx, y=suby, yerr=suberror_y, color=color, ls='')

        # Here the code prints the dataframe of rescaling factors, that contains log10(time), slope, slope_err...
        # rescale_df.drop(labels='plot_color', axis=1, inplace=True)     # before printing that dataframe, the code removes the columns of plot_color
        # resc_slopes_df.drop(labels='plot_color', axis=1, inplace=True) # since this column was needed only for assigning the plot colors
                                                                       # these columns have no scientific meaning

        if print_status: # when this option is selected in the function it prints the following

            print("Individual point rescaling:")
            print(rescale_df) # the dataframe of rescaling factors

            print("\nSlopes of rescale factors for each filter:")
            print(resc_slopes_df) # the dataframe that contains the fitting parameters of rescaling factors

        compatibilitylist=[] # here we initialize the list that contains the ranges of (slope-3sigma,slope+3sigma) for each filter

        for band in resc_slopes_df.index: # this code appends the values of (slope-3sigma,slope+3sigma) in case the slope is not a "nan"
                                          # and in case both the slope and slope_err are different from zero
            if resc_slopes_df.loc[band, 'slope_lin']!=0 and resc_slopes_df.loc[band, 'slope_lin_err']!=0 and np.isnan(resc_slopes_df.loc[band, 'slope_lin'])==False and np.isnan(resc_slopes_df.loc[band, 'slope_lin_err'])==False and np.abs(resc_slopes_df.loc[band, 'slope_lin_err/slope_lin'])<=1:
                compatibilitylist.append([band,[resc_slopes_df.loc[band, 'slope_lin']-(3*resc_slopes_df.loc[band, 'slope_lin_err']),
                                        resc_slopes_df.loc[band, 'slope_lin']+(3*resc_slopes_df.loc[band, 'slope_lin_err'])]])

        nocolorevolutionlist=[] # this is the list of filters with slopes that are compatible with zero in 3 sigma
        colorevolutionlist=[] # this is the list of filters with slopes that ARE NOT compatible with zero in 3 sigma
        for l in compatibilitylist:
            if l[1][0]<=0<=l[1][1]: # if slope-3sigma<=0<=slope+3sigma (namely, compat. with zero in 3sigma)
                    nocolorevolutionlist.append(l[0])                           # then for the given filter the slope is compatible with zero in 3sigma, NO COLOR EVOLUTION
            else:
                colorevolutionlist.append(l[0]) # in the other cases, slopes are not compatible with zero in 3 sigma, COLOR EVOLUTION

        if print_status:
            if len(nocolorevolutionlist)==0: # if there are no filters compatible with zero in 3 sigma
                print('No filters compatible with zero in 3sigma')

            else:
                print('Filters compatible with zero in 3sigma and such that |slope_err/slope| <=1 : ',*nocolorevolutionlist) # if there are filters without color evolution, namely, compatible with zero in 3 sigma

            if len(colorevolutionlist)==0: # if there are not filters with color evolution, namely, that are not compatible with zero in 3 sigma
                print('No filters compatible with zero in >3sigma')

            else: # otherwise
                print('Filters not compatible with zero in 3sigma: ',*colorevolutionlist)
            print('\n')
            print('No color evolution: ',*nocolorevolutionlist,' ; Color evolution: ',*colorevolutionlist) # print of the two lists


        string="" # this is the general printing of all the slopes
        for band in resc_slopes_df.index:
            string=string+band+":"+str(round(resc_slopes_df.loc[band, 'slope_lin'],3))+"+/-"+str(round(resc_slopes_df.loc[band, 'slope_lin_err'],3))+"; "

        print(string)

        fmt = '%.1f' # set the number of significant digits you want
        xyticks = mt.FormatStrFormatter(fmt)
        for ax in axs:
            ax.yaxis.set_major_formatter(xyticks)
            ax.xaxis.set_major_formatter(xyticks)

        #axs[1].rcParams['legend.title_fontsize'] = 'xx-large'
        #axs[1].title('Rescaling factors for '+ str(grbtitle),fontsize=20)
        axs[0].invert_yaxis()
        #axs[0].set_xlabel('Log time (s)',fontsize=22)
        axs[0].set_ylabel('Magnitude', labelpad=15, fontsize=16, fontdict=dict(weight='bold'))
        axs[0].tick_params(labelsize=16, direction='in', width=2)
        axs[0].locator_params(axis='x', nbins=5)
        axs[0].locator_params(axis='y', nbins=5)

        for tick in axs[0].get_xticklabels():
            tick.set_fontweight('bold')
        for tick in axs[0].get_yticklabels():
            tick.set_fontweight('bold')

        for axis in ['top', 'bottom', 'left', 'right']:

            axs[0].spines[axis].set_linewidth(2.2)  # change width  

        
        #axs[1].title('Rescaling factors for '+ str(grbtitle),fontsize=20)
        axs[1].set_xlabel(r"$\bf{log_{10} t\, (s)}$", fontsize=17, fontdict=dict(weight='bold'))
        axs[1].set_ylabel('Resc. fact. (rf)', labelpad=15, fontsize=17, fontdict=dict(weight='bold')) #+filterforrescaling, labelpad=15, fontsize=15)
        axs[1].tick_params(labelsize=16, direction='in', width=2)
        
        for tick in axs[1].get_xticklabels():
            tick.set_fontweight('bold')
        for tick in axs[1].get_yticklabels():
            tick.set_fontweight('bold')
        
        for axis in ['top', 'bottom', 'left', 'right']:

            axs[1].spines[axis].set_linewidth(2.2)  # change width

        font = font_manager.FontProperties(weight='bold', style='normal', size=20) # 20

        fig.legend(title="$\\bf{Band: rf=(a \pm \sigma_{a}) *\\log_{10}(t) + (b \pm \sigma_{b})}$", bbox_to_anchor=(1, 1), loc='upper left', fontsize="20", title_fontsize="20", prop=font) # 21 21

        #fig.legend(title='Band: slope±err', bbox_to_anchor=(1, 0.946), loc='upper left', fontsize="21", title_fontsize="21")
        #plt.rcParams['legend.title_fontsize'] = 'x-large'
        plt.rcParams['figure.figsize'] = [16, 9] #15,8 16,9
        #plt.rcParams.update({'legend.fontsize': 14})
        plt.tight_layout()
        
        #plt.savefig("new-plots/"+str(grbtitle)+"_lc_colorevol.pdf", bbox_inches='tight') #DECOMMENT TO EXPORT THE PLOTS
        
        if save_plot:
            #plt.savefig(str(self.name)+'_colorevol.pdf', dpi=300) # option to export the pdf plot of rescaling factors
            plt.savefig(os.path.join(save_in_folder+'/'+str(self.name.split("/")[-1])+'_colorevol.pdf'), bbox_inches='tight', dpi=300) # option to export the pdf plot of rescaling factors        
        
        plt.show()

        plt.clf()

        ############################################## Plotting the case where a=0 ############################################

        fig2, axs2 = plt.subplots(2, 1, sharex=True)
        #fig = plt.figure()

        for i, band in enumerate(filters): # loop on the given filter
            colour = scalarMap.to_rgba(i) # mapping the colour into the RGBA
            #print(colour,band)
            index = rescale_df['band'] == band # selects the magnitudes that have the filter equal to the band on which the loop iterates
            axs2[1].scatter(x_all[index], y_all[index], # options for the plot of the central values
                        s=15,
                        color=colour) # color-coding of the plot
            axs2[1].errorbar(x_all[index], y_all[index], yerr_all[index], #options for the plot of the error bars, these must be added in this command
                        fmt='o', # this is the data marker, a circle
                        barsabove=True, # bars plotted above the data marker
                        ls='', # line style = None, so no lines are drawn between the points (later the fit will be done and plotted)
                        color=colour # color-coding
                        )
            for j in rescale_df[index].index:
                rescale_df.at[j,"plot_color"] = colour # this loop assigns each filter to a color in the plot

        axs2[0].text(0.1, 0.1, "GRB "+(self.name.split("/")[-1]).split("_")[0], fontsize=22, fontweight='bold', horizontalalignment='left', verticalalignment='bottom', transform=axs2[0].transAxes)


        for band in resc_slopes_df.index: # this loop defines the labels to be put in the rescaling factor plot legend

            ind = rescale_df.index[rescale_df['band'] == band][0]
            resc_slopes_df.loc[band, "plot_color"] = str(rescale_df.loc[ind, "plot_color"])
            resc_band_df = rescale_df[rescale_df['band'] == band]

            x = resc_band_df['Log10(t)'] # we here define the dataframe to fit, log10(time)

            if np.isnan(resc_slopes_df.loc[band, "intercept_a0"])==True: # in case the fitting is not done, the label will be "filter: no fitting"
                if len(x)==1:
                    label=band+": "+str(len(x))+" data point"
                else:
                    label=band+": "+str(len(x))+" data points"
            else:
                label=band+": rf=(a=0)"+ r'$*log_{10}(t)$' + '+ (' + str(resc_slopes_df.loc[band, "intercept_a0"]) + r'$\pm$' + str(resc_slopes_df.loc[band, "inter_a0_err"]) + r')'
                # when the slopes are estimated, the label is "filter: slope +/- slope_err"

            ind = rescale_df.index[rescale_df['band'] == band][0] # initializing the variables to be plotted for each filter
            color = rescale_df.loc[ind, "plot_color"]
            axs2[1].scatter(x=[], y=[],
                        color=color,
                        label=label # here the labels for each filter are inserted
                        )

        axs2[0].scatter(np.log10(mostnumerousx), mostnumerousy, marker='D', c='k', label=filterforrescaling+": used for rescaling")
        axs2[0].errorbar(np.log10(mostnumerousx), mostnumerousy, yerr=mostnumerousyerr, c='k', ls='')

        for band in resc_slopes_df.index:
            #color = resc_slopes_df.loc[band]["plot_color"]
            color=tuple(np.array(re.split('[(),]', resc_slopes_df.loc[band, "plot_color"])[1:-1], dtype=float))
            #print(color)
            sublight=light.loc[(light['band_approx'] == band)]
            subx=np.log10(sublight['time_sec'].values)
            suby=sublight['mag'].values
            suberror_y=sublight['mag_err'].values
            axs2[0].scatter(subx, suby, color=color)
            axs2[0].errorbar(x=subx, y=suby, yerr=suberror_y, color=color, ls='')

            if not np.isnan(resc_slopes_df.loc[band, 'intercept_a0']):
                #print(resc_slopes_df.loc[band, 'intercept_a0'])
                y_fit = (subx*0) + resc_slopes_df.loc[band, 'intercept_a0'] # fitted y-value according to linear model
                axs2[1].plot(subx, y_fit, color=tuple(np.array(re.split('[(),]', resc_slopes_df.loc[band, "plot_color"])[1:-1], dtype=float)))


        fmt = '%.1f' # set the number of significant digits you want
        xyticks = mt.FormatStrFormatter(fmt)
        for ax in axs2:
            ax.yaxis.set_major_formatter(xyticks)
            ax.xaxis.set_major_formatter(xyticks)

        #axs[1].rcParams['legend.title_fontsize'] = 'xx-large'
        #axs[1].title('Rescaling factors for '+ str(grbtitle),fontsize=20)
        axs2[0].invert_yaxis()
        #axs[0].set_xlabel('Log time (s)',fontsize=22)
        axs2[0].set_ylabel('Magnitude', labelpad=15, fontsize=16, fontdict=dict(weight='bold'))
        axs2[0].tick_params(labelsize=16, direction='in', width=2)
        axs2[0].locator_params(axis='x', nbins=5)
        axs2[0].locator_params(axis='y', nbins=5)

        for tick in axs2[0].get_xticklabels():
            tick.set_fontweight('bold')
        for tick in axs2[0].get_yticklabels():
            tick.set_fontweight('bold')

        for axis in ['top', 'bottom', 'left', 'right']:

            axs2[0].spines[axis].set_linewidth(2.2)  # change width  

        
        #axs[1].title('Rescaling factors for '+ str(grbtitle),fontsize=20)
        axs2[1].set_xlabel(r"$\bf{log_{10} t\, (s)}$", fontsize=17, fontdict=dict(weight='bold'))
        axs2[1].set_ylabel('Resc. fact. (rf)', labelpad=15, fontsize=17, fontdict=dict(weight='bold')) #+filterforrescaling, labelpad=15, fontsize=15)
        axs2[1].tick_params(labelsize=16, direction='in', width=2)
        
        for tick in axs2[1].get_xticklabels():
            tick.set_fontweight('bold')
        for tick in axs2[1].get_yticklabels():
            tick.set_fontweight('bold')
        
        for axis in ['top', 'bottom', 'left', 'right']:

            axs2[1].spines[axis].set_linewidth(2.2)  # change width

        font = font_manager.FontProperties(weight='bold', style='normal', size=22) # 20

        fig2.legend(title="$\\bf{Band: rf=(a=0) *\\log_{10}(t) + (b \pm \sigma_{b})}$", bbox_to_anchor=(1, 1), loc='upper left', fontsize="18", title_fontsize="20", prop=font) # 21 21

        #fig.legend(title='Band: slope±err', bbox_to_anchor=(1, 0.946), loc='upper left', fontsize="21", title_fontsize="21")
        #plt.rcParams['legend.title_fontsize'] = 'x-large'
        plt.rcParams['figure.figsize'] = [16, 9] #15,8 16,9
        plt.tight_layout()
        
        #plt.savefig("new-plots/"+str(grbtitle)+"_lc_colorevol.pdf", bbox_inches='tight') #DECOMMENT TO EXPORT THE PLOTS
        
        if save_plot:
            #plt.savefig(str(self.name)+'_colorevol.pdf', dpi=300) # option to export the pdf plot of rescaling factors
            plt.savefig(os.path.join(save_in_folder+'/'+str(self.name.split("/")[-1])+'_colorevol_a0.pdf'), bbox_inches='tight', dpi=300) # option to export the pdf plot of rescaling factors        
        
        plt.show()

        plt.clf()

        # Here the code prints the dataframe of rescaling factors, that contains log10(time), slope, slope_err...
        rescale_df.drop(labels='plot_color', axis=1, inplace=True)     # before printing that dataframe, the code removes the columns of plot_color
        resc_slopes_df.drop(labels='plot_color', axis=1, inplace=True) # since this column was needed only for assigning the plot colors
                                                                       # these columns have no scientific meaning

        if reportfill:

            if len(nocolorevolutionlist)>len(colorevolutionlist):
                rescflag='yes'
            else:
                rescflag='no'

            reportfile = open('report_colorevolution.txt', 'a')
            reportfile.write(self.name.split("/")[-1]+" "+str(nocolorevolutionlist).replace(' ','')+" "+str(colorevolutionlist).replace(' ','')+" "+rescflag+" "+filterforrescaling+"\n")
            reportfile.close()

            #stringnew=self.name+" " # this is the general printing of all the slopes
            #for band in resc_slopes_df.index:

                #if (np.isnan(resc_slopes_df.loc[band, 'slope'])) or (np.isnan(resc_slopes_df.loc[band, 'slope_err'])):
                    #stringnew=stringnew
                #else:
                    #stringnew=stringnew+"$"+band+"$"+":"+str(round(resc_slopes_df.loc[band, 'slope'],3))+"+/-"+str(round(resc_slopes_df.loc[band, 'slope_err'],3))+"; "

            #slopesfile = open('report_slopes.txt', 'a')
            #slopesfile.write(stringnew+"\n")
            #slopesfile.close()

        if return_rescaledf: # variables returned in case the option return_rescaledf is enabled

            # Option that saves the dataframe that contains the rescaling factors
            rescale_df.to_csv(os.path.join(save_in_folder+'/'+str(self.name.split("/")[-1])+'_rescalingfactors_to_'+str(filterforrescaling)+'.txt'),sep=' ',index=False)
            
            resc_slopes_df.insert(0, 'GRB', str(self.name.split("/")[-1]))
            resc_slopes_df.insert(1, 'filter_chosen', str(filterforrescaling))
            
            resc_slopes_df.to_csv(os.path.join(save_in_folder+'/'+str(self.name.split("/")[-1])+'_fittingresults'+'.txt'),sep=' ',index=True)
            
            return fig, resc_slopes_df, nocolorevolutionlist, colorevolutionlist, filterforrescaling, light, rescale_df

        return fig, resc_slopes_df, nocolorevolutionlist, colorevolutionlist, filterforrescaling, light # the variables in the other case

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