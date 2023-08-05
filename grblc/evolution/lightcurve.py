# Standard libs
import os

# Third party libs
import numpy as np
import pandas as pd
import lmfit as lf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import plotly.express as px

# Custom modules
from . import io


class Lightcurve:
    _name_placeholder = "unknown grb"
    _flux_fixed_inplace = False



    def __init__(
        self,
        path: str = None,
        data_space: str = 'lin',
        name: str = None,
    ):
        """The main module for fitting lightcurves.

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
        appx_bands: str, optional
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
            self.df = io.read_data(path, data_space=data_space)



    def displayGRB(
            self, 
            save_static = False, 
            save_static_type = '.png', 
            save_interactive = False, 
            save_in_folder = 'plots/'
    ):

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

        tailpoint_list = []
        for t,m,e in zip(self.xdata,self.ydata,self.yerr):
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

        if save_static:
            fig.write_image(save_in_folder+self.name+save_static_type)

        if save_interactive:
            fig.write_html(save_in_folder+self.name+'.html')

        return fig



    # The function that calls io.py to read the data and performs band approximations.
    # It is not callable outside the class.
    def set_data(
        self, 
        appx_bands = False
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
        appx_bands: str, optional
            Whether to approximate bands for color evolution analysis. The following approximations are performed:
            - primed SDSS ugriz bands are treated same as unprimed
            - Ks = K' = K
            - Js = J
            - Mould photometric system is treated same as Johnson - Cousins
        """

        # Aprroximations related to the analysis are performed within the class 
        # as they are checked for validity only in the context of color evolution

        # Excluding limiting magnitudes
        assert len(self.df[self.df['mag_err'] != 0]) != 0, "Only limiting magnitudes present."
        assert len(self.df[self.df['mag_err'] != 0]) > 1, "Has only one data point."

        self.df = self.df[(self.df['mag_err'] != 0) & (self.df['time_sec']>0)]
        
        # Band approximations
        def _convert_bands(data):

            data = list(data)

            for i, band in enumerate(data):
                if band.lower() in ['clear', 'unfiltered', 'lum']:
                    band == band.lower()

            if appx_bands:
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
        
        self.xdata = self.df["time_sec"].to_numpy()
        self.ydata = self.df["mag"].to_numpy()
        self.yerr = self.df["mag_err"].to_numpy()
        self.band_og = self.df["band"].to_list()
        self.band = self.df["band"] = _convert_bands(self.df["band"])
        self.system = self.df["system"].to_list()
        self.telescope = self.df["telescope"].to_list()
        self.extcorr = self.df["extcorr"].to_list()
        self.source = self.df["source"].to_list()
    

   
    def colorevolGRB( 
            self, 
            chosenfilter = 'numerous', 
            print_status = True, 
            save_plot = False, 
            save_in_folder = 'rescale/'
            #return_rescaledf=False
    ):
        """
        This monstrosity performs the color evolution analysis.
        """
            

        '''filters = pd.DataFrame(self.band.value_counts())
        filters.rename(columns={'band':'occur'}, inplace=True)

        assert ref_band == 'numerous' or ref_band in self.band, "Rescaling band provided is not present in data!"

        # Identifying the most numerous filter in the GRB 
        if ref_band == 'numerous':
            ref_band = filters.index[0]

        if print_status:
            print(self.name)
            print('-------')
            print(filters, '\nThe reference filter for rescaling of this GRB: ', ref_band, 
                ', with', filters.loc[ref_band, 'occur'], 'occurrences.\n')
            

        # Set the color map to match the number of filter
        cmap = plt.get_cmap('gist_ncar')
        cNorm  = colors.Normalize(vmin=0, vmax=len(filters))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        filters['plot_color'] = ""
        for i, band in enumerate(filters.index):
            colour = scalarMap.to_rgba(i)
            filters.at[band, 'plot_color'] = colour'''
        
        global nocolorevolutionlist, colorevolutionlist, light, lightonlyrescalable #, overlap
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
        light['resc_fact'] = ""
        light['resc_fact_err'] = ""
        light['time_difference'] = ""
        light['mag_overlap'] = ""
        light['mag_chosenfilter'] = "" # magnitude of the filter chosen for rescaling (either the most numerous or another one)
        light['mag_chosenfilter_err'] = "" # error on the magnitude of the filter chosen for rescaling


        light = light[(light['mag_err']!=0) & (light['time_sec']>0)] # here the code requires only magnitudes and not limiting magnitudes,
                                                                     # there are some transients observed in the optical before the
                                                                     # satellite trigger, thus they have negative times since in our
                                                                     # analysis, we consider the trigger time as start time of the LC

        assert len(light)!=0, "Has only limiting magnitudes." # assert is a command that verifies the condition written, if the condition
                                                              # doesn't hold then the error on the right is printed
        assert len(light)>1, "Has only one data point."       # here we highlight if the dataframe has only limiting magnitudes
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

        assert chosenfilter == 'mostnumerous' or chosenfilter in self.band, "Rescaling band provided is not present in data!"

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

        mostnumerouslight=light.loc[(light['band'] == filterforrescaling)] # mostnumerouslight dataframe is the one constituted of the chosen filter for rescaling,
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
        fig = plt.figure()

        for i, band in enumerate(filters): # loop on the given filter
            colour = scalarMap.to_rgba(i) # mapping the colour into the RGBA
            index = rescale_df['band'] == band # selects the magnitudes that have the filter equal to the band on which the loop iterates
            plt.scatter(x_all[index], y_all[index], # options for the plot of the central values
                        s=15,
                        color=colour) # color-coding of the plot
            plt.errorbar(x_all[index], y_all[index], yerr_all[index], #options for the plot of the error bars, these must be added in this command
                        fmt='o', # this is the data marker, a circle
                        barsabove=True, # bars plotted above the data marker
                        ls='', # line style = None, so no lines are drawn between the points (later the fit will be done and plotted)
                        color=colour # color-coding
                        )
            for j in rescale_df[index].index:
                rescale_df.at[j,"plot_color"] = colour # this loop assigns each filter to a color in the plot

        resc_slopes_df = pd.DataFrame() # initialize of the rescaling factors fitting dataframe
        resc_slopes_df.index = filters # the filters are taken as index
        resc_slopes_df['slope'] = "" # placeholder, default set to empty, then it will change - slope of the linear fit
        resc_slopes_df['slope_err'] = "" # placeholder, default set to empty, then it will change - error on slope
        resc_slopes_df['intercept'] = "" # placeholder, default set to empty, then it will change - intercept of linear fit
        resc_slopes_df['inter_err'] = "" # placeholder, default set to empty, then it will change - error on intercept
        resc_slopes_df['slope_err/slope'] = "" # placeholder, default set to empty, then it will change - slope_err/slope = |slope_err|/|slope|
        resc_slopes_df['red_chi2'] = "" # placeholder, default set to empty, then it will change - reduced chi^2
        resc_slopes_df['comment'] = "" # placeholder, default set to empty, then it will change - the comment that will say "no color evolution","color evolution"
        resc_slopes_df['plot_color'] = "" # placeholder, default set to empty, then it will change - color-coding for the fitting lines

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
                resc_slopes_df.loc[band, 'slope'] = np.around(linear_fit.params['line_slope'].value, decimals=3) # slope of the fitting
                resc_slopes_df.loc[band, 'slope_err'] = np.around(linear_fit.params['line_slope'].stderr, decimals=3) # slope error
                resc_slopes_df.loc[band, 'intercept'] = np.around(linear_fit.params['line_intercept'].value, decimals=3) # intercept
                resc_slopes_df.loc[band, 'inter_err'] = np.around(linear_fit.params['line_intercept'].stderr, decimals=3) # intercept error
                resc_slopes_df.loc[band, 'slope_err/slope'] = np.around(np.abs(np.around(linear_fit.params['line_slope'].stderr, decimals=3) /np.around(linear_fit.params['line_slope'].value, decimals=3)), decimals=3) # slope_err/slope = |slope_err|/|slope|
                resc_slopes_df.loc[band, 'red_chi2'] = np.around(linear_fit.redchi, decimals=3) # reduced chi^2

            else: # not enough data points, less than 3 rescaling factors for the filter
                resc_slopes_df.loc[band, 'slope'] = np.nan
                resc_slopes_df.loc[band, 'slope_err'] = np.nan
                resc_slopes_df.loc[band, 'intercept'] = np.nan
                resc_slopes_df.loc[band, 'inter_err'] = np.nan
                resc_slopes_df.loc[band, 'slope_err/slope'] = np.nan
                resc_slopes_df.loc[band, 'comment'] = "insufficient data"
                resc_slopes_df.loc[band, 'red_chi2'] = 'insufficient data'

            if resc_slopes_df.loc[band, 'slope'] < 0.001: # in case of zero-slope, since the fittings have 3 digits precision we consider the precision 0.001 as the "smallest value different from zero"
                    if resc_slopes_df.loc[band, 'slope_err/slope'] <= 10:
                        y_fit = resc_slopes_df.loc[band, 'slope'] * x + resc_slopes_df.loc[band, 'intercept']
                        plt.plot(x, y_fit, color=tuple(np.array(re.split('[(),]', resc_slopes_df.loc[band, "plot_color"])[1:-1], dtype=float))) # plot of the fitting line between log10(t) and resc_fact

                        resc_slopes_df.loc[band, 'comment'] = "no color evolution"


            if resc_slopes_df.loc[band, 'slope'] >= 0.001: # in the case of non-zero slope
                if resc_slopes_df.loc[band, 'slope_err/slope'] <= 10: # this boundary of slope_err/slope is put ad-hoc to show all the plots
                                                                   # it's a large number that can be modified
                    y_fit = resc_slopes_df.loc[band, 'slope'] * x + resc_slopes_df.loc[band, 'intercept'] # fitted y-value according to linear model
                    plt.plot(x, y_fit, color=tuple(np.array(re.split('[(),]', resc_slopes_df.loc[band, "plot_color"])[1:-1], dtype=float))) # plot of the fitting line between log10(t) and resc_fact

                    if resc_slopes_df.loc[band, 'slope']-(3*resc_slopes_df.loc[band, 'slope_err'])<=0<=resc_slopes_df.loc[band, 'slope']+(3*resc_slopes_df.loc[band, 'slope_err']):
                        resc_slopes_df.loc[band, 'comment'] = "no color evolution" # in case it is comp. with zero in 3 sigma, there is no color evolution
                    else:
                        resc_slopes_df.loc[band, 'comment'] = "color evolution" # in case the slope is not compatible with zero in 3 sigma

                else:
                    resc_slopes_df.loc[band, 'comment'] = "slope_err/slope>10sigma"  # when the slope_err/slope is very high

        for band in resc_slopes_df.index: # this loop defines the labels to be put in the rescaling factor plot legend

            if np.isnan(resc_slopes_df.loc[band, "slope"])==True: # in case the fitting is not done, the label will be "filter: no fitting"
                label=band+": failed fitting"
            else:
                label=band+": "+ str(resc_slopes_df.loc[band, "slope"]) + r'$\pm$' + str(resc_slopes_df.loc[band, "slope_err"])
                # when the slopes are estimated, the label is "filter: slope +/- slope_err"

            ind = rescale_df.index[rescale_df['band'] == band][0] # initializing the variables to be plotted for each filter
            color = rescale_df.loc[ind, "plot_color"]
            plt.scatter(x=[], y=[],
                        color=color,
                        label=label # here the labels for each filter are inserted
                        )

        plt.rcParams['legend.title_fontsize'] = 'xx-large' #options for the plot of rescaling factors
        plt.xlabel('Log time (s)',fontsize=22)
        plt.ylabel('Rescaling factor to '+filterforrescaling+' (mag)',fontsize=22)
        plt.rcParams['figure.figsize'] = [15, 10]
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.title("GRB "+self.name.split("/")[-1], fontsize=22)
        plt.legend(title='Band & slope', bbox_to_anchor=(1.015, 1.015), loc='upper left', fontsize='xx-large') # legend, it uses the colors and labels

        if save_plot:
            plt.savefig(os.path.join(save_in_folder+'/'+str(self.name)+'_colorevol.pdf'), dpi=300) # option to export the pdf plot of rescaling factors

        plt.show()

        # Here the code prints the dataframe of rescaling factors, that contains log10(time), slope, slope_err...
        rescale_df.drop(labels='plot_color', axis=1, inplace=True)     # before printing that dataframe, the code removes the columns of plot_color
        resc_slopes_df.drop(labels='plot_color', axis=1, inplace=True) # since this column was needed only for assigning the plot colors
                                                                       # these columns have no scientific meaning

        if print_status: # when this option is selected in the function it prints the following

            print("Individual point rescaling:")
            print(rescale_df) # the dataframe of rescaling factors

            print("\nSlopes of rescale factors for each filter:")
            print(resc_slopes_df) # the dataframe that contains the fitting parameters of rescaling factors

        compatibilitylist=[] # here we initialize the list that contains the ranges of (slope-3sigma,slope+3sigma) for each filter

        for band in resc_slopes_df.index: # this code appends the values of (slope-3sigma,slope+3sigma) in case the slope is not a "nan"
                                          # and in case both the slope and slope_err are different from zero
            if resc_slopes_df.loc[band, 'slope']!=0 and resc_slopes_df.loc[band, 'slope_err']!=0 and np.isnan(resc_slopes_df.loc[band, 'slope'])==False and np.isnan(resc_slopes_df.loc[band, 'slope_err'])==False:
                compatibilitylist.append([band,[resc_slopes_df.loc[band, 'slope']-(3*resc_slopes_df.loc[band, 'slope_err']),
                                        resc_slopes_df.loc[band, 'slope']+(3*resc_slopes_df.loc[band, 'slope_err'])]])

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
                print('Filters compatible with zero in 3sigma: ',*nocolorevolutionlist) # if there are filters without color evolution, namely, compatible with zero in 3 sigma

            if len(colorevolutionlist)==0: # if there are not filters with color evolution, namely, that are not compatible with zero in 3 sigma
                print('No filters compatible with zero in >3sigma')

            else: # otherwise
                print('Filters not compatible with zero in 3sigma: ',*colorevolutionlist)
            print('\n')
            print('No color evolution: ',*nocolorevolutionlist,' ; Color evolution: ',*colorevolutionlist) # print of the two lists


        string="" # this is the general printing of all the slopes
        for band in resc_slopes_df.index:
            string=string+band+":"+str(round(resc_slopes_df.loc[band, 'slope'],3))+"+/-"+str(round(resc_slopes_df.loc[band, 'slope_err'],3))+"; "

        print(string)

        if return_rescaledf: # variables returned in case the option return_rescaledf is enabled

            # Option that saves the dataframe that contains the rescaling factors
            rescale_df.to_csv(os.path.join(save_rescaled_in+'/'+str(self.name)+'_rescalingfactors_to_'+str(filterforrescaling)+'.txt'),sep=' ',index=False)

            return fig, rescale_df, resc_slopes_df, nocolorevolutionlist, colorevolutionlist, filterforrescaling, light

        return fig, resc_slopes_df, nocolorevolutionlist, colorevolutionlist, filterforrescaling, light # the variables in the other case


    def rescaleGRB(self, output_colorevolGRB, chosenfilter='mostnumerous', save_rescaled_in=''): # this function makes the rescaling of the GRB

        # the global option is needed when these variables inputed in the current function are output of another function recalled, namely, colorevolGRB
        global filterforrescaling, light, overlap #, nocolorevolutionlist

        def overlap(mag1lower,mag1upper,mag2lower,mag2upper): # this is the condition to state if two magnitude ranges overlap
            if mag1upper <mag2lower or mag1lower > mag2upper:
                return 0 # in the case of no overlap, zero is returned
            else:
                return max(mag1upper, mag2upper) # in the case of overlap, a value>0 is returned

        # here the code uses the colorevolGRB function defined above; the outputs of the function colorevolGRB will be used as input in the current function

        #output_colorevolGRB = self.colorevolGRB(print_status=False, return_rescaledf=False, save_plot=False, chosenfilter=chosenfilter, save_in_folder=save_rescaled_in)
        input = output_colorevolGRB
        nocolorevolutionlist = input[2] # 3rd output of colorevolGRB function, this is the list of filters whose resc.fact. slopes are compatible with zero in 3sigma or are < 0.10
        colorevolutionlist =input[3] # 4th output of colorevolGRB function, this is the list of filters whose resc.fact. slopes are incompatible with zero in 3sigma and are>0.10
        filterforrescaling = input[4] # 5th output of colorevolGRB function, this is the filter chosen for rescaling
        light = input[5] # 6th output of colorevolGRB function, is the original dataframe (since the filter chosen for rescaling is present here and it is needed)

        # Before rescaling the magnitudes, the following instructions plot the magnitudes in the unrescaled case
        figunresc = px.scatter(
                x=np.log10(light['time_sec'].values), # the time is set to log10(time) only in the plot frame
                y=light['mag'].values,
                error_y=light['mag_err'].values,
                color=light['band'].values,
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


        # The plot of the rescaled dataframe
        figresc = px.scatter(
                x=np.log10(light["time_sec"].values), # the time is set to log10(time) only in the plot frame
                y=light["mag_rescaled_to_"+filterforrescaling].values,
                error_y=light["mag_rescaled_err"].values,
                color=light["band_approx"],
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

        # The option for exporting the rescaled magnitudes as a dataframe
        if not os.path.exists(save_rescaled_in):
            os.makedirs(save_rescaled_in)
        rescmagdataframe.to_csv(os.path.join(save_rescaled_in+'/' + str(self.name).split("/")[-1]+  '_rescaled_to_'+str(filterforrescaling)+'.txt'),sep=' ',index=False)

        return rescmagdataframe
