# Standard libs
import os
import re
import math

# Third-party libs
import numpy as np
import pandas as pd
import lmfit as lf
from lmfit import Parameters, Model
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.ticker as mt
import matplotlib.font_manager as font_manager


def _colorevolGRB(
    grb: str, 
    df: pd.DataFrame, 
    chosenfilter: str ='mostnumerous', 
    print_status: bool =True, 
    save_in_folder: str =None, 
    debug: bool =False
):
    """
    Function to perform colour evolution analysis.

    """

    light = pd.DataFrame()  # Here the original light curve dataframe is defined
    light['time_sec'] = df.time_sec  # Time is linear
    light['mag'] = df.mag
    light['mag_err'] = df.mag_err
    light['band'] = df.band  # Here the band is the original one, not approximated
    light['band_appx'] = df.band_appx  # Here the band is the approximated one, e.g., u' -> u, Ks -> K
    light['band_appx_occurrences'] = ""
    light['system'] = df.system
    light['telescope'] = df.telescope
    light['extcorr'] = df.extcorr
    light['source'] = df.source
    light['flag'] = df.flag
    light['resc_fact'] = "-"
    light['resc_fact_err'] = "-"
    light['time_difference'] = "-"
    light['mag_overlap'] = "-"
    light['mag_chosenfilter'] = "-"  # Magnitude of the filter chosen for rescaling (either the most numerous or another one)
    light['mag_chosenfilter_err'] = "-"  # Error on the magnitude of the filter chosen for rescaling

    light = light[
        (light['mag_err'] != 0) &
        (light['time_sec'] > 0) &
        (light['flag'] != "yes")
    ]
    # Here the code requires only magnitudes and not limiting magnitudes,
    # there are some transients observed in the optical before the
    # satellite trigger, thus they have negative times since in our
    # analysis, we consider the trigger time as start time of the LC
    # We furthermore exclude the datapoints that are outliers,
    # namely, have the "yes" in the last column
    # We also exclude points with mag_err > 0.5

    assert len(light) != 0, "The magnitude file has only limiting magnitudes."
    # Assert is a command that verifies the condition written, if the condition
    # doesn't hold then the error on the right is printed
    assert len(light) > 1, "The magnitude file has only one data point."
    # Here we highlight if the dataframe has only limiting magnitudes
    # or if it has only one data point


    occur = light['band_appx'].value_counts()
    # This command returns a dataframe that contains in one column the
    # label of the filter and in another column the occurrences
    # Example: filter occurrences
    # R 18
    # I 6
    # B 5
    # F606W 4

    # In this loop, the column of band_appx_occurrences is filled with the countings of each filter
    # E.g. if the filter R is present 50 times in the magnitudes this will append 50 to the row where this filter is present

    for row in light.index:
        for ff in occur.index:
            if light.loc[row, "band_appx"] == ff:
                light.loc[row, "band_appx_occurrences"] = occur[ff]

    # Identifying the most numerous filter in the GRB

    assert chosenfilter == 'mostnumerous' or chosenfilter in df.band, (
        "Rescaling band provided as <<chosenfilter>> is not present in data. Check the filters present in the magnitudes file."
    )

    # chosenfilter is an input of the function colorevolGRB(...)
    # This assert condition is needed to verify that the filter is present in the LC; for example, if "g" is selected but it's
    # not present then the string "Rescaling..." is printed

    if chosenfilter == 'mostnumerous':
        # Here I select by default the filterforrescaling as the most numerous inside the LC
        # If chosenfilter input is 'mostnumerous', then I automatically take the most numerous
        filterforrescaling = occur.index[0]  # Namely, the first element in the occur frame (with index 0, since in Python the counting starts from zero)
        filteroccurrences = occur[0]  # This is the number of occurrences of the filterforrescaling
    else:
        for ii in occur.index:
            if ii == chosenfilter:
                filterforrescaling = ii
                filteroccurrences = occur[ii]

    if print_status:
        # The print_status option is set to true by default, and it prints
        print(grb)  # the GRB name
        print('-------')  # and the details of the filter chosen for rescaling, name + occurrences
        print(occur)
        print(
            '\n The filter chosen in this GRB: ', filterforrescaling, ', with', filteroccurrences, 'occurrences.\n' +
            'This filter will be considered for rescaling'
        )

    # In the following rows the code extracts only the datapoints with the filter chosen for rescaling (usually, the most numerous)

    mostnumerouslight = light.loc[(light['band_appx'] == filterforrescaling)]  # mostnumerouslight dataframe is the one constituted of the chosen filter for rescaling,
    mostnumerousx = mostnumerouslight['time_sec'].values  # for simplicity it is called mostnumerouslight
    mostnumerousy = mostnumerouslight['mag'].values  # time_sec is linear
    mostnumerousyerr = mostnumerouslight['mag_err'].values

    # The following loop fills the columns of the general dataframe with the rescaling factor, its error, the time difference
    # between the filter in the dataframe and the filter chosen for rescaling, and the magnitude overlap (if overlap is zero,
    # then the magnitude values do not overlap)

    for row in light.index:  # Running on all the magnitudes
        if light.loc[row, "band_appx"] == filterforrescaling:
            # When the filter in the dataframe is the filter chosen for rescaling,
            light.loc[row, "resc_fact"] = "-"  # The rescaling factor is obviously not existing and the columns are filled with "-"
            light.loc[row, "resc_fact_err"] = "-"
            light.loc[row, "time_difference_percentage"] = "-"
            light.loc[row, "mag_chosenfilter"] = "-"
            light.loc[row, "mag_chosenfilter_err"] = "-"
        else:
            compatiblerescalingfactors = []  # The compatiblerescalingfactors is a list that contains all the possible rescaling factors for a magnitude, since more of them can fall in the 2.5 percent criteria
            for pp in range(len(mostnumerouslight)):  # Running on the magnitudes of the filter chosen for rescaling
                if np.abs(mostnumerousx[pp] - light.loc[row, "time_sec"]) <= (0.025 * mostnumerousx[pp]):
                    # If the filter chosen for rescaling is in the 2.5 percent time condition with the magnitude of the loop
                    rescfact = mostnumerousy[pp] - light.loc[row, "mag"]  # This is the rescaling factor, mag_filterforrescaling - mag_filter
                    rescfacterr = np.sqrt(mostnumerousyerr[pp]**2 + light.loc[row, "mag_err"]**2)  # Rescaling factor error, propagation of uncertainties on both the magnitudes
                    timediff = np.abs(mostnumerousx[pp] - light.loc[row, "time_sec"]) / mostnumerousx[pp]
                    # Linear time difference between the time of filter chosen for rescaling and the filter to be rescaled, divided by the chosen filter time
                    magchosenf = mostnumerousy[pp]
                    magchosenferr = mostnumerousyerr[pp]
                    compatiblerescalingfactors.append([rescfact, rescfacterr, timediff, magchosenf, magchosenferr])
                    # All these values are appended in the compatiblerescalingfactors list, since in principle there may be more than one for a single datapoint

            if len(compatiblerescalingfactors) == 0:
                # If there are no rescaling factors that respect the 2.5 percent for the given filter, then once again the columns are filled with "-"
                light.loc[row, "resc_fact"] = "-"
                light.loc[row, "resc_fact_err"] = "-"
                light.loc[row, "time_difference_percentage"] = "-"
                light.loc[row, "mag_chosenfilter"] = "-"
                light.loc[row, "mag_chosenfilter_err"] = "-"
                continue
            else:
                # In the other cases
                minimumtimediff = min([k[2] for k in compatiblerescalingfactors])
                # The rescaling factor with the minimum time difference between the filter chosen for rescaling and the filter to be rescaled is taken
                acceptedrescalingfactor = list(filter(lambda x: x[2] == minimumtimediff, compatiblerescalingfactors))
                # This line locates the rescaling factor with minimum time difference
                light.loc[row, "resc_fact"] = acceptedrescalingfactor[0][0]
                light.loc[row, "resc_fact_err"] = acceptedrescalingfactor[0][1]
                light.loc[row, "time_difference_percentage"] = acceptedrescalingfactor[0][2]
                light.loc[row, "mag_chosenfilter"] = acceptedrescalingfactor[0][3]
                light.loc[row, "mag_chosenfilter_err"] = acceptedrescalingfactor[0][4]

    lightonlyrescalable = light[light["resc_fact"] != '-']  # The lightonlyrescalable selects only the datapoints with rescaling factors

    filt = []  # These are the empty lists to be filled with filter
    filtoccur = []  # Occurrences of the filter
    resclogtime = []  # log10(time) of the rescaling factor for the filter
    rescfact = []  # Rescaling factor of the filter
    rescfacterr = []  # Rescaling factor error of the filter
    rescfactweights = []  # Weights of the rescaling factor
    for jj in lightonlyrescalable.index:
        filt.append(lightonlyrescalable.loc[jj, "band_appx"])  # Here we have the filters that are rescaled to the selected filter for rescaling
        filtoccur.append(lightonlyrescalable.loc[jj, "band_appx_occurrences"])  # Here we have the occurrences of the filters
        resclogtime.append(np.log10(lightonlyrescalable.loc[jj, "time_sec"]))  # WATCH OUT! For the plot and fitting, we take the log10(time) of rescaling factor
        rescfact.append(light.loc[jj, "resc_fact"])  # The rescaling factor value
        rescfacterr.append(light.loc[jj, "resc_fact_err"])  # The rescaling factor error
        rescfactweights.append((1 / light.loc[jj, "resc_fact_err"]))  # The weights on the rescaling factor

    # The weights are considered as 1/yerr given that the lmfit package will be used: https://lmfit.github.io/lmfit-py/model.html#the-model-class

    # The following command defines the dataframe of rescaling factors

    rescale_df = pd.DataFrame(
        list(
            zip(
                filt, filtoccur, resclogtime, rescfact,
                rescfacterr, rescfactweights
            )
        ),
        columns=[
            'band', 'Occur_band', 'Log10(t)', 'Resc_fact', 'Resc_fact_err', 'Resc_fact_weights'
        ]
    )

    x_all = rescale_df['Log10(t)']  # List of log10 times for the rescaling factors
    y_all = rescale_df['Resc_fact']  # List of the rescaling factors
    yerr_all = rescale_df['Resc_fact_err']  # List of the rescaling factors errors
    filters = [*set(rescale_df['band'].values)]  # List of filters in the rescaling factors sample
    rescale_df['plot_color'] = ""  # Empty list that will be filled with the color map condition

    # Set the color map to match the number of filter
    cmap = plt.get_cmap('gist_ncar')  # Import the color map
    cNorm = colors.Normalize(vmin=0, vmax=len(filters))  # Linear map of the colors in the colormap from data values vmin to vmax
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)  # The ScalarMappable applies data normalization before returning RGBA colors from the given colormap

    # Plot each filter
    fig_avar, axs = plt.subplots(2, 1, sharex=True)

    for i, band in enumerate(filters):  # Loop on the given filter
        colour = scalarMap.to_rgba(i)  # Mapping the colour into the RGBA
        index = rescale_df['band'] == band  # Selects the magnitudes that have the filter equal to the band on which the loop iterates
        axs[1].scatter(
            x_all[index], y_all[index],  # Options for the plot of the central values
            s=15,
            color=colour  # Color-coding of the plot
        )
        axs[1].errorbar(
            x_all[index], y_all[index], yerr_all[index],  # Options for the plot of the error bars, these must be added in this command
            fmt='o',  # This is the data marker, a circle
            barsabove=True,  # Bars plotted above the data marker
            ls='',  # Line style = None, so no lines are drawn between the points (later the fit will be done and plotted)
            color=colour  # Color-coding
        )
        for j in rescale_df[index].index:
            rescale_df.at[j, "plot_color"] = colour  # This loop assigns each filter to a color in the plot

    axs[0].text(
        0.1, 0.1, "GRB " + (grb.split("/")[-1]).split("_")[0], fontsize=22, fontweight='bold', horizontalalignment='left', verticalalignment='bottom', transform=axs[0].transAxes
    )

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

            resc_slopes_df.loc[band, 'intercept_a0'] = np.around(slopezero_fit.params['c'].value, decimals=3) 
            # intercept with a=0
            resc_slopes_df.loc[band, 'inter_a0_err'] = np.around(slopezero_fit.params['c'].stderr, decimals=3) 
            # intercept error with a=0
            resc_slopes_df.loc[band, 'intercept_a0_err/inter_a0'] = np.around(np.abs(
                                            np.around(slopezero_fit.params['c'].stderr, decimals=3) /np.around(slopezero_fit.params['c'].value,
                                            decimals=3)), decimals=3) 
            # intercept_err/intercept = |intercept_err|/|intercept| when slope (a) =0
            resc_slopes_df.loc[band, 'red_chi2_a0'] = np.around(slopezero_fit.redchi, decimals=3) 
            # reduced chi^2 for a=0

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

            if np.abs(resc_slopes_df.loc[band, 'slope_lin']) >= 0.001: 
                # in the case of non-zero slope
                
                if np.abs(resc_slopes_df.loc[band, 'slope_lin_err/slope_lin']) <= 1000: 
                    # this boundary of slope_err/slope is put ad-hoc to show all the plots
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
        sublight=light.loc[(light['band_appx'] == band)]
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

        rescale_df_print = rescale_df.drop('plot_color', axis=1)
        resc_slopes_df_print = resc_slopes_df.drop('plot_color', axis=1)
        print("Individual point rescaling:")
        print(rescale_df_print) # the dataframe of rescaling factors

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

    if print_status:    
        print(string)

    fmt = '%.1f' # set the number of significant digits you want
    xyticks = mt.FormatStrFormatter(fmt)
    for ax in axs:
        ax.yaxis.set_major_formatter(xyticks)
        ax.xaxis.set_major_formatter(xyticks)

    axs[0].invert_yaxis()
    axs[0].set_ylabel('Magnitude', labelpad=15, fontsize=16, fontdict=dict(weight='bold'))
    axs[0].tick_params(labelsize=16, direction='in', width=2)
    axs[0].locator_params(axis='x', nbins=5)
    axs[0].locator_params(axis='y', nbins=5)

    for tick in axs[0].get_xticklabels():
        tick.set_fontweight('bold')
    for tick in axs[0].get_yticklabels():
        tick.set_fontweight('bold')

    for axis in ['top', 'bottom', 'left', 'right']:
        axs[0].spines[axis].set_linewidth(2.2)

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

    fig_avar.legend(
        title="$\\bf{Band: rf=(a \pm \sigma_{a}) *\\log_{10}(t) + (b \pm \sigma_{b})}$", 
        bbox_to_anchor=(1, 1), 
        loc='upper left', 
        fontsize="20", 
        title_fontsize="20", 
        prop=font
        )

    plt.rcParams['figure.figsize'] = [16, 9] 
    plt.tight_layout()
    
    if save_in_folder:
        plt.savefig(os.path.join(save_in_folder+'/'+str(grb.split("/")[-1])+'_colorevol.pdf'), bbox_inches='tight', dpi=300) # option to export the pdf plot of rescaling factors        
    
    ############################################## Plotting the case where a=0 ############################################

    fig_a0, axs2 = plt.subplots(2, 1, sharex=True)

    for i, band in enumerate(filters): # loop on the given filter
        colour = scalarMap.to_rgba(i) # mapping the colour into the RGBA
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

    axs2[0].text(0.1, 0.1, "GRB "+(grb.split("/")[-1]).split("_")[0], fontsize=22, fontweight='bold', horizontalalignment='left', verticalalignment='bottom', transform=axs2[0].transAxes)


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
        sublight=light.loc[(light['band_appx'] == band)]
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

    fig_a0.legend(title="$\\bf{Band: rf=(a=0) *\\log_{10}(t) + (b \pm \sigma_{b})}$", bbox_to_anchor=(1, 1), loc='upper left', fontsize="18", title_fontsize="20", prop=font) # 21 21
    
    plt.rcParams['figure.figsize'] = [16, 9] #15,8 16,9
    plt.tight_layout()
    
    if save_in_folder:
        plt.savefig(os.path.join(save_in_folder+'/'+str(grb.split("/")[-1])+'_colorevol_a0.pdf'), bbox_inches='tight', dpi=300) 
        # option to export the pdf plot of rescaling factors        

    nocolorevolutionlista0=[] # this is the list of filters with slopes that are compatible with zero in 3 sigma
    colorevolutionlista0=[] # this is the list of filters with slopes that ARE NOT compatible with zero in 3 sigma        
    for band in resc_slopes_df.index:

        try:
            float(resc_slopes_df.loc[band, 'prob_a0'])
        except:
            continue
        
        if float(resc_slopes_df.loc[band, 'prob_a0']) >= 0.05:
            nocolorevolutionlista0.append(band)
        elif float(resc_slopes_df.loc[band, 'prob_a0']) < 0.05:
            colorevolutionlista0.append(band)
        else:
            continue

    if print_status:
        if len(nocolorevolutionlista0)==0: # if there are no filters compatible with zero in 3 sigma
            print('No filters that have no color evolution according to a=0')

        else:
            print('Filters with no color evolution according to a=0 : ',*nocolorevolutionlista0) # if there are filters without color evolution, namely, compatible with zero in 3 sigma

        if len(colorevolutionlista0)==0: # if there are not filters with color evolution, namely, that are not compatible with zero in 3 sigma
            print('No filters that have color evolution according to a=0')

        else: # otherwise
            print('Filters with color evolution according to a=0 : ',*colorevolutionlista0)
        print('\n')
        print('No color evolution: ',*nocolorevolutionlista0,' ; Color evolution: ',*colorevolutionlista0) # print of the two lists



    # Here the code prints the dataframe of rescaling factors, that contains log10(time), slope, slope_err...
    rescale_df.drop(labels='plot_color', axis=1, inplace=True)     # before printing that dataframe, the code removes the columns of plot_color
    resc_slopes_df.drop(labels='plot_color', axis=1, inplace=True) # since this column was needed only for assigning the plot colors
                                                                    # these columns have no scientific meaning
    resc_slopes_df.insert(0, 'GRB', str(grb.split("/")[-1]))
    resc_slopes_df.insert(1, 'filter_chosen', str(filterforrescaling))
    
    # Option that saves the dataframe that contains the rescaling factors
    if save_in_folder:
        rescale_df.to_csv(os.path.join(save_in_folder+'/'+str(grb.split("/")[-1])+'_rescalingfactors_to_'+str(filterforrescaling)+'.txt'),sep='\t',index=False)
        resc_slopes_df.to_csv(os.path.join(save_in_folder+'/'+str(grb.split("/")[-1])+'_fittingresults'+'.txt'), sep='\t',index=True)
        
    if debug:

        if len(nocolorevolutionlista0)>len(colorevolutionlista0):
            rescflag='yes'
        else:
            rescflag='no'

        reportfile = open('report_colorevolution.txt', 'a')
        reportfile.write(grb.split("/")[-1]+" "+str(nocolorevolutionlista0).replace(' ','')+" "+str(colorevolutionlista0).replace(' ','')+" "+rescflag+" "+filterforrescaling+"\n")
        reportfile.close()

    return fig_avar, fig_a0, filterforrescaling, nocolorevolutionlist, colorevolutionlist, nocolorevolutionlista0, colorevolutionlista0, light, resc_slopes_df, rescale_df