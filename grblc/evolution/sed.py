import os
import math
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
import warnings
warnings.filterwarnings("ignore")

from ..photometry.extinction import pei_av
from ..photometry.match import calibration
from ..io import read_data


def model_MW(
    beta,
    intercept,
    AV,
    x,
    z
):
    lam = 10**x
    sed = intercept - 2.5*beta*x - 2.5*(pei_av(lam,A_V=AV,gal=1,R_V=3.08) - pei_av(lam/(1+z),A_V=AV,gal=1,R_V=3.08))
    return sed

def model_LMC(
    beta,
    intercept,
    AV,
    x,
    z
):
    lam = 10**x
    sed = intercept - 2.5*beta*x - 2.5*(pei_av(lam,A_V=AV,gal=2,R_V=3.16) - pei_av(lam/(1+z),A_V=AV,gal=2,R_V=3.16))
    return sed

def model_SMC(
    beta,
    intercept,
    AV,
    x,
    z
):
    lam = 10**x
    sed = intercept - 2.5*beta*x - 2.5*(pei_av(lam,A_V=AV,gal=3,R_V=2.93) - pei_av(lam/(1+z),A_V=AV,gal=3,R_V=2.93))
    return sed

def maketable(
    path
):
    dtype = {
        "time_sec": np.float64,
        "mag": np.float64,
        "mag_err": np.float64,
        "band": str,
        "system": str,
        "telescope": str,
        "extcorr": str,
        "source": str
    }
    names = list(dtype.keys())
    #import raw data file
    mag_table = read_data(path=path)

    df = {k: [] for k in ("time_sec", "mag_corr", "mag_err", "lambda", "band", "source", "telescope", "flag")}
    for __, row in mag_table.iterrows():
        time_sec = row["time_sec"]
        mag = row["mag"]
        mag_err = row["mag_err"]
        band = row["band"]
        source = row["source"]
        telescope = row["telescope"]
        flag = row["flag"]
        
        #applying calibration without coefficient for galactic extinction correction
        lambda_x, *__ = calibration(band, telescope)
        A_x = 0 # again assuming correction has ALREADY BEEN APPLIED
        mag_corr = mag - A_x

        if mag_err/mag_corr < 0:
            continue
        else:
            df["time_sec"].append(np.log10(time_sec))
            df["mag_corr"].append(mag_corr)
            df["mag_err"].append(mag_err)
            df["lambda"].append(lambda_x)
            df["band"].append(band)
            df["source"].append(source)
            df["telescope"].append(telescope)
            df["flag"].append(flag)

    df = pd.DataFrame(df)
    df = df.sort_values(by=['time_sec'], ascending=True)
    
    df = df[df["flag"]=="no"]
    df = df[df["mag_err"]<=0.5] # after the 01 November 2023 NAOJ colloquium, we put a cut on the tail of the magerr distribution
    
    spectral = pd.DataFrame()
    skips=[]
    for i,t in enumerate(df['time_sec']):
      flg=0 # flg is used for removing exactly the same datapoints
      df2 = df[np.abs(10**df['time_sec']-10**t) <= (10**t)*0.025]  # 2.5% method
      if len(set(df2["band"].values))>= 3: # use only if there are more than 2 filters
        for j in range(len(skips)):
          if len(df2.values)==len(skips[j]):
              if np.all(df2.values==skips[j]):
                  flg = 1
        if flg == 0:
          skips.append(df2.values)
          df_sub = df2
          df_sub.insert(0,'time_index', i)
          spectral = pd.concat([spectral, df_sub])
    return spectral

def beta_marquardt(
    grb, 
    path,
    z,
    print_status = True,
    save_in_folder = 'beta/'
):
    
    # This function provides the Spectral Energy Distribution fitting following the Equation 3 of https://arxiv.org/pdf/2405.02263
    # The fitting parameters are: beta (optical spectral index), A_V (host extinction in V-band), and the intercept
    # For the spectral shape, a power-law is assumed, in combination with the host extinction galaxy through the maps of Pei (1992)
    # This function takes as input the GRB dataframe ("path") and the redshift ("z")
    # The cases with uncertainty on the beta greater than the beta values themselves are excluded
    # In the comparison between the 3 host models (SMC,LMC,MW) for each time epoch, the most probable fitting is the chosen one
    # The function gives as output the "dfexp" variable that contains all the fitting details

    if save_in_folder:
        if not os.path.exists(save_in_folder):
            os.mkdir(save_in_folder)

    if print_status:
        print("GRB = ", grb, ", at z =", z, ", host galaxy model (Pei 1992)")
        
    plotnumber = 0

    gamma = []
    beta_bf_marquardt = []
    beta_bf_marquardt_err = []

    dataframegrb = []
    dataframebetas = []
    dataframebetaerrors = []
    dataframeAV = []
    dataframeAVerrors = []
    dataframegalmodel = []
    dataframetimes = []
    dataframefilters = []
    dataframeredchi2 = []
    dataframeintercept = []
    dataframeintercepterrors = []
    dataframersquared = []
    dataframeprobability = []
    dataframeoutliersources = []
    dataframeplotnumber = []

    log_lam_marquardt_outlier = []
    mag_marquardt_outlier = []
    mag_err_marquardt_outlier = []
    filters_list_outlier = []
    timebands_list_outlier = []
    source_list_outlier = []
    telescope_list_outlier = []

    gooddata_source = []
    gooddata_telescope = []
            
    # create table of matching times
    df = maketable(path=path)

    if print_status:
        print(df)
    if len(df) != 0:
        df = df[df["mag_err"] != 0]
        iters = [*set(df["time_index"].values)]
        bands = [*set(df["band"].values)]
        sources = [*set(df["source"].values)]
        finaltimeslist = []
        finalspectralist = []
        
        for i in iters:
            plotnumber = plotnumber + 1
            spectral = df.loc[df["time_index"] == i]
            timespectra = spectral["time_sec"].tolist()
            timebands = spectral["band"].tolist()
            sources = spectral["source"].tolist()
            timespectra = list(set(timespectra))
            timebands = list(set(timebands))
            finaltimeslist.append(timespectra)
            finalspectralist.append(timebands)
            plotnumbersublist = []
            mag = []
            mag_err = []
            log_lam = []
            filters_list = []
            timebands_list = []
            sources_list = []
            telescope_list = []
            for band in bands:
                band_slice = spectral.loc[spectral["band"] == band]
                log_lams = np.log10(band_slice["lambda"].values)
                mags = band_slice["mag_corr"].values
                magerrs = band_slice["mag_err"].values
                sourcelabels = band_slice["source"].values
                telescopelabels = band_slice["telescope"].values
                filters_list.extend(band_slice["band"].values)
                timebands_list.extend(band_slice["time_sec"].values)
                for jj in range(len(magerrs)):
                    if magerrs[jj] != 0.0:
                        log_lam.append(log_lams[jj])
                        mag.append(mags[jj])
                        mag_err.append(magerrs[jj])
                        sources_list.append(sourcelabels[jj])
                        telescope_list.append(telescopelabels[jj])
            
            if len(set(filters_list)) > 3:

                X = log_lam
                y = mag
                weights_list = [1 / err for err in mag_err] 

                # Weights are 1/err since lmfit makes residuals=weights*(data-model) and then minimizes the square of the residuals https://stackoverflow.com/questions/58251958/take-errors-on-data-into-account-when-using-lmfit
                # Look also in https://lmfit.github.io/lmfit-py/model.html
                
                modelSMC = Model(model_SMC, independent_vars=['x'])
                parsSMC = Parameters()
                
                parsSMC.add('intercept', value=40, min=0, max=100)
                parsSMC.add('beta', value=0.8, min=-10, max=10)
                parsSMC.add('AV', value=1, min=0, max=10)
                parsSMC.add('z', value=z)
                parsSMC['z'].vary = False

                if print_status:
                    print("Fitting SMC model...")
                resultsSMC = modelSMC.fit(y, parsSMC, x=X, weights=weights_list)                       
                chisquareSMC = resultsSMC.chisqr
                rsquaredSMC = resultsSMC.rsquared
                redchisquareSMC = resultsSMC.redchi
                slopeSMC = resultsSMC.params.get("beta").value
                slopeerrSMC = resultsSMC.params.get("beta").stderr
                avfitSMC = resultsSMC.params.get("AV").value
                avfiterrSMC = resultsSMC.params.get("AV").stderr
                interceptfitSMC = resultsSMC.params.get("intercept").value
                interceptfiterrSMC = resultsSMC.params.get("intercept").stderr

                nuSMC=len(y)
                xxSMC=redchisquareSMC*nuSMC
                probSMC=(2**(-nuSMC/2)/math.gamma(nuSMC/2))*scipy.integrate.quad(lambda x: math.exp(-x/2)*x**(-1+(nuSMC/2)),xxSMC,np.inf)[0]

                modelLMC = Model(model_LMC, independent_vars=['x'])
                parsLMC = Parameters()
                
                parsLMC.add('intercept', value=40, min=0, max=100)
                parsLMC.add('beta', value=0.8, min=-10, max=10)
                parsLMC.add('AV', value=1, min=0, max=10)
                parsLMC.add('z', value=z)
                parsLMC['z'].vary = False

                if print_status:
                    print("Fitting LMC model...")
                resultsLMC = modelLMC.fit(y, parsLMC, x=X, weights=weights_list)                       
                chisquareLMC = resultsLMC.chisqr
                rsquaredLMC = resultsLMC.rsquared
                redchisquareLMC = resultsLMC.redchi
                slopeLMC = resultsLMC.params.get("beta").value
                slopeerrLMC = resultsLMC.params.get("beta").stderr
                avfitLMC = resultsSMC.params.get("AV").value
                avfiterrLMC = resultsLMC.params.get("AV").stderr
                interceptfitLMC = resultsLMC.params.get("intercept").value
                interceptfiterrLMC = resultsLMC.params.get("intercept").stderr

                nuLMC=len(y)
                xxLMC=redchisquareLMC*nuLMC
                probLMC=(2**(-nuLMC/2)/math.gamma(nuLMC/2))*scipy.integrate.quad(lambda x: math.exp(-x/2)*x**(-1+(nuLMC/2)),xxLMC,np.inf)[0]

                modelMW = Model(model_MW, independent_vars=['x'])
                parsMW = Parameters()
                
                parsMW.add('intercept', value=40, min=0, max=100)
                parsMW.add('beta', value=0.8, min=-10, max=10)
                parsMW.add('AV', value=1, min=0, max=10)
                parsMW.add('z', value=z)
                parsMW['z'].vary = False

                if print_status:
                    print("Fitting MW model...")
                resultsMW = modelMW.fit(y, parsMW, x=X, weights=weights_list)                       
                chisquareMW = resultsMW.chisqr
                rsquaredMW = resultsMW.rsquared
                redchisquareMW = resultsMW.redchi
                slopeMW = resultsMW.params.get("beta").value
                slopeerrMW = resultsMW.params.get("beta").stderr
                avfitMW = resultsMW.params.get("AV").value
                avfiterrMW = resultsMW.params.get("AV").stderr
                interceptfitMW = resultsMW.params.get("intercept").value
                interceptfiterrMW = resultsMW.params.get("intercept").stderr

                nuMW=len(y)
                xxMW=redchisquareMW*nuMW
                probMW=(2**(-nuMW/2)/math.gamma(nuMW/2))*scipy.integrate.quad(lambda x: math.exp(-x/2)*x**(-1+(nuMW/2)),xxMW,np.inf)[0]

                probhostmodels=[1-probMW, 1-probLMC, 1-probSMC]

                pivot=np.where(probhostmodels == np.min(np.abs(probhostmodels)))[0][0]

                if pivot==0:
                    hostmodel="MW"
                    galnumber=1
                    RVnumber=3.08
                
                    chisquare = chisquareMW
                    rsquared = rsquaredMW
                    redchisquare = redchisquareMW
                    slope = slopeMW
                    slopeerr = slopeerrMW
                    avfit = avfitMW
                    avfiterr = avfiterrMW
                    interceptfit = interceptfitMW
                    interceptfiterr = interceptfiterrMW

                    prob = probMW

                if pivot==1:
                    hostmodel="LMC"
                    galnumber=2
                    RVnumber=3.16
                
                    chisquare = chisquareLMC
                    rsquared = rsquaredLMC
                    redchisquare = redchisquareLMC
                    slope = slopeLMC
                    slopeerr = slopeerrLMC
                    avfit = avfitLMC
                    avfiterr = avfiterrLMC
                    interceptfit = interceptfitLMC
                    interceptfiterr = interceptfiterrLMC

                    prob = probMW
                
                if pivot==2:
                    hostmodel="SMC"
                    galnumber=3
                    RVnumber=2.93
                
                    chisquare = chisquareSMC
                    rsquared = rsquaredSMC
                    redchisquare = redchisquareSMC
                    slope = slopeSMC
                    slopeerr = slopeerrSMC
                    avfit = avfitSMC
                    avfiterr = avfiterrSMC
                    interceptfit = interceptfitSMC
                    interceptfiterr = interceptfiterrSMC

                    prob = probSMC

                if print_status:
                    print("The selected model according to its probability is ",hostmodel)
                
                reject_plot = False

                if slope < 0:
                    reject_plot = False

                if print_status:
                    print(plotnumber,prob)
                
                # 26feb commented
                if abs(slopeerr) > abs(slope):
                    reject_plot = True

                # if abs(avfiterr) > abs(avfit):
                #     reject_plot = True
                
                if avfit < 0:
                    reject_plot = True

                # if abs(interceptfiterr) > abs(interceptfit):
                #     reject_plot = True

                y_marquardtforoutliers = [interceptfit - 2.5*slope*xi -2.5*(pei_av(10**xi,A_V=avfit,gal=galnumber,R_V=RVnumber)-pei_av((10**xi)/(1+z),A_V=avfit,gal=galnumber,R_V=RVnumber))  for xi in X]
                
                for i in range(len(mag)):

                    if abs(y[i] - y_marquardtforoutliers[i]) <= 3 * mag_err[i]:
                        gooddata_source.append(sources_list[i])
                        gooddata_telescope.append(telescope_list[i])
                
                for i in range(len(mag)):
                    
                    if abs(y[i] - y_marquardtforoutliers[i]) > 3 * mag_err[i]:
                        log_lam_marquardt_outlier.append(log_lam[i])
                        mag_marquardt_outlier.append(mag[i])
                        mag_err_marquardt_outlier.append(mag_err[i])
                        filters_list_outlier.append(filters_list[i])
                        timebands_list_outlier.append(timebands_list[i])
                        source_list_outlier.append(sources_list[i])
                        telescope_list_outlier.append(telescope_list[i])

                        if print_status:
                            print("Single outlier telescope")
                            print(telescope_list[i])
                        
                        if (telescope_list[i] in gooddata_telescope) and (np.char.isnumeric(sources_list[i])):
                            warningstring=str(grb)+" "+str(telescope_list[i])+" "+str(sources_list[i])
                            if print_status:
                                print("WARNING ",warningstring)

                            with open(str(grb)+'_warnings.txt', 'a') as warnfile:
                                warnfile.write(warningstring)
                                warnfile.write('\n')
                                warnfile.close()
                                
                if len(log_lam_marquardt_outlier)==0:
                    outliersprint="NaN"
                    outlierlabelpath=""
                else:
                    outliersprint="_".join(set(source_list_outlier))
                    outlierlabelpath="_outliers"
                
                for band_now in set(bands):
                    if reject_plot:
                        break
                    
                    main_points_log_lam = log_lam[timebands_list == band_now]
                    
                    try:
                        #If there is only 1 point or No points corresponding to band_now, then proceed to the next band
                        if len(main_points_log_lam) <= 1:
                            continue
                        else:
                            pass
                        
                    except:
                        continue
                                    
                    main_points_mag = mag[timebands_list == band_now]
                    main_points_mag_err = mag_err[timebands_list == band_now]
                    sorted_indices = np.argsort(main_points_mag)
                    point_set_sorted = main_points_mag[sorted_indices]
                    point_set_err_sorted = main_points_mag_err[sorted_indices]
                    
                    for iii in range(len(point_set_sorted) - 1):
                        point_x = point_set_sorted[iii]
                        point_x_plus_1 = point_set_sorted[iii+1]
                        point_x_err = point_set_err_sorted[iii]
                        point_x_plus_1_err = point_set_err_sorted[iii+1]
                        
                        if abs(point_x - point_x_plus_1) > abs(point_x_err) + abs(point_x_plus_1_err):
                            # Reject plot and go to next fit
                            reject_plot = True
                            break
                    
                    if reject_plot:
                        # Even if one band has points separated beyond 1 - sigma, this fit IS REJECTED
                        break
                    
                
                if reject_plot:
                    continue       
                
                betaoldmarquardt = slope
                betaoldmarquardterr = slopeerr if slopeerr is not None else np.inf
                gamma.append(grb)
                beta_bf_marquardt.append(np.negative(betaoldmarquardt))
                beta_bf_marquardt_err.append(betaoldmarquardterr)

                if print_status:
                    print(
                        "Marquardt Levenberg Beta:",
                        slope,
                        "Marquardt Levenberg Beta error:",
                        betaoldmarquardterr,
                    )

                X = np.sort(X)
                y_marquardt = [interceptfit - 2.5*slope*xi -2.5*(pei_av(10**xi,A_V=avfit,gal=galnumber,R_V=RVnumber)-pei_av((10**xi)/(1+z),A_V=avfit,gal=galnumber,R_V=RVnumber))  for xi in X]                        
                fig, ax = plt.subplots()
                ax.invert_yaxis()
                plt.plot(X, y_marquardt, label=r"$\beta_{opt}:\,$"+str(round(slope,3))+"+/-"+str(round(betaoldmarquardterr,3))+'\n'+
                        "$A_{V}:\,$"+str(round(avfit,3))+"+/-"+str(round(avfiterr,3))+'\n'+str(hostmodel))
                plt.errorbar(log_lam, mag, yerr=mag_err, fmt="o")
                plotnumbersublist.append(plotnumber)
                plt.xlabel(r'$\log_{10}\lambda (\AA)$')
                plt.ylabel("Magnitude")
                
                if min(timebands_list) == max(timebands_list):
                    plt.title(
                        'Bands=' + str(set(filters_list))
                        + '\n' + 'log10(t)_min-max='
                        + str(round(min(timebands_list), 3))
                        + '-' + str(round(max(timebands_list), 3))
                        + '\n' + r'$\chi^{2}=$' + str(round(chisquare, 2))
                        + ',Red. ' + r'$\chi^{2}=$' + str(round(redchisquare, 2))
                        + ',Prob. ' + str(round(prob, 3)))
                    plt.legend()
                    fig.tight_layout()
                    fig.text(0.1, 0.1, "GRB " + str(grb), fontsize=18, fontweight='bold', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

                    if save_in_folder: 
                        plt.savefig(
                            str(grb) + "_beta"
                            + str(round(slope, 3))
                            + "_betaerr"
                            + str(round(betaoldmarquardterr, 3))
                            + "_plotn"
                            + str(plotnumber)
                            + outlierlabelpath
                            + ".png")
                    plt.clf()

                    dataframegrb.append(str(grb))
                    dataframebetas.append(str(round(slope,3)))
                    dataframebetaerrors.append(str(round(betaoldmarquardterr, 3)))
                    dataframeAV.append(str(round(avfit,3)))
                    dataframeAVerrors.append(str(round(avfiterr,3)))
                    dataframegalmodel.append(str(hostmodel))
                    dataframeintercept.append(str(round(interceptfit,3)))
                    dataframeintercepterrors.append(str(round(interceptfiterr,3)))
                    dataframetimes.append(str(round(min(timebands_list), 4)))
                    dataframefilters.append(str(set(filters_list)))
                    dataframeredchi2.append(str(round(redchisquare, 2)))
                    dataframersquared.append(rsquared)
                    dataframeprobability.append(prob)
                    dataframeoutliersources.append(outliersprint)
                    dataframeplotnumber.append(str(plotnumber))

                else:
                    plt.title(
                        'Bands=' + str(set(filters_list))
                        + '\n' + 'log10(t)_min-max='
                        + str(round(min(timebands_list), 3))
                        + '-' + str(round(max(timebands_list), 3))
                        + '\n' + r'$\chi^{2}=$' + str(round(chisquare, 2))
                        + ',Red. ' + r'$\chi^{2}=$' + str(round(redchisquare, 2))
                        + ',Prob. ' + str(round(prob, 3)))
                    plt.legend()
                    fig.tight_layout()
                    fig.text(0.1, 0.1, "GRB " + str(grb), fontsize=18, fontweight='bold', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
                    
                    if save_in_folder: 
                        plt.savefig(
                        str(grb) + "_beta"
                        + str(round(slope, 3))
                        + "_betaerr"
                        + str(round(betaoldmarquardterr, 3))
                        + "_plotn"
                        + str(plotnumber)
                        + outlierlabelpath
                        + ".png")
                    plt.clf()

                    dataframegrb.append(str(grb))
                    dataframebetas.append(str(round(slope,3)))
                    dataframebetaerrors.append(str(round(betaoldmarquardterr, 3)))
                    dataframeAV.append(str(round(avfit,3)))
                    dataframeAVerrors.append(str(round(avfiterr,3)))
                    dataframegalmodel.append(str(hostmodel))
                    dataframeintercept.append(str(round(interceptfit,3)))
                    dataframeintercepterrors.append(str(round(interceptfiterr,3)))
                    dataframetimes.append(str(round(min(timebands_list), 4))+"-"+str(round(max(timebands_list), 4)))
                    dataframefilters.append(str(set(filters_list)))
                    dataframeredchi2.append(str(round(redchisquare, 2)))
                    dataframersquared.append(str(rsquared))
                    dataframeprobability.append(prob)
                    dataframeoutliersources.append(outliersprint)
                    dataframeplotnumber.append(str(plotnumber))

            else:
                if print_status:
                    print("Not 4 different bands at least for the fitting")

        else:
            if print_status:
                print("No matching GRB found in the dataset.")


        dict = {"GRB": dataframegrb, "beta": dataframebetas, "beta_err": dataframebetaerrors, "AV": dataframeAV, "AV_err": dataframeAVerrors,
                "Gal.model": dataframegalmodel, "intercept": dataframeintercept, "intercept_err": dataframeintercepterrors, "log10t": dataframetimes, 
                    "filters": dataframefilters, "Red.Chi2": dataframeredchi2, "R-squared": dataframersquared, "probability": dataframeprobability, "outliers": dataframeoutliersources, "plotnumb": dataframeplotnumber}
                        
        dfexp = pd.DataFrame(dict, columns=["GRB","beta","beta_err","AV","AV_err","Gal.model","intercept","intercept_err","log10t","filters","Red.Chi2","R-squared","probability","outliers","plotnumb"]) # "redchi2"

        if save_in_folder: 
            dfexp.to_csv(str(grb)+'_sed_results.txt', sep='\t')

    return dfexp
