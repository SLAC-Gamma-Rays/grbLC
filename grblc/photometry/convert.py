import os
import re

import numpy as np
import pandas as pd

from .constants import *
from .match import *
from ..io import read_data
from .sfd import data_dir
from .extinction import *


def _toAB(
    grb: str,
    ra: str,
    dec: str,
    band: str,
    system: str,
    extcorr: str,
    telescope: str,
    mag: float
):
    """
    Corrects individual magnitude points.
    """

    try:
        lambda_x, shift_toAB, coeff, filter_id, coeff_source = calibration(band, telescope)

    except KeyError:
        raise KeyError(f"Band '{band}' of telescope '{telescope}' is not currently supported.")

    # get correction for galactic extinction to be added to magnitude if not already supplied
    if extcorr == 'y':
        mag_corr = mag    

    else:
        A_x = coeff * ebv(grb, ra, dec)
        mag_corr = mag - A_x

    if 'ab' or 'sdss' or 'panstarrs' in system.casefold:
        mag_corr = mag_corr

    else:
        if shift_toAB != 'notfound':
            mag_corr = mag_corr + float(shift_toAB)
                    
        else:
            for i in defaultshift_toAB.keys():
                if band==i:
                    shiftAB=defaultshift_toAB[i]
                    break
            mag_corr = mag_corr + float(shiftAB)

    return mag_corr, filter_id, coeff_source


# Conversion to AB system and galactic extinction correction
def _convertGRB(
    grb: str = None,
    ra: str = None,
    dec: str = None,
    mag_table: pd.DataFrame = None,
    save_in_folder: str = None,
    debug: bool = False,
):
    """
    Function to convert GRB magnitudes to AB system and correct for galactic extinction.

    """

    converted = {k: [] for k in ('time_sec', 'mag', 'mag_err', 'band', 'system', 'telescope', 'extcorr', 'source', 'flag')}

    if debug:
        converted_debug = {
            k: []
            for k in (
                'time_sec',
                'mag_corr',
                'mag_init',
                'mag_err',
                'band',
                'band_match',
                'system_init',
                'system_final',
                'telescope',
                'extcorr',
                'coeff_source'
                'mag_source',
                'flag'
            )
        }

    for __, row in mag_table.iterrows():

        time_sec = row['time_sec']
        mag = row['mag']
        mag_err = row['mag_err']
        system = row['system']
        band = row['band']
        telescope = row['telescope']
        extcorr = row['extcorr']
        source = row['source']
        flag = row['flag']

        # attempt to convert a single magnitude to flux given a band, position in the sky, mag_err, and photon index
        try:
            mag_corr, filter_id, coeff_source = _toAB(
                grb,
                ra,
                dec,
                band,
                system,
                extcorr,
                telescope,
                mag
                )

            converted['time_sec'].append(time_sec)
            converted['mag'].append(mag_corr)
            converted['mag_err'].append(mag_err)
            converted['band'].append(band)
            converted['system'].append("AB")
            converted['telescope'].append(telescope)
            converted['extcorr'].append("y")
            converted['source'].append(source)
            converted['flag'].append(flag)

            # verbosity if you want it
            if debug:
                converted_debug['time_sec'].append(time_sec)
                converted_debug['mag_corr'].append(mag_corr)
                converted_debug['mag_init'].append(mag)
                converted_debug['mag_err'].append(mag_err)
                converted_debug['band'].append(band)
                converted_debug['band_match'].append(filter_id)
                converted_debug['system_init'].append(system)
                converted_debug['system_final'].append("AB")
                converted_debug['telescope'].append(system)
                converted_debug['extcorr'].append("y")
                converted_debug['coeff_source'].append(coeff_source)
                converted_debug['mag_source'].append(source)
                converted_debug['flag'].append(flag)

        except KeyError as error:
            print(f"Filter {band} of {telescope} not found for the data point at {time_sec} seconds.")
            continue

    # after converting everything, go from dictionary -> DataFrame -> csv!
    if not debug:
        converted_df = pd.DataFrame.from_dict(converted)

        if save_in_folder:
            save_path = os.path.join(save_in_folder+'/', f"{grb}_khost_corrected.txt")
            
    else:
        converted_df = pd.DataFrame.from_dict(converted_debug)

        if save_in_folder:
            save_path = os.path.join(save_in_folder+'/', f"{grb}_khost_corrected_DEBUG.txt")
            
    converted_df.to_csv(save_path, sep='\t', index=False)
        
    return converted_df


def _hostpei_kcorr(
    grb: str,
    sed_results: str,
    band: str,
    telescope: str,
    time_sec: float,
    mag: float,
    magerr: float,
    kcorr = True,
    hostcorr = True
):
    """
    Function to correct for host extinction and perform k-correction for individual magnitudes.

    """

    try:
        data = sed_results.loc[grb]
        data = data.reset_index()
    except KeyError:
        raise KeyError(f"GRB '{grb}' is not present in the sample.")    
    
    if kcorr:   
        if data.shape[1]!=13:
            betaneeded=data.iloc[6][1]
            betaneedederr=data.iloc[7][1]
            AVneeded=data.iloc[8][1]
            AVneedederr=data.iloc[9][1]
            modelneeded=data.iloc[10][1]
            zneeded=data.iloc[11][1]
            stopcondition=True

        else:
            firstline=data.iloc[0]
            lastline=data.iloc[-1]
            
            stopcondition=False

            while stopcondition==False:

                if time_sec <= firstline["tmin"]:
                    betaneeded=firstline["betaavg"]
                    betaneedederr=firstline["betaavg_err"]
                    AVneeded=firstline["AV"]
                    AVneedederr=firstline["AV_err"]
                    modelneeded=firstline["bestmodel"]
                    zneeded=firstline["z"]
                    stopcondition=True

                if time_sec >= lastline["tmax"]:
                    betaneeded=lastline["betaavg"]
                    betaneedederr=lastline["betaavg_err"]
                    AVneeded=lastline["AV"]
                    AVneedederr=lastline["AV_err"]
                    modelneeded=lastline["bestmodel"]
                    zneeded=lastline["z"]
                    stopcondition=True
                
                for i in range(len(data)):
                    
                    if data.iloc[i]["tmin"]<=time_sec<=data.iloc[i]["tmax"]:
                        betaneeded=data.iloc[i]["betaavg"]
                        betaneedederr=data.iloc[i]["betaavg_err"]
                        AVneeded=data.iloc[i]["AV"]
                        AVneedederr=data.iloc[i]["AV_err"]   
                        modelneeded=data.iloc[i]["bestmodel"]  
                        zneeded=data.iloc[i]["z"]       
                        stopcondition=True
                    
                for i in range(len(data)-1):
                    
                    if (time_sec>data.iloc[i]["tmax"]) and (time_sec<data.iloc[i+1]["tmin"]):
                        betaneeded=data.iloc[i]["betaavg"]
                        betaneedederr=data.iloc[i]["betaavg_err"]
                        AVneeded=data.iloc[i]["AV"]
                        AVneedederr=data.iloc[i]["AV_err"]
                        modelneeded=data.iloc[i]["bestmodel"]
                        zneeded=data.iloc[i]["z"]             
                        stopcondition=True

        kcorr = 2.5*(betaneeded-1)*np.log10(1+zneeded)
        kcorrerr = (2.5*np.log10(1+zneeded)*betaneedederr)**2 # already squared        
  
    else:
        kcorr = 0
        kcorrerr = 0

    if hostcorr:
    
        model = modelneeded
        AV = AVneeded
        AVerr = AVneedederr
        
        if model == 'negligible':
            AV = 0
            AVerr = 0
            hostext = 0
            hostexterr = 0
            filter_id = 'None' # since no host correction 

        # if np.abs(AVerr) >= np.abs(AV):
        #     AV = 0
        #     AVerr = 0
        #     hostext = 0
        #     hostexterr = 0
        
        else:
            try:
                lambda_x, shift_toAB, coeff, filter_id, coeff_source = calibration(band, telescope)
            except KeyError:
                raise KeyError(f"Band '{band}' of telescope '{telescope}' is not currently supported.")

            AV = AVneeded
            AVerr = AVneedederr
            
            if model == 'MW':
                galaxy = 1
                RVvalue = 3.08
            elif model == 'LMC':
                galaxy = 2
                RVvalue = 3.16
            elif model == 'SMC':
                galaxy = 3
                RVvalue = 2.93

            else:
                raise KeyError(f"GRB '{grb}' has no selected dust model.")

            if model != 'negligible':
                if AVerr == '0' or AVerr == '-':
                    raise KeyError(f"GRB '{grb}' has no constrained A_V value for the host extinction.")

            if betaneeded == '0' or betaneedederr == '-':
                raise KeyError(f"GRB '{grb}' has no constrained beta_opt value.")

            AV = float(AV)
            AVerr = float(AVerr)

            x = 1/(lambda_x*10**-4) # 1/wavelength(micrometers)

            if x<0.21 or x>10:
                print('Warning: the host correction is performed outside of the empirical extinction curves boundaries. Results may not be reliable')

            # Considering a magnitude already in AB and corrected for Galactic extinction
            # The following equations correct it for host extinction and for K-correction

            hostext = pei_av(lambda_x,A_V=AV,gal=galaxy,R_V=RVvalue)*AV
            #print('host extinction in '+band+' band: '+str(hostext/AV))
            hostexterr = (pei_av(lambda_x,A_V=AV,gal=galaxy,R_V=RVvalue)*AVerr)**2 # already squared
            #print('host extinction error in '+band+' band: '+str(hostexterr))
    
    else:
        hostext = 0
        hostexterr = 0
        
    finalmag = mag - hostext - kcorr # the K-correction and host correction don't contain the negative sign
    
    if magerr==0:
        finalmagerr = magerr # for the limiting magnitudes, the error must be zero
    else:
        finalmagerr = np.sqrt(magerr**2 + hostexterr + kcorrerr) # hostexterr and kcorrerr are already squared
        
    return finalmag, finalmagerr, filter_id


def _host_kcorrectGRB(
    grb: str = None,
    mag_table: pd.DataFrame = None,
    sed_results: str = None,
    save_in_folder: str = None,
    debug: bool = False,
):
    """
    Function to perform host extinction correction and k-correction for redshift effects.

    """

    converted = {k: [] for k in ('time_sec', 'mag', 'mag_err', 'band', 'system', 'telescope', 'extcorr', 'source', 'flag')}

    if debug:
        converted_debug = {
            k: []
            for k in (
                'time_sec',
                'mag_corr',
                'mag_init',
                'mag_err',
                'band',
                'band_match',
                'system_init',
                'system_final',
                'telescope',
                'extcorr',
                'coeff_source',
                'mag_source',
                'flag'
            )
        }

    for __, row in mag_table.iterrows():

        time_sec = row['time_sec']
        mag = row['mag']
        mag_err = row['mag_err']
        system = row['system']
        band = row['band']
        telescope = row['telescope']
        extcorr = row['extcorr']
        source = row['source']
        flag = row['flag']

        # attempt to convert a single magnitude to flux given a band, position in the sky, mag_err, and photon index
        try:
            mag_corr, mag_err_corr, filter_id = _hostpei_kcorr(
                grb,
                sed_results,
                band,
                telescope,
                time_sec,
                mag,
                mag_err,
                kcorr = True,
                hostcorr = True
                )

            converted['time_sec'].append(time_sec)
            converted['mag'].append(mag_corr)
            converted['mag_err'].append(mag_err_corr)
            converted['band'].append(band)
            converted['system'].append(system)
            converted['telescope'].append(telescope)
            converted['extcorr'].append(extcorr)
            converted['source'].append(source)
            converted['flag'].append(flag)

            # verbosity if you want it
            if debug:
                converted_debug['time_sec'].append(time_sec)
                converted_debug['mag_corr'].append(mag_corr)
                converted_debug['mag_init'].append(mag)
                converted_debug['mag_err'].append(mag_err_corr)
                converted_debug['band'].append(band)
                converted_debug['band_match'].append(filter_id)
                converted_debug['system'].append(system)
                converted_debug['telescope'].append(telescope)
                converted_debug['extcorr'].append(extcorr)
                converted_debug['mag_source'].append(source)
                converted_debug['flag'].append(flag)

        except KeyError as error:
            print(f"Filter {band} of {telescope} not found for the data point at {time_sec} seconds.")
            continue

    # after converting everything, go from dictionary -> DataFrame -> csv!
    if not debug:
        converted_df = pd.DataFrame.from_dict(converted)

        if save_in_folder:
            save_path = os.path.join(save_in_folder+'/', f"{grb}_khost_corrected.txt")
            
    else:
        converted_df = pd.DataFrame.from_dict(converted_debug)

        if save_in_folder:
            save_path = os.path.join(save_in_folder+'/', f"{grb}_khost_corrected_DEBUG.txt")
            
    converted_df.to_csv(save_path, sep='\t', index=False)
        
    return converted_df


# simple checker that downloads the SFD dust map if it's not already there
def _check_dust_maps():
    if not os.path.exists(os.path.join(data_dir(), "sfd")):
        from .sfd import sfd
        sfd.fetch()


# sets directory to the current working directory, or whatever folder you're currently in
directory = os.getcwd()
