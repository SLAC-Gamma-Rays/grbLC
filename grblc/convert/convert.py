import os
import re

import numpy as np
import pandas as pd
import astropy.units as u

from .constants import *
from .match import *
from grblc.evolution import io
from .sfd import data_dir
from .extinction import *


# conversion for mag in AB and corrected for extinction
@np.vectorize
def toAB(grb: str,
         ra: str,
         dec: str,
         band: str,
         system: str,
         extcorr: str,
         telescope: str,
         mag: float):

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


# main conversion function to call
def correctGRB(
    grb: str,
    ra: str,
    dec: str,
    path: str = None,
    save_in_folder: str = None,
    debug: bool = False,
):
    
    assert bool(grb and ra and dec), "Must provide either grb name or location."

    # try to import magnitude table to convert
    try:
        global directory
        mag_table = io.read_data(path, approximate_band=True)
    except ValueError as error:
        raise error
    except IndexError:
        raise ImportError(message=f"Couldn't find grb table at {path}.")

    converted = {k: [] for k in ('time_sec', 'mag', 'mag_err', 'band', 'system', 'telescope', 'extcorr', 'source')}

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
                'mag_source'
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

        # attempt to convert a single magnitude to flux given a band, position in the sky, mag_err, and photon index
        try:
            mag_corr, filter_id, coeff_source = toAB(
                grb,
                ra,
                dec,
                band,
                system,
                extcorr,
                telescope,
                mag
                )
        except KeyError as error:
            print("KeyError")
            continue

        converted['time_sec'].append(time_sec)
        converted['mag'].append(mag_corr)
        converted['mag_err'].append(mag_err)
        converted['band'].append(band)
        converted['system'].append("AB")
        converted['telescope'].append(telescope)
        converted['extcorr'].append("y")
        converted['source'].append(source)

        # verbosity if you want it
        if debug:
            converted_debug['time_sec'].append(time_sec)
            converted_debug['mag_corr'].append(mag_corr)
            converted_debug['mag_init'].append(mag)
            converted_debug['mag_err'].append(mag_err)
            converted_debug['band_init'].append(band)
            converted_debug['band_match'].append(filter_id)
            converted_debug['system_init'].append(system)
            converted_debug['system_final'].append("AB")
            converted_debug['telescope'].append(system)
            converted_debug['extcorr'].append("y")
            converted_debug['coeff_source'].append(coeff_source)
            converted_debug['mag_source'].append(source)

    # after converting everything, go from dictionary -> DataFrame -> csv!
    if not debug:
        save_path = os.path.join(save_in_folder+'/', f"{grb}_magAB_extcorr.txt")
        pd.DataFrame.from_dict(converted).to_csv(save_path, sep='\t', index=False)
    else:
        save_path = os.path.join(save_in_folder+'/', f"{grb}_magAB_extcorr_DEBUG.txt")
        pd.DataFrame.from_dict(converted_debug).to_csv(save_path, sep='\t', index=False)

# 05 April 2024: correction for host extinction and k-correction

def hostpeikcorr(
    grb: str,
    data: str,
    band: str,
    telescope: str,
    time_sec: float,
    mag: float,
    magerr: float,
    kcorr = True,
    hostcorr = True
):
    
    
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
        #print('k-correction: '+str(kcorr))
        kcorrerr = (2.5*np.log10(1+zneeded)*betaneedederr)**2 # already squared
        #print('k-correction error: '+str(kcorrerr))
        
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

# main conversion function to call
def KcorrecthostGRB(
    grb: str,
    #ra: str,
    #dec: str,
    filename: str = None,
    save_in_folder: str = None,
    debug: bool = False,
):

    # assign column names and datatypes before importing
    dtype = {
        'time_sec': np.float64,
        'mag': np.float64,
        'mag_err': np.float64,
        'band': str,
        'system': str,
        'telescope': str,
        'extcorr': str,
        'source': str,
        'flag': str,
    }
    names = list(dtype.keys())

    # try to import magnitude table to convert
    try:
        global directory
        mag_table = pd.read_csv(
            filename,
            delimiter=r'\t+|\s+',
            names=names,
            dtype=dtype,
            index_col=None,
            header=0,
            engine='python',
            encoding='ISO-8859-1'
        )
    except ValueError as error:
        raise error
    except IndexError:
        raise ImportError(message=f"Couldn't find grb table at {filename}.")

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
            mag_corr, mag_err_corr, filter_id = hostpeikcorr(
                grb,
                band,
                telescope,
                time_sec,
                mag,
                mag_err,
                kcorr = True,
                hostcorr = True
                )
            
        except KeyError as error:
            print("KeyError")
            continue


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
            converted_debug['band_init'].append(band)
            converted_debug['band_match'].append(filter_id)
            converted_debug['system'].append(system)
            converted_debug['telescope'].append(telescope)
            converted_debug['extcorr'].append(extcorr)
            converted_debug['mag_source'].append(source)
            converted_debug['flag'].append(flag)

    # after converting everything, go from dictionary -> DataFrame -> csv!
    if not debug:
        save_path = os.path.join(save_in_folder+'/', f"{grb}_corrected.txt")
        pd.DataFrame.from_dict(converted).to_csv(save_path, sep='\t', index=False)
    else:
        save_path = os.path.join(save_in_folder+'/', f"{grb}_corrected_DEBUG.txt")
        pd.DataFrame.from_dict(converted_debug).to_csv(save_path, sep='\t', index=False)


# simple checker that downloads the SFD dust map if it's not already there
def _check_dust_maps():
    if not os.path.exists(os.path.join(data_dir(), "sfd")):
        from .sfd import sfd
        sfd.fetch()


# sets directory to the current working directory, or whatever folder you're currently in
directory = os.getcwd()
