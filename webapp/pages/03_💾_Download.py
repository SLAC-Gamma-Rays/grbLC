import streamlit as st

import os
import shutil
import glob
from zipfile import ZipFile

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

st.markdown("# Download")

# Load data

@st.cache_data
def load_data():
    ''' Reads the list of GRBs'''
    data = pd.read_csv(
        "grblist.csv", delimiter=","
    )  # grb attributes list
    return data
data = load_data()


# Cleaning download directory on each new session

if 'filenameBULK' not in st.session_state: 
    if os.path.exists('BD/') == True:
        shutil.rmtree('BD/')

# Required functions

def relationCHECKER(RAmax, RAmin, DECmax, DECmin, Zmax, Zmin):
    if (RAmax>RAmin) & (DECmax>DECmin) & (Zmax>Zmin):
        return 1
    else:
        return 0


def download_option(data, RAmax, RAmin, DECmax, DECmin, Zmax, Zmin):
        if (np.sum(np.array([RAmax,RAmin,DECmax,DECmin,Zmax,Zmin]) == '-') >0) and not (relationCHECKER(RAmax,RAmin,DECmax,DECmin,Zmax,Zmin)):
            st.write("Enter proper values!")
        else:
            downloadDATA = data.loc[(data.RA_deg>RAmin) & (data.RA_deg<RAmax) & (data.DEC_deg>DECmin) & (data.DEC_deg<DECmax) & (data.z>=Zmin) & (data.z<=Zmax)]
            LCfiles = glob.glob('mastersample/*.txt')
            filenameBULK = 'BD/'+str(np.round(RAmin,decimals=1)) + str(np.round(RAmax,decimals=1)) + '_'  + str(np.round(DECmin,decimals=1)) + str(np.round(DECmax,decimals=1)) + '_' + str(np.round(Zmin,decimals=1)) + str(np.round(Zmax,decimals=1)) + '.zip'
            
            try:
                os.mkdir('BD')
            except FileExistsError as e:
                pass
            
            with ZipFile(filenameBULK, 'w') as zipObj2:
                for i,row in downloadDATA.iterrows():
                    zipObj2.write(row.path,)

            return filenameBULK, len(downloadDATA)


# Connecting to streamlit

c = SkyCoord(data.RA.to_numpy(),data.DEC.to_numpy(),frame='icrs')

data['RA_deg'] = c.ra.deg
data['DEC_deg'] = c.dec.deg
min_z = min(data.Redshift.to_numpy())
max_z = max(data.Redshift.to_numpy())

if 'filename' not in st.session_state:
    st.session_state.filenameBULK = None

with st.form("my_form"):
    RAmin,RAmax = st.slider(
            'Right Ascension',
            0.0, 360.0, step=0.0001, value =(0.0, 360.0))

    DECmin,DECmax = st.slider(
            'Declination',
            -90.0, 90.0, step=0.0001, value =(-90.0, 90.0))
    Zmin,Zmax = st.slider(
            'Redshift',
            min_z, max_z, step=0.0001, value =(float(min_z), float(max_z)))
    submitted = st.form_submit_button("Generate Download Link")
    if submitted:
        st.session_state.filenameBULK, length = download_option(data, RAmax,RAmin,DECmax,DECmin,Zmax,Zmin)        


if st.session_state.filenameBULK != None:
    with open(st.session_state.filenameBULK, 'rb') as f:
        st.write(str(length) + ' LC files found, Click the button below to download the zip file.')
        #st.write(st.session_state.filenameBULK)
        st.download_button('Download Zip', f, file_name=st.session_state.filenameBULK)