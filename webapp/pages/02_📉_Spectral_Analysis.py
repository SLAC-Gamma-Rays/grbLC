import streamlit as st
import sys
sys.path.append('D:/naoj-grb/grbLC/')
from grblc.evolution.sed import beta_marquardt

#from ..utils import _load_data
import pandas as pd

@st.cache_data  # atlernative to @st.cache for dataframes
def _load_data(path, sep=",", dtype=None, names=None, header=None, index_col=None):
    ''' Reads the list of GRBs'''
    data =  pd.read_csv(path, 
                        sep=sep, 
                        engine='python', 
                        dtype=dtype,
                        names=names,
                        header=header, 
                        index_col=index_col, 
                        #encoding='ISO-8859-1'
                        )
    return data

st.markdown("Spectral Analysis")
if 'select_event' not in st.session_state:
    st.session_state['select_event'] = "970228A"

data = _load_data(path='grblist.csv', 
                 sep=',', 
                 dtype=None,
                 names=None,
                 header=0, 
                 index_col="GRB"
                 )

st.sidebar.markdown("## Select GRB")
st.session_state['select_event'] = st.sidebar.selectbox("*Mandatory field", data.index)

path = 'mag-AB-extcorr-30-05-2023/'+st.session_state['select_event']+'_magAB_extcorr.txt'
redshift = data.loc[data.index == st.session_state['select_event'], "Redshift"].to_numpy()[0]

beta_marquardt(
    grb=st.session_state['select_event'], 
    filename=path, 
    z= redshift
    )