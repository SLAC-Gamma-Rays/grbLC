import streamlit as st
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

if __name__ == '__main__':
    _load_data()