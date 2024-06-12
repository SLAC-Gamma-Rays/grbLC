##############################################################################
# Imports
##############################################################################


import streamlit as st
import pandas as pd

import sys
sys.path.append('D:/naoj-grb/archive-grblc/grbLC-dev/')
#import grblc.evolution.io as io
from grblc.evolution.lightcurve import Lightcurve
# from streamlit_navigation_bar import st_navbar

from utils import _load_data


# Page config

apptitle = "GRBLC"

# page = st_navbar(["Quickview", "Colour Evolution", "Download", "Documentation", "GitHub"])#, urls=urls, options=options)
# st.write(page)
st.set_page_config(page_title=apptitle, layout='wide')

st.title("Gamma Ray Bursts Optical Afterglow Repository", anchor="main")


if 'select_event' not in st.session_state:
    st.session_state['select_event'] = "970228A"
if 'data_type' not in st.session_state:
    st.session_state['data_type'] = "_magAB_extcorr"

# App contents

data = _load_data(path='grblist.csv', 
                 sep=',', 
                 dtype=None,
                 names=None,
                 header=0, 
                 index_col="GRB"
                 )

st.sidebar.markdown("## Select GRB")
st.session_state['select_event'] = st.sidebar.selectbox("*Mandatory field", data.index)

## Information
st.sidebar.markdown("## Information")
st.markdown(
    """
<style>
[data-testid="stMetricValue"] {
    font-size: 25px;
}
</style>
""",
    unsafe_allow_html=True,
)
with st.container():
    st.sidebar.metric("## Right ascension",data.loc[data.index == st.session_state['select_event'], "RA"].to_numpy()[0])
    st.sidebar.metric("## Declination",data.loc[data.index == st.session_state['select_event'], "DEC"].to_numpy()[0])
    st.sidebar.metric("## Redshift",data.loc[data.index == st.session_state['select_event'], "Redshift"].to_numpy()[0])
    st.sidebar.metric("## Optical spectral index", str(data.loc[data.index == st.session_state['select_event'], "Beta"].to_numpy()[0]) + "+/-" + str(data.loc[data.index == st.session_state['select_event'], "Beta_Err"].to_numpy()[0]))
    st.sidebar.metric("## Class", str(data.loc[data.index == st.session_state['select_event'], "Class"].to_numpy()[0]))

plot = st.empty()
c1, c2 = st.columns([5,1])
path_file = 'mag-AB-extcorr-30-05-2023/'+st.session_state['select_event']+'_magAB_extcorr.txt'
format = c1.checkbox("Show raw data before homogenisation of photometric system and extinction correction")
if format:
    path_file = 'mag-30-05-2023/'+st.session_state['select_event']+'_mag.txt'
    st.session_state['data_type']  = "_mag"

## Create Lightcurve object
lc = Lightcurve(path = path_file,
                #data_space= 'lin',
                name = st.session_state['select_event'])

plot.plotly_chart(lc.displayGRB())

### Download

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(sep='\t').encode('utf-8')
txt = convert_df(lc.df)
c2.download_button(
        label="Download data",
        data=txt,
        file_name=st.session_state['select_event']+st.session_state['data_type']+'.txt',
        mime='text/csv',
    )

# import streamlit as st
# import pandas as pd

# import sys
# sys.path.append('D:/naoj-grb/archive-grblc/grbLC-dev/')
# from grblc.evolution.lightcurve import Lightcurve

if 'appx' not in st.session_state:
    st.session_state['appx'] = True
if 'outliers' not in st.session_state:
    st.session_state['outliers'] = False
if 'select_band' not in st.session_state:
    st.session_state['select_band'] = 'resc_band'

@st.cache_data  # atlernative to @st.cache for dataframes
def load_data(path, sep=",", dtype=None, names=None, header=None, index_col=None):
    ''' Reads the list of GRBs'''
    data =  pd.read_csv(path, 
                        sep=sep, 
                        engine='python', 
                        dtype=dtype,
                        names=names,
                        header=header, 
                        index_col=index_col
                        )
    return data

data = load_data(path='D:/naoj-grb/archive-webapp/grb-webapp/grblist.csv', 
                 sep=',', 
                 dtype=None,
                 names=None,
                 header=0, 
                 index_col="GRB"
                 )

# st.sidebar.markdown("### Select GRB")
# st.session_state['select_event'] = st.sidebar.selectbox("*Mandatory field", data.index)

lc = Lightcurve(path = 'mag-AB-extcorr-30-05-2023/'+st.session_state['select_event']+'_magAB_extcorr.txt',
                #data_space= 'lin',
                name = st.session_state['select_event'])

lc.set_data(path = 'mag-AB-extcorr-30-05-2023/'+st.session_state['select_event']+'_magAB_extcorr.txt', appx_bands=st.session_state['appx'])
#st.plotly_chart(lc.displayGRB())

st.sidebar.markdown('### Set data')
not_appx = st.sidebar.checkbox(label="Do not approximate bands")
if not_appx:
    st.session_state['appx'] = False
    #lc.set_data(path = 'mag-AB-extcorr-30-05-2023/'+st.session_state['select_event']+'_magAB_extcorr.txt', appx_bands=st.session_state['appx'])
    #st.plotly_chart(lc.displayGRB())

outliers = st.sidebar.checkbox(label="Remove outliers")
if outliers:
    st.session_state['outliers'] = True

lc.set_data(path = 'mag-AB-extcorr-30-05-2023/'+st.session_state['select_event']+'_magAB_extcorr.txt', 
            appx_bands=st.session_state['appx'],
            remove_outliers = st.session_state['outliers'])

resc_band = st.sidebar.checkbox("Choose band to rescale")
if resc_band:
    #lc.set_data(path = 'mag-AB-extcorr-30-05-2023/'+st.session_state['select_event']+'_magAB_extcorr.txt', appx_bands=st.session_state['appx'])
    filters = [*set(lc.band)] #light['band'])]
    st.session_state['select_band'] = st.sidebar.selectbox("Choose the band", filters)
else:
    st.session_state['select_band'] = 'mostnumerous'

lc.set_data(path = 'mag-AB-extcorr-30-05-2023/'+st.session_state['select_event']+'_magAB_extcorr.txt', appx_bands= st.session_state['appx'])

st.sidebar.markdown('### Actions')




analyse = st.sidebar.checkbox(label="Analyse")

if analyse:
    st.markdown("## Colour Evolution Analysis")
    st.markdown("")
    colorevolplot = st.empty()
    st.markdown("")
    colorevoltab = st.empty()
    st.markdown("## Rescaling")
    rescaleplot = st.empty()

    #try:
    fig, resc_slopes_df, nocolorevolutionlista0, colorevolutionlista0, filterforrescaling, light, rescale_df, nocolorevolutionlist, colorevolutionlist, fig2 = lc.colorevolGRB(chosenfilter=st.session_state['select_band'], print_status=False, return_rescaledf=True)
    #resc_slopes_df = resc_slopes_df[resc_slopes_df['prob_lin'] != "insufficient_data"]

    colorevolplot.pyplot(fig2)
    colorevoltab.table(resc_slopes_df)
    output_colorevol = [fig, resc_slopes_df, nocolorevolutionlista0, colorevolutionlista0, filterforrescaling, light]
    
    #except AssertionError:
    #    st.error("Rescaling band provided is not present in data! Check your band approximation settings.")

rescale = st.sidebar.checkbox(label="Rescale")

if rescale:
    #lc.set_data(path = 'mag-AB-extcorr-30-05-2023/'+st.session_state['select_event']+'_magAB_extcorr.txt', appx_bands=st.session_state['appx'])
    try:
        #fig, resc_slopes_df, nocolorevolutionlista0, colorevolutionlista0, filterforrescaling, light, rescale_df, nocolorevolutionlist, colorevolutionlist, fig2 = lc.colorevolGRB(chosenfilter=st.session_state['select_band'], print_status=False, return_rescaledf=True)
        output_colorevol = [fig, resc_slopes_df, nocolorevolutionlista0, colorevolutionlista0, filterforrescaling, light]
        figunresc, figresc, dfresc, *__ = lc.rescaleGRB(output_colorevolGRB=output_colorevol)
        rescaleplot.plotly_chart(figresc)

    except AssertionError:
        rescaleplot.error("Rescaling band provided is not present in data! Check your band approximation settings.")
    

# view = st.sidebar.button(label="View original")
# if view:
#     lc.set_data(path = 'mag-AB-extcorr-30-05-2023/'+st.session_state['select_event']+'_magAB_extcorr.txt', appx_bands=st.session_state['appx'])
#     st.plotly_chart(lc.displayGRB())