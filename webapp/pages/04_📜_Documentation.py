import streamlit as st

st.markdown("# About")

## Information
##############################################################################

st.write("""
    We present the largest optical photometry compilation of Gamma-Ray Bursts (GRBs) with redshifts (z). 
    We include 64813 observations of 535 events (including upper limits) from 28 February 1997 up to 18 August 2023. 
    We also present a user-friendly web tool *grbLC* which allows users the visualization of photometry, coordinates, 
    redshift, host galaxy extinction, and spectral indices for each event in our database. 
    Furthermore, we have added a Gamma Ray Coordinate Network (GCN) scraper that can be used to collect data 
    by gathering magnitudes from the GCNs. The web tool also includes a package for uniformly investigating colour evolution. 
    We compute the optical spectral indices for 138 GRBs for which we have at least 4 filters at the same epoch in our sample 
    and craft a procedure to distinguish between GRBs with and without colour evolution. By providing a uniform format and 
    repository for the optical catalogue, this web-based archive is the first step towards unifying several community efforts 
    to gather the photometric information for all GRBs with known redshifts. This catalogue will enable population studies 
    by providing light curves (LCs) with better coverage since we have gathered data from different ground-based locations. 
    Consequently, these LCs can be used to train future LC reconstructions for an extended inference of the redshift.
    The data gathering also allows us to fill some of the orbital gaps from Swift in crucial points of the LCs, 
    e.g., at the end of the plateau emission or where a jet break is identified.
""")

## Data
##############################################################################


#st.write("""### Data
# """)

## App Guide
##############################################################################


#st.write("### App demo")

#st.video('demo.webm', format="video/webm")
#st.video('https://youtu.be/5bFyADbBAAk', format='url')

## Usage policy
##############################################################################


st.write("""### Usage Policy
         
The data gathered in this catalogue was obtained by public sources and private communications. 
The GRBLC package and web-based repository are designed to be open-access and available to all members of the community.
All are welcome to use our software, though we ask that if the provided data or software is used in any publication,
the authors cite this paper as well as include the following statement in the acknowledgments: 
    
"Data used in our work is taken from the catalogue Dainotti et al. (2024), and the original data sources are cited within."
                  
""")


## Links
##############################################################################


'''
    [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/SLAC-Gamma-Rays/grbLC) 

'''
st.markdown("<br>",unsafe_allow_html=True)

