# standard libs
import os
import re
import sys
from functools import reduce

# third party libs
import numpy as np
import pandas as pd
import plotly.express as px

pd.set_option('display.max_rows', None)

# custom modules
from grblc.util import get_dir
from . import io
from .rescale import _rescaleGRB
from .colorevol import _colorevolGRB


class Lightcurve: # define the object Lightcurve
    _name_placeholder = "unknown grb" # assign the name for GRB if not provided


    def __init__(
        self,
        path: str = None,
        appx_bands: str = True, # if it is True it enables the approximation of bands, e.g. u' approximated to u,.....
        name: str = None,
    ):
        """The main module for fitting lightcurves.

        Parameters
        ----------
        path : str, optional
            Name of file containing light curve data, by default None
        xdata : array_like, optional
            X values, length (n,), by default None
        ydata : array_like, optional
            Y values, by default None
        xerr : array_like, optional
            X error, by default None
        yerr : array_like, optional
            Y error, by default None
        data_space : str, {log, lin}, optional
            Whether the data inputted is in log or linear space, by default 'log'
        name : str, optional
            Name of the GRB, by default :py:class:`Model` name, or ``unknown grb`` if not
            provided.
        """

        # some default conditions for the name of GRBs and the path of the data file
        if name:
            self.name = name  # asserting the name of the GRB
        else:
            self.name = self._name_placeholder  # asserting the name of the GRB as 'Unknown GRB' if the name is not provided

        if isinstance(path, str):
            self.path = path  # asserting the path of the data file
            self.set_data(path, appx_bands=appx_bands) #, data_space='lin') # reading the data from a file
            print(self.df.head())


    def set_data(self, path: str, appx_bands=True, remove_outliers=False): #, data_space='lin'):
        """
            Reads in data from a file. The data must be in the correct format.
            See the :py:meth:`io.read_data` for more information.

            Set the `xdata` and `ydata`, and optionally `xerr` and `yerr` of the lightcurve.

        .. warning::
            Data stored in :py:class:`lightcurve` objects are always in logarithmic
            space; the parameter ``data_space`` is only used to convert data to log space
            if it is not already in such. If your data is in linear space [i.e., your
            time data is sec, and not log(sec)], then you should set ``data_space``
            to ``lin``.

        Parameters
        ----------
        xdata : array_like
            X data
        ydata : array_like
            Y data
        xerr : array_like, optional
            X error, by default None
        yerr : array_like, optional
            Y error, by default None
        data_space : str, {log, lin}, optional
            Whether the data inputted is in logarithmic or linear space, by default 'log'.
        """

        df = io.read_data(path) # reads the data, sorts by time, excludes negative time
        df.insert(4, "band_appx", "") # initialising new column

        df = df[df['mag_err'] != 0] # asserting those data points only which does not have limiting nagnitude
        assert len(df)!=0, "Only limiting magnitudes present."

        self.xdata = df["time_sec"].to_numpy()  # passing the time in sec as a numpy array in the x column of the data
        self.ydata = df["mag"].to_numpy() # passing the magnitude as a numpy array in the y column of the data
        self.yerr = df["mag_err"].to_numpy()  # passing the magnitude error as an numpy array y error column of the data
        self.band_original = df["band"].to_list() # passing the original bands (befotre approximation of the bands) as a list
        self.band = df["band_appx"] = io.convert_data(df["band"]) # passing the reassigned bands (after the reapproximation of the bands) as a list
        self.system = df["system"].to_list()  # passing the filter system as a list
        self.telescope = df["telescope"].to_list()  # passing the telescope name as a list
        self.extcorr = df["extcorr"].to_list()  # passing the galactic extinction correction detail (if it is corrected or not) as a list
        self.source = df["source"].to_list()  # passing the source from where the particular data point has been gathered as a list
        try:
            self.flag = df["flag"].to_list()
        except:
            self.flag = None
        if remove_outliers:
            df = df[df.flag == 'no']
        self.df = df  # passing the whole data as a data frame


    def displayGRB(self, save_static=False, save_static_type='.png', save_interactive=False, save_in_folder='plots/'):
        # This function plots the magnitudes, excluding the limiting magnitudes

        '''
        For an interactive plot
        '''

        #print(self.xdata)

        fig = px.scatter(data_frame=self.df,
                    x=np.log10(self.xdata),
                    y=self.ydata,
                    error_y=self.yerr,
                    color=self.band,
                    color_discrete_sequence=px.colors.qualitative.Set1,
                    hover_data=['telescope'],
                )

        font_dict=dict(family='arial',
                    size=18,
                    color='black'
                    )
        title_dict=dict(family='arial',
                    size=20,
                    color='black'
                    )

        fig['layout']['yaxis']['autorange'] = 'reversed'
        fig.update_yaxes(title_text="<b>Magnitude<b>",
                        title_font_color='black',
                        title_font_size=20,
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
                        title_font_size=20,
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
                        title_font_size=24,
                        font=font_dict,
                        legend = dict(font = font_dict),
                        legend_title = dict(text= "<b>Bands<b>", font=title_dict),
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


    def colorevolGRB(self, print_status=True, return_rescaledf=False, save_plot=False, chosenfilter='mostnumerous', save_in_folder='', reportfill=False):
        self.output_colorevol = _colorevolGRB(
                                            grb=self.name, 
                                            df=self.df, 
                                            print_status=print_status, 
                                            return_rescaledf=return_rescaledf, 
                                            save_plot=save_plot, 
                                            chosenfilter=chosenfilter, 
                                            save_in_folder=save_in_folder, 
                                            reportfill=reportfill
                                            )
        return self.output_colorevol


    def rescaleGRB(self, save_in_folder='rescale/', remove_dups=True):
        return _rescaleGRB(
                        grb = self.name, 
                        output_colorevolGRB = self.output_colorevol, 
                        save_in_folder = save_in_folder, 
                        duplicateremove = remove_dups
                        )


major, *__ = sys.version_info # this command checks the Python version installed locally
readfile_kwargs = {"encoding": "utf-8"} if major >= 3 else {} # this option specifies the enconding of imported files in Python
                                                              # the encoding is utf-8 for Python versions superior to 3.
                                                              # otherwise it is left free to the code

def _readfile(path): # function for basic importation of text files, using the options defined in lines 1043,1044
    with open(path, **readfile_kwargs) as fp:
        contents = fp.read()
    return contents


# re.compile(): compile the regular expression specified by parenthesis to make it match
version_regex = re.compile('__version__ = "(.*?)"') #
contents = _readfile(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "__init__.py"
    )
) # this command reads __init__.py that gives the basic functions for the package, namely get_dir, set_dir
__version__ = version_regex.findall(contents)[0]

__directory__ = get_dir()