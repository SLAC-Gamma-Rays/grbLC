# standard libs
import os
import re
import sys

# third party libs
import numpy as np
import pandas as pd
import plotly.express as px

# custom modules
from .util import get_dir
from .io import read_data, _appx_bands
from .data.load import get_grb
from .photometry.constants import grbinfo
from .photometry.convert import _convertGRB, _host_kcorrectGRB
from .photometry.sed import _beta_marquardt
from .evolution.colorevol import _colorevolGRB
from .evolution.rescale import _rescaleGRB


class Lightcurve: # define the object Lightcurve
    _name_placeholder = "unknown grb" # assign the name for GRB if not provided


    def __init__(
        self,
        grb: str = None,
        path: str = None,
        data_space: str = 'lin',
        appx_bands: bool = False,
        remove_outliers: bool = False, 
        save: bool = True,
    ):
        """
        Function to set the `xdata` and `ydata`, and optionally `xerr` and `yerr` of the lightcurve.
        Reads in data from a file. The data must be in the correct format.
        See the :py:meth:`io.read_data` for more information.

        Parameters:
        -----------
        - grb: str: GRB name.
        - path: str: Path to the magnitude file. Enter 'raw' or 'converted' to directly access our catalogue.
        - data_space : str, {log, lin}: Whether to convert the data to logarithmic or linear space, by default 'lin'.
        - appx_bands: bool: If True, approximates certain bands, e.g. u' approximated to u, etc. By default False.
                            See the :py:meth:`io.convert_data` for more information.
        - remove_outliers: bool: If True, removes outliers identified in our analysis, by default False. 
                                See Dainotti et al. (2024): https://arxiv.org/pdf/2405.02263.
        - save: bool: If True, creates folder to save results.

        Returns:
        --------
        - None

        Raises:
        -------
        - AssertionError: If only limiting magnitudes are present.
        
        """

        # some default conditions for the name of GRBs and the path of the data file
        self.name = self._name_placeholder  
        if grb:
            self.name = grb 

        if path == 'raw' or 'converted':
            self.path = get_grb(grb, type=path)

        else:
            self.path = path 

        # reading the data from a file
        self.set_data(data_space, appx_bands, remove_outliers)

        # create directory to save results
        if save:
            self.main_dir = self.name+"/"
            if not os.path.exists(self.main_dir):
                os.mkdir(self.main_dir)


    def set_data(
        self, 
        data_space: str = 'lin',
        appx_bands: bool = True, 
        remove_outliers: bool = False
    ): 
        """
        Function to set the data.
        
        Parameters:
        -----------
        - data_space : str, {log, lin}: Whether to convert the data to logarithmic or linear space, by default 'lin'.
        - appx_bands: bool: If True, approximates certain bands, e.g. u' approximated to u, etc.
                            See the :py:meth:`io.convert_data` for more information.
        - remove_outliers: bool: If True, removes outliers identified in our analysis, by default False. 
                                See Dainotti et al. (2024): https://arxiv.org/pdf/2405.02263.
        
        Raises:
        -------
        - AssertionError: If only limiting magnitudes are present.

        Returns:
        --------
        - None
        
        """

        # reads the data, sorts by time, excludes negative time
        df = read_data(path = self.path, data_space = data_space) 
        
        # initialising a new column
        # asserting those data points only which does not have limiting nagnitude
        df = df[df['mag_err'] != 0] 
        assert len(df)!=0, "Only limiting magnitudes present."

        # initialising data to self
        self.xdata = df["time_sec"].to_numpy()  
            # passing the time in sec as a numpy array in the x column of the data

        self.ydata = df["mag"].to_numpy() 
            # passing the magnitude as a numpy array in the y column of the data

        self.yerr = df["mag_err"].to_numpy()  
            # passing the magnitude error as an numpy array y error column of the data

        self.band_original = df["band"].to_list() 
            # passing the original bands (befotre approximation of the bands) as a list
        
        if appx_bands:
            df.insert(4, "band_appx", "") 
            self.band = df["band_appx"] = _appx_bands(df["band"]) 
            # passing the reassigned bands (after the reapproximation of the bands) as a list
        
        else:
            self.band = self.band_original

        self.system = df["system"].to_list()  # passing the filter system as a list
        
        self.telescope = df["telescope"].to_list() 
            # passing the telescope name as a list
        
        self.extcorr = df["extcorr"].to_list()  
            # passing the galactic extinction correction detail (if it is corrected or not) as a list
        
        self.source = df["source"].to_list()  
            # passing the source from where the particular data point has been gathered as a list
        
        try:
            self.flag = df["flag"].to_list()
        
        except:
            self.flag = None

        if remove_outliers:
            df = df[df.flag == 'no']

        self.df = df  # passing the whole data as a data frame


    def displayGRB(
        self, 
        save_static: bool = False, 
        save_static_type: str = '.png', 
        save_interactive: bool = False, 
    ):
        """
        Function to create an interactive plot of magnitude lightcurve, excluding limiting magnitudes.

        Parameters:
        -----------
        - self:  Lightcurve object should be initialised to call displayGRB()
        - save_static: bool: If True, saves static plot of magnitude lightcurve.
        - save_static_type: str: By default, the static type is '.png'.
        - save_interactive: bool: If True, saves '.html' plot of magnitude lightcurve.

        Returns:
        --------
        - fig: plotly.Figure object: Interactive plot of magnitude lightcurve.

        """

        save_in_folder = None
        if save_static or save_interactive:
            save_in_folder = self.main_dir + 'plots/'
            if not os.path.exists(save_in_folder):
                os.mkdir(save_in_folder)

        fig = px.scatter(data_frame=read_data(df = self.df, data_space='log'),
                    x='time_sec',
                    y='mag',
                    error_y='mag_err',
                    color='band',
                    color_discrete_sequence=px.colors.qualitative.Set1,
                    hover_data=['telescope', 'source'],
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


    def convertGRB(
        self,
        save: bool = True,
        debug: bool = False
    ):
        """
        Function to convert magnitudes to the AB system and correct for galactic extinction.
        This is an optional step. If the files are already converted, can skip this.

        Parameters:
        -----------
        - self: Lightcurve object should be initialised to call correctGRB().
        - save: bool: If True, saves converted file.
        - debug: bool: More information saved for debugging the conversion. By default, False.

        Returns:
        --------
        - Converted magnitude tab-separated '.txt'.

        Raises:
        -------
        - KeyError: If the telescope and filter is not found.
        - ImportError: If the code can't find grb table at the given path.

        """

        save_in_folder = None
        if save:
            save_in_folder = self.main_dir + 'converted/'
            if not os.path.exists(save_in_folder):
                os.mkdir(save_in_folder)
        else:
            save_in_folder = None

        self.df = _convertGRB(
                            grb = self.name,
                            ra = grbinfo.loc[self.name, 'ra'],
                            dec = grbinfo.loc[self.name, 'dec'],
                            mag_table = self.df,
                            save_in_folder = save_in_folder,
                            debug = debug
                            )
        return self.df
    

    def betaGRB(
        self,
        print_status: bool = False,
        save: bool = False,
    ):
        """
        Function to fit the Spectral Energy Distribution. See Equation 3 of Dainotti et al. (2024): https://arxiv.org/pdf/2405.02263
        For the spectral shape, a power-law is assumed.
        The host extinction galaxy is given by the maps of Pei (1992).
        In the comparison between the 3 host models (SMC,LMC,MW) for each time epoch, the most probable fitting is the chosen one.
        Using Marquardt method, we obtain the best fit for the following parameters: 
            - beta (optical spectral index), 
            - A_V (host extinction in V-band),
            - intercept
        The cases with uncertainty on the beta greater than the beta values themselves are excluded.
        
        Parameters:
        -----------
        - self:  Lightcurve object should be initialised to call correctGRB().
        - print_status: If True, prints intermediate steps. By default, False.
        - save: bool: If True, saves results.
        
        Returns:
        --------
        - dfexp: pd.DataFrame: Contains fit result.

        """

        save_in_folder = None
        if save:
            save_in_folder = self.main_dir + 'beta/'
            if not os.path.exists(save_in_folder):
                os.mkdir(save_in_folder)

        self.sed_results = _beta_marquardt(
                                        grb = self.name, 
                                        path = self.path,
                                        z = grbinfo.loc[self.name, 'z'],
                                        print_status = print_status,
                                        save_in_folder = save_in_folder,
                                        )
        
        return self.sed_results
    

    def host_kcorrectGRB(
        self,
        save: bool = False,
        debug: bool = False
    ):
        """
        Function to perform host extinction correction and k-correction for redshift effects.

        Parameters:
        -----------
        - self: Lightcurve object should be initialised and betaGRB() must be performed to be able to call host_kcorrectGRB().
        - save: bool: If True, saves files.
        - debug: bool: If True, more information saved for debugging the conversion, by default, False.

        Returns:
        --------
        - converted: pandas.DataFrame: Corrected magnitude file.

        Raises:
        -------
        - KeyError: If the telescope and filter is not found.
        - ImportError: If the code can't find grb table at the given path.

        """
       
        save_in_folder = None
        if save:
            save_in_folder = self.main_dir + 'host_kcorrect/'
            if not os.path.exists(save_in_folder):
                os.mkdir(save_in_folder)
       
        return _host_kcorrectGRB(
                                grb = self.name,
                                mag_table = self.df,
                                sed_results = self.sed_results,
                                save_in_folder = save_in_folder,
                                debug = debug,
                                )
        

    def colorevolGRB(
        self, 
        chosenfilter: str = 'mostnumerous', 
        print_status: bool = False,
        save: bool = False, 
        debug: bool = False
    ):

        """
        Function performs the colour evolution analysis (see Section 3.3 of Dainotti et al. (2024): https://arxiv.org/abs/2405.02263).
        
        Parameters:
        -----------
        - self:  Lightcurve object should be initialised to call colorevolGRB().
        - chosenfilter: str: The filter with respect to which rescaling is evaluated, by default the 'mostnumerous' one.
        - print_status: bool: If True, prints intermediate steps.
        - save: bool: If True, saves results.
        - debug: bool: If True, generates a report of the analysis called "report_colorevolution.txt".

        Returns:
        --------
        - fig_avar: plotly.Figure object: Interactive plot of magnitudes and rescaling factors versus log10(time)
                                            in the case of "variable a" fitting.
        - fig_a0: plotly.Figure object: Interactive plot of magnitudes and rescaling factors versus log10(time)
                                            in the case of "a0" fitting.
        - filterforrescaling: str: The filter used for rescaling (by default, the most numerous, otherwise the customized one).
        - nocolorevolutionlist: list: Filters that show no colour evolution according to the "variable a" fitting.
        - colorevolutionlist: list: Filters that show no colour evolution according to the "variable a" fitting.
        - nocolorevolutionlista0: list: Filters that show no colour evolution according to the "a=0" fitting.
        - colorevolutionlista0: list: Filters that show no colour evolution according to the "a=0" fitting.
        - light: pd.DataFrame: Contains magnitude information.
        - resc_slopes_df: pd.DataFrame: Information on the rescaling factors fitting both in the cases of "variable a" and "a=0".
        - rescale_df: pd.DataFrame: : Contains information of the rescaling factors.

        Raises:
        -------
        - None.

        """
        
        save_in_folder = None
        if save:
            save_in_folder = self.main_dir + 'colorevol/'
            if not os.path.exists(save_in_folder):
                os.mkdir(save_in_folder)

        self.output_colorevol = _colorevolGRB(
                                            self.name, 
                                            self.df, 
                                            chosenfilter, 
                                            print_status,
                                            save_in_folder, 
                                            debug
                                            )
        return self.output_colorevol


    def rescaleGRB(
        self, 
        remove_duplicate = False,
        save: bool = False
    ):
        """
        Function to rescale the GRB after colour evolution analysis has been performed.
        Rescaling of the filters is applied in the cases only where there is no colour evolution.

        Parameters:
        -----------
        - self: Lightcurve object should be initialised and colorevolGRB() must be performed to be able to call rescaleGRB().
        - save_in_folder: Path to store the rescaled magnitude file.
        - remove_duplicate: Remove multiple data points at coincident time, by default False.

        Returns:
        --------
        - figunresc: plotly.Figure object: Interactive plot of magnitude lightcurve before rescaling.
        - figresc: plotly.Figure object: Interactive plot of magnitude lightcurve after rescaling.
        - resc_mag_df: pd.DataFrame: DataFrame containing the rescaled magnitudes.

        Raises:
        -------
        - ValueError: If no filters to rescale, i.e. all filters show colour evolution

        """

        save_in_folder = None
        if save:
            save_in_folder = self.main_dir + 'colorevol/'
            if not os.path.exists(save_in_folder):
                os.mkdir(save_in_folder)

        return _rescaleGRB(
                        grb = self.name, 
                        output_colorevolGRB = self.output_colorevol, 
                        remove_duplicate = remove_duplicate,
                        save_in_folder = save_in_folder
                        )


major, *__ = sys.version_info # this command checks the Python version installed locally
readfile_kwargs = {"encoding": "utf-8"} if major >= 3 else {} # this option specifies the enconding of imported files in Python
                                                              # the encoding is utf-8 for Python versions superior to 3.
                                                              # otherwise it is left free to the code

def _readfile(path): 
    """
    Function for basic importation of text files.

    """

    with open(path, **readfile_kwargs) as fp:
        contents = fp.read()
    return contents


# re.compile(): compile the regular expression specified by parenthesis to make it match
version_regex = re.compile('__version__ = "(.*?)"') #
contents = _readfile(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "__init__.py"
    )
) # this command reads __init__.py that gives the basic functions for the package, namely get_dir, set_dir
__version__ = version_regex.findall(contents)[0]

__directory__ = get_dir()