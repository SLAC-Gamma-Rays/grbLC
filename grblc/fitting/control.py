import csv
import glob2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import clear_output
import numpy as np
import os
from fitting import *

def run_fit(filepaths):
    num_points = LC_summary(filepaths)
    mt5 = sum(np.asarray(list(num_points.values()))>=5)

    print(mt5,'>= 5 points. Plotting now.')
    dList = []
    for filepath in filepaths:
        
        grb = os.path.split(filepath)[-1].rstrip("_flux_accepted.txt")
        
        if num_points[grb] > 5:
            plot_data(filepath)
        
            auto_guess = input('Do you want to choose the inputs? (y/n)')
            if auto_guess == 'y':
                T = float(input('T : '))
                F = float(input('F : '))
                alpha = float(input('alpha : '))
                t = float(input('t : '))
                fit_vals = [T, F, alpha, t]
                # save fit_vals to csv

                try:
                    call_fits(filepath,guess = fit_vals)
                
                    with open(grb+'_fit_vals.csv','w') as file:
                        writer = csv.DictWriter(file,['T', 'F', 'alpha', 't']) 
                        writer.writerow({'T': fit_vals[0]})
                        writer.writerow({'F': fit_vals[1]})
                        writer.writerow({'alpha': fit_vals[2]})
                        writer.writerow({'t': fit_vals[3]}) 
                    
                except Exception as e:
                    raise e
                
            elif auto_guess == 'n':
                try:
                    call_fits(filepath)
                except Exception as e:
                    raise e
            
            

def call_fits(filepath, guess = False):
    print(guess)
    failct = 0
    grb = os.path.split(filepath)[-1].rstrip("_flux_cleaned.txt")
    df = pd.read_csv(filepath,
                               delimiter=r'\t+|\s+',
                               engine='python',
                               header=0)
    
    if sum(df["time_sec"]<=0) == 0:
        flux = df["flux"]
        flux_err = df["flux_err"]
        time = df["time_sec"]
    
        xdata = list(np.log10(time))
        ydata = list(np.log10(flux))
        known_yerr = flux_err / (flux * np.log(10))
        
        if guess is not False:
            
            try:
                p, pcov = fit_w07(xdata, ydata, p0 = guess, logTerr=None, logFerr=known_yerr, return_guess=False, maxfev=10000)
                plot_w07_fit(xdata, ydata, p, logTerr=None, logFerr=known_yerr, guess=guess)
                #fitting.plot_w07_toy_fit(xdata, ydata, pfit=p, ptrue=guess, logFerr=known_yerr)
                plot_chisq(xdata, ydata, p, pcov)
            except RuntimeError:
                failct +=1
                print('Fitting does not work :(')
                
            
            
        else:
            
            try:
                p,pcov,guess = fit_w07(xdata, ydata, logTerr=None, logFerr=known_yerr, return_guess=True, maxfev=10000)
                plot_w07_fit(xdata, ydata, p, logTerr=None, logFerr=known_yerr, guess=guess)
                #plot_w07_toy_fit(xdata, ydata, pfit=p, ptrue=guess, logFerr=known_yerr)
                plot_chisq(xdata, ydata, p, pcov)
            except RuntimeError:
                failct +=1
                print('Fitting does not work :(')
                
            
def plot_data(filepath):
    
    grb = os.path.split(filepath)[-1].rstrip("_flux_cleaned.txt")
    
    df = pd.read_csv(filepath,
                               delimiter=r'\t+|\s+',
                               engine='python',
                               header=0)
    
    fig = px.scatter(df,
						x = np.log10(df["time_sec"]),
						y = np.log10(df["flux"]),
						error_y = df["flux_err"] / (df["flux"] * np.log(10)),
						color = "band",
						width = 800,
						height = 500
						)
    
    fig.update_layout(
			title=grb,
			xaxis_title=r"$\text{logT (sec)}$",
			yaxis_title=r"$\text{logF (erg cm}^{-2}\text{ s}^{-1})$",
			legend_title="Band",
			yaxis_zeroline=True,
			xaxis_zeroline=True)
    
    fig.show()
    
    
def LC_summary(filepaths):
    lc_data = {}
    fig, ax = None, None
    for filepath in filepaths:
        try:
            grb = os.path.split(filepath)[-1].rstrip("_flux_accepted.txt")
        except:
            grb = os.path.split(filepath)[-1].rstrip("_flux.txt")
        df = pd.read_csv(filepath,
                            delimiter=r"\t+|\s+",
                            engine='python',
                            header=0)
        num_rows = len(df.index)
        bands = ",".join(list(df["band"])) # because lists aren't hashable >:(
        lc_data[grb] = [num_rows, bands]
  
    return {grb: l for grb, (l, _) in lc_data.items()}