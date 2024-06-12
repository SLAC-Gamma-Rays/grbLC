# standard libs
import os

# third party libs
import numpy as np
import pandas as pd
import plotly.express as px

pd.set_option('display.max_rows', None)

def _rescaleGRB(grb, output_colorevolGRB, save_rescaled_in='', duplicateremove=True): 
    
    # This function performs the rescaling of the GRB after the colour evolution analysis.
    # This function takes as input the output previously obtained in the "colevol" function applied on the same "grb"
    # The output of the function is the "rescmagdataframe", namely the input dataframe where the rescaling of the filters
    # is applied in the cases where there is no colour evolution.
    # All the other cases (filters with colour evolution or undetermined behaviour) are left in the "rescmagdataframe" without any rescaling.

    # the global option is needed when these variables inputed in the current function are output of another function recalled, namely, colorevolGRB
    global filterforrescaling, light, overlap #, nocolorevolutionlist

    def overlap(mag1lower,mag1upper,mag2lower,mag2upper): # this is the condition to state if two magnitude ranges overlap
        if mag1upper <mag2lower or mag1lower > mag2upper:
            return 0 # in the case of no overlap, zero is returned
        else:
            return max(mag1upper, mag2upper) # in the case of overlap, a value>0 is returned

    def duplicate_remover(df):
        #freq_df = df['band_init'].value_counts()
        freq_df = pd.DataFrame(df['band_approx'].value_counts())
        freq_df.rename(columns = {'band_approx':'freq'}, inplace = True)
        time=df['time_sec']
        #df['logtime']=logtime

        for i,j in zip(df.index, df['band_approx']):
            df.loc[i, 'freq'] = freq_df.loc[j, 'freq']

        df = df.sort_values('freq', ascending=False).drop_duplicates(subset='time_sec', keep="first")


        df = df.sort_values('time_sec', ascending=True)

        #display(df)

        return df

    # here the code uses the colorevolGRB function defined above; the outputs of the function colorevolGRB will be used as input in the current function

    #output_colorevolGRB = self.colorevolGRB(print_status=False, return_rescaledf=False, save_plot=False, chosenfilter=chosenfilter, save_in_folder=save_rescaled_in)
    input = output_colorevolGRB
    nocolorevolutionlist = input[2] # 3rd output of colorevolGRB function, this is the list of filters whose resc.fact. slopes are compatible with zero in 3sigma or are < 0.10
    colorevolutionlist =input[3] # 4th output of colorevolGRB function, this is the list of filters whose resc.fact. slopes are incompatible with zero in 3sigma and are>0.10
    filterforrescaling = input[4] # 5th output of colorevolGRB function, this is the filter chosen for rescaling
    light = input[5] # 6th output of colorevolGRB function, is the original dataframe (since the filter chosen for rescaling is present here and it is needed)
    
    # Before rescaling the magnitudes, the following instructions plot the magnitudes in the unrescaled case
    
    color=light['band_approx'].tolist()
    filterslist=[[x,color.count(x)] for x in set(color)]

    freq=[]
    for k in color:
        for j in filterslist:
            if j[0]==k:
                freq.append(str(k)+' ('+str(j[1])+')')

    figunresc = px.scatter(
            x=np.log10(light['time_sec'].values), # the time is set to log10(time) only in the plot frame
            y=light['mag'].values,
            error_y=light['mag_err'].values,
            #color=light['band'].values,
            color=freq,
            color_discrete_sequence=px.colors.qualitative.Set1,
            labels={"color": "      <b>Band<b>"}
            #hover_data=light['source'].values
            )

    figunresc.update_layout(legend=dict(font=dict(size=26)))

    font_dict=dict(family='arial', # font of the plot's layout
                size=26,
                color='black'
                )

    figunresc.update_layout(legend_font_size=26, legend_font_color='black')
    figunresc.for_each_trace(lambda t: t.update(name = '<b>' + t.name +'</b>'))

    figunresc['layout']['yaxis']['autorange'] = 'reversed' # here the y axis is reversed, so that higher magnitudes are down and lower are up
    figunresc.update_yaxes(title_text="<b>Magnitude<b>", # updating plot options in the y-axes
                    title_font_color='black',
                    title_font_size=28,
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
                    tickprefix="<b>",
                    ticksuffix ="</b><br>"
                    )

    figunresc.update_xaxes(title_text="${\\bf\\huge log_{10} t\, (s)}$", # updating plot options in the x-axes
                    title_font_color='black',
                    title_font_size=35,
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
                    tickprefix="<b>",
                    ticksuffix ="</b><br>"
                    )

    #figunresc.update_layout(title="GRB " + self.name, # updating the layout of the plot
    #                title_font_size=25,
    #                font=font_dict,
    
    figunresc.update_layout(plot_bgcolor='white',
                    width=960,
                    height=540,
                    margin=dict(l=40,r=40,t=50,b=40)
                    )

    figunresc.add_annotation(
    text = ('<b>'+"GRB "+ grb +'</b>')
    , showarrow=False
    , x = 0.08
    , y = 0.15
    , xref='paper'
    , yref='paper' 
    , xanchor='left'
    , yanchor='bottom'
    , xshift=-1
    , yshift=-5
    , font=dict(size=35, color="black")
    , align="left"
    ,)

    figunresc.update_traces(marker={'size': 9})

    figunresc.show()

    # Two additive columns must be inserted in the light dataframe

    light["mag_rescaled_to_"+filterforrescaling] = "" # the column with rescaled magnitudes
    light["mag_rescaled_err"] = "" # the column with magnitude errors, propagating the uncertainties on the magnitude itself and on the rescaling factor

    print(nocolorevolutionlist)

    for rr in light.index:
        # In these cases, the rescaled magnitude is the same of the original magnitude:
        # 1) The datapoint has the filter chosen for rescaling (obviously, can't be rescaled to itself)
        # 2) If the rescaling factor is not estimated (time difference > 2.5 percent)
        # 3) If the magnitudes of the filter chosen for rescaling and the filter to be rescaled overlap (mag overlap >0)
        # 4) If the filter belongs to the list of filters that have color evolution
        if light.loc[rr, "mag_chosenfilter"] == "-" or light.loc[rr, "mag_chosenfilter_err"] =="-":
            light.loc[rr, "mag_rescaled_to_"+filterforrescaling] = light.loc[rr, "mag"]
            light.loc[rr, "mag_rescaled_err"] = light.loc[rr, "mag_err"]
        else:
            magoverlapcheck=overlap(light.loc[rr, "mag"]-light.loc[rr, "mag_err"], # the overlap between the filter chosen for rescaling and the filter to be rescaled; zero if no overlap, number>0 if overlap
                                        light.loc[rr, "mag"]+light.loc[rr, "mag_err"],
                                        light.loc[rr, "mag_chosenfilter"]-light.loc[rr, "mag_chosenfilter_err"],
                                        light.loc[rr, "mag_chosenfilter"]+light.loc[rr, "mag_chosenfilter_err"])

        if (light.loc[rr, "band_approx"]==filterforrescaling) or (light.loc[rr, "resc_fact"] == "-") or (magoverlapcheck != 0) or (light.loc[rr, "band_approx"] in colorevolutionlist):
            light.loc[rr, "mag_rescaled_to_"+filterforrescaling] = light.loc[rr, "mag"]
            light.loc[rr, "mag_rescaled_err"] = light.loc[rr, "mag_err"]

        # In the following cases, magnitudes are rescaled
        # If the rescaling factor is estimated AND the time difference is smaller than 2.5 percent AND the overlap is zero AND the band is in the list of filters that have no color evolution
        elif (light.loc[rr, "resc_fact"] != "-") and (light.loc[rr, "time_difference_percentage"]<=0.025) and (magoverlapcheck==0) and (light.loc[rr, "band_approx"] in nocolorevolutionlist):
            light.loc[rr, "mag_rescaled_to_"+filterforrescaling] = light.loc[rr, "mag"]+light.loc[rr, "resc_fact"]
            light.loc[rr, "mag_rescaled_err"] = light.loc[rr, "resc_fact_err"] # the error on the rescaled magnitude is already estimated with the rescaling factor error

        # In all the other cases, the magnitudes can't be rescaled and the default options follow
        else:
            light.loc[rr, "mag_rescaled_to_"+filterforrescaling] = light.loc[rr, "mag"]
            light.loc[rr, "mag_rescaled_err"] = light.loc[rr, "mag_err"]

    if duplicateremove:

        light=duplicate_remover(light)

    color=light['band_approx'].tolist()
    filterslist=[[x,color.count(x)] for x in set(color)]

    freq=[]
    for k in color:
        for j in filterslist:
            if j[0]==k:
                freq.append(str(k)+' ('+str(j[1])+')')

    # The plot of the rescaled dataframe
    figresc = px.scatter(
            x=np.log10(light["time_sec"].values), # the time is set to log10(time) only in the plot frame
            y=light["mag_rescaled_to_"+filterforrescaling].values,
            error_y=light["mag_rescaled_err"].values,
            #color=light["band_approx"],
            color=freq,
            color_discrete_sequence=px.colors.qualitative.Set1,
            labels={"color": "      <b>Band<b>"}
            )

    font_dict=dict(family='arial',
                size=26,
                color='black'
                )

    #figresc.update_layout(legend=dict(font=dict(size=26)))

    figresc.update_layout(legend_font_size=26, legend_font_color='black')
    figresc.for_each_trace(lambda t: t.update(name = '<b>' + t.name +'</b>'))

    figresc['layout']['yaxis']['autorange'] = 'reversed'
    figresc.update_yaxes(title_text="<b>Magnitude<b>",
                    title_font_color='black',
                    title_font_size=28,
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
                    tickprefix="<b>",
                    ticksuffix ="</b><br>"
                    )

    figresc.update_xaxes(title_text="${\\bf\\huge log_{10} t\, (s)}$",
                    title_font_color='black',
                    title_font_size=35,
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
                    tickprefix="<b>",
                    ticksuffix ="</b><br>"
                    )

    #figresc.update_layout(title="GRB " + self.name + " rescaled",
    #                title_font_size=25,
    #                font=font_dict,
    
    figresc.update_layout(plot_bgcolor='white',
                    width=960,
                    height=540,
                    margin=dict(l=40,r=40,t=50,b=40)
                    )
    
    figresc.add_annotation(
    text = ('<b>'+"GRB " + grb + "<br>    rescaled"+'</b>')
    , showarrow=False
    , x = 0.05
    , y = 0.12
    , xref='paper'
    , yref='paper' 
    , xanchor='left'
    , yanchor='bottom'
    , xshift=-1
    , yshift=-5
    , font=dict(size=35, color="black")
    , align="left"
    ,)

    figresc.update_traces(marker={'size': 9})

    figresc.show()

    # The definition of the rescaled dataframe
    # the list of values must be merged again in a new dataframe before exporting

    rescmagdataframe = pd.DataFrame()
    rescmagdataframe['time_sec'] = light['time_sec']
    rescmagdataframe['mag_rescaled_to_'+str(filterforrescaling)] = light['mag_rescaled_to_'+filterforrescaling]
    rescmagdataframe['mag_err'] = light['mag_rescaled_err']
    rescmagdataframe['band_original'] = light['band'] # the original band (not approximated is re-established) to allow a more precise photometric estimation if mags are converted into fluxes
    rescmagdataframe['system'] = light['system']
    rescmagdataframe['telescope'] = light['telescope']
    rescmagdataframe['extcorr'] = light['extcorr']
    rescmagdataframe['source'] = light['source']
    rescmagdataframe['flag'] = light['flag']

    # The option for exporting the rescaled magnitudes as a dataframe
    if not os.path.exists(save_rescaled_in):
        os.makedirs(save_rescaled_in)
    rescmagdataframe.to_csv(os.path.join(save_rescaled_in+'/' + str(grb).split("/")[-1]+  '_rescaled_to_'+str(filterforrescaling)+'.txt'),sep=' ',index=False)

    return rescmagdataframe
