#%%
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,LogNorm, Normalize, ListedColormap
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.dates as mdates
from matplotlib.cm import ScalarMappable
import matplotlib
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import HydroErr as he

from util_functions import *


# Import packages
#%%
# flaoting variables
img_extent = (-180, 180, -90, 90)

plt_extent = [-180,180,-90,90]

plt_extent_nh = [-180,180,45,90]

plt_extent_sh = [-180,180,-90,-45]

integer_list = [0, 1, 2, 3]

colors = ['orange', 'k', 'crimson', 'cyan','b', 'lime','m','g','r']

#%%
# Function to format y-axis labels
def scientific_notation_formatter(x, pos):
    """
    Format y-axis labels using scientific notation with a base of 1e6.
    """
    return f'{x / 1e6:.0f}M'
#---------------------------------------------------------------
# Function to create a formatter function for scientific notation
def make_scientific_formatter(decimals):
    """
    Create a formatter function that formats tick labels in scientific notation with a base of 1e6.
    
    Parameters:
    decimals (int): The number of decimal places to use when rounding the tick labels.
    
    Returns:
    function: A formatter function for use with Matplotlib's FuncFormatter.
    """
    def scientific_notation_formatter(x, pos):
        format_string = f'{{:.{decimals}f}}M'
        return format_string.format(x / 1e6)
    return scientific_notation_formatter
#--------------------------------------------------------------------------

def custom_formatter(x, pos):
    if float(x).is_integer():
        return int(x)  
    else:
        return f'{x:.0f}'
# Figure 5
def plot_daily_percentage_mismatch(hit_miss_df):
    """
    Plots the daily percentage mismatch for different categories (Combined, Water, Snow Free, Snow Cover, Ice).

    Parameters:
    hit_miss_df (DataFrame): The DataFrame containing the data to be plotted.
    """
    fig, axes = plt.subplots(5, 1, figsize=(15, 10), dpi=1000, sharex=True,
                             gridspec_kw={'height_ratios': [1] * 5})  # Equal height for all subplots
    plt.subplots_adjust(hspace=0.27, bottom=0.09, left=0.1)

    lw = 1  # Line width for general plots
    colors = ['orange', 'g', 'm', 'b']

    # Plot settings for each subplot
    categories = [
        {'column': ['ML-ECC-GMASI-miss', 'ML-E-GMASI-miss', 'ML-EC-GMASI-miss', 'Climatology-GMASI-miss'],
         'title': 'Combined', 'ylim': (0, 3.5), 'yticks': [0, 1.5, 3.5]},
        {'column': ['ML-E-wter-miss', 'ML-ECC-wter-miss', 'ML-EC-wter-miss', 'Climatology-wter-miss'],
         'title': 'Water', 'ylim': (0, 3), 'yticks': [0, 1.5, 3]},
        {'column': ['ML-E-snflnd-miss', 'ML-ECC-snflnd-miss', 'ML-EC-snflnd-miss', 'Climatology-snflnd-miss'],
         'title': 'Snow Free', 'ylim': (0, 9), 'yticks': [0, 5.0, 10.0]},
        {'column': ['ML-E-snclnd-miss', 'ML-ECC-snclnd-miss', 'ML-EC-snclnd-miss', 'Climatology-snclnd-miss'],
         'title': 'Snow Cover', 'ylim': (0, 10), 'yticks': [0, 5.0, 10.0]},
        {'column': ['ML-E-ice-miss', 'ML-ECC-ice-miss', 'ML-EC-ice-miss', 'Climatology-ice-miss'],
         'title': 'Ice', 'ylim': (0, 13), 'yticks': [0, 7.0, 14.0]}
    ]

    for i, ax in enumerate(axes):
        # Plot each line in the category
        for idx, (column, style, label) in enumerate(zip(categories[i]['column'], 
                                                         ['-', ':', '-.', '-'], 
                                                         ['ML-E', 'ML-ECC', 'ML-EC', 'CLIM'])):
            ax.plot(hit_miss_df.index, hit_miss_df[column], ls=style, c=colors[idx], lw=lw if label != 'ML-ECC' else 3, label=label)

        ax.set_title(categories[i]['title'], fontsize=18, loc='left')
        ax.set_ylim(categories[i]['ylim'])
        ax.set_yticks(categories[i]['yticks'])
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', labelsize=15, top=True, right=True)
        ax.grid(which='major', linestyle='--', linewidth=0.5, color='grey')

    # Add shared labels
    fig.text(0.06, 0.5, 'Daily Percentage Mismatch [%]', ha='center', va='center', rotation='vertical', fontsize=18)
    fig.text(0.5, 0.035, 'Year', ha='center', va='center', rotation='horizontal', fontsize=18)

    # Add legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03),
               ncol=5, fontsize=18, frameon=False)

#---------------------------------------------------------------
# Figure 8
def plot_time_series_4x1(area_extent_df,plt_term, plt_type,decforma):
    # Define colors for each model or dataset
    colors = ['orange', 'g', 'm', 'b','k']

    if plt_type == 'px_cnt':
        models = ['ML-E', 'ML-EC', 'ML-ECC', 'CLIM', 'GMASI']
    else:
        models = ['ML-E', 'ML-EC', 'ML-ECC', 'CLIM']

    # Convert index to datetime if it isn't already
    area_extent_df.index = pd.to_datetime(area_extent_df.index)

    # Creating subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True, dpi=1000,
                             gridspec_kw={'height_ratios': [1]*4})
    plt.subplots_adjust(hspace=0.4, bottom=0.1, left=0.12)

    # Plotting
    variables = ['wtr', 'snfr', 'snc', 'ice']
    titles = ['Water', 'Snow free', 'Snow cover', 'Ice']
    lws = [1, 1.5, 3, 1, 1]
    lss = ['-', '-.', ':', '-','--']
    for i, ax in enumerate(axes):
        var = variables[i]
        for model, lab, color, lw, ls in zip(models, models, colors, lws, lss):
            ax.plot(area_extent_df.index, area_extent_df[f'{model}_{var}_{plt_term}'], 
                    label=lab, color=color, lw=lw, ls=ls)

       # Set titles and adjust axes
        ax.set_title(titles[i], fontsize=18, loc='left')
        # Show only January and July on the x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))  # Major ticks for January and July
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as "Jan 1988, Jul 1988"
        # if decforma == True:
        #     ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2e}'))
            # ax.yaxis.set_major_formatter(FuncFormatter(make_scientific_formatter(2)))
        ax.grid(True)

        # Adding seasonal shading
        for year in range(area_extent_df.index.year.min(), area_extent_df.index.year.max() + 1):
            winter_start = pd.Timestamp(year=year, month=12, day=1)
            winter_end = pd.Timestamp(year=year + 1, month=2, day=28)
            summer_start = pd.Timestamp(year=year, month=6, day=1)
            summer_end = pd.Timestamp(year=year, month=8, day=31)

            # Shade the January-February winter months of the first year
            if year == area_extent_df.index.year.min():
                winter_start_ = pd.Timestamp(year=year, month=1, day=1)
                winter_end_ = pd.Timestamp(year=year, month=2, day=28)
                ax.axvspan(winter_start_, winter_end_, color='lightblue', alpha=0.3)  # Winter

            ax.axvspan(winter_start, winter_end, color='lightblue', alpha=0.3)  # Winter
            ax.axvspan(summer_start, summer_end, color='lightgrey', alpha=0.3)  # Summer

    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, 
                       right=True, bottom=True, left=True, labelsize=15)
        ax.grid(which='major', linestyle='--', linewidth='0.5', color='grey')

    # Labels and legend
    fig.text(0.5, 0.04, 'Year', ha='center', va='center', fontsize=18)
    fig.text(0.06, 0.5, 'Percent bias in area extent [%]', va='center', 
             rotation='vertical', fontsize=18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
               ncol=4, columnspacing=1, frameon=False, fontsize=18)

    # Save or display
    # plt.show()

#---------------------------------------------------------------
def scattter_compare(df2plot,plt_columns,dec_form):
    f,ax = plt.subplots(4,4, figsize=(20,14), gridspec_kw={'wspace':0.25,'hspace':0.15}, 
                    tight_layout=True,)# , dpi = 1000
    
    #***********************************************
    
    #define the limits ofthe scatter plot xy axes    
    x_limits = [(min(df2plot['gmasi_wtr_px_cnt'].min(), 
                 df2plot['ml_e_wtr_px_cnt'].min(), 
                 df2plot['ml_ec_wtr_px_cnt'].min(),
                 df2plot['climatology_wtr_px_cnt'].min(),
                 df2plot['ml_ecc_wtr_px_cnt'].min(),
                 ), # df2plot['e_wtr_px_cnt'].min()

             max(df2plot['gmasi_wtr_px_cnt'].max(),
                 df2plot['ml_e_wtr_px_cnt'].max(),
                 df2plot['ml_ec_wtr_px_cnt'].max(),
                 df2plot['climatology_wtr_px_cnt'].max(),
                 df2plot['ml_ecc_wtr_px_cnt'].max(),
                 )), #df2plot['e_wtr_px_cnt'].max()
            
             (min(df2plot['gmasi_snfr_px_cnt'].min(),
                  df2plot['ml_e_snfr_px_cnt'].min(),
                  df2plot['ml_ec_snfr_px_cnt'].min(),
                  df2plot['climatology_snfr_px_cnt'].min(),
                  df2plot['ml_ecc_snfr_px_cnt'].min(),
                  ), # df2plot['e_snfr_px_cnt'].min()
            
             max(df2plot['gmasi_snfr_px_cnt'].max(),
                 df2plot['ml_e_snfr_px_cnt'].max(),
                 df2plot['ml_ec_snfr_px_cnt'].max(),
                 df2plot['climatology_snfr_px_cnt'].max(),
                 df2plot['ml_ecc_snfr_px_cnt'].max(),
                 )), # df2plot['e_snfr_px_cnt'].max()

             (min(df2plot['gmasi_snc_px_cnt'].min(),
                  df2plot['ml_e_snc_px_cnt'].min(),
                  df2plot['ml_ec_snc_px_cnt'].min(),
                  df2plot['climatology_snc_px_cnt'].min(),
                  df2plot['ml_ecc_snc_px_cnt'].min(),
                  ), # df2plot['e_snc_px_cnt'].min()

             max(df2plot['gmasi_snc_px_cnt'].max(),
                 df2plot['ml_e_snc_px_cnt'].max(),
                 df2plot['ml_ec_snc_px_cnt'].max(),
                 df2plot['climatology_snc_px_cnt'].max(),
                 df2plot['ml_ecc_snc_px_cnt'].max(),
                 )), # df2plot['e_snc_px_cnt'].max()

             (min(df2plot['gmasi_ice_px_cnt'].min(),
                  df2plot['ml_e_ice_px_cnt'].min(),
                  df2plot['ml_ec_ice_px_cnt'].min(),
                  df2plot['climatology_ice_px_cnt'].min(),
                  df2plot['ml_ecc_ice_px_cnt'].min(),
                  ), #df2plot['e_ice_px_cnt'].min()

             max(df2plot['gmasi_ice_px_cnt'].max(),
                 df2plot['ml_e_ice_px_cnt'].max(),
                 df2plot['ml_ec_ice_px_cnt'].max(),
                 df2plot['climatology_ice_px_cnt'].max(),
                 df2plot['ml_ecc_ice_px_cnt'].max(),
                 ))] # df2plot['e_ice_px_cnt'].max()
    #***********************************************

    # # Share axes column-wise
    # for colmn in range(4):  # Assuming 4 columns based on your x_limits
    #     for ro in range(1, 3):  # Skip the first row, start with the second
    #         ax[ro, colmn].sharex(ax[0, colmn])
    #         ax[ro, colmn].sharey(ax[0, colmn])

    for i in enumerate(plt_columns):
        ax_ro = i[0]
        for ii in enumerate(i[1]):
            ax_col = ii[0]

            if ax_col == 0:
                ttle = 'Water'
            elif ax_col == 1:
                ttle = 'Snow free land'
            if ax_col == 2:
                ttle = 'Snow covered land'
            if ax_col == 3:
                ttle = 'Ice'

            if ax_ro == 0:
                mfcl,mecl = colors[0], 'k'
                prdct = 'ML-E'
                prd_psex,prd_psey = 1.02, 0.35

            elif ax_ro == 1:
                mfcl,mecl = colors[7], 'r'
                prdct = 'ML-EC'
                prd_psex,prd_psey = 1.02, 0.35

            elif ax_ro == 2:
                mfcl,mecl = colors[6], 'r'
                prdct = 'ML-ECC'
                prd_psex,prd_psey = 1.02, 0.35

            elif ax_ro == 3:
                mfcl,mecl = colors[4], 'k'
                prdct = 'CLIM'
                prd_psex,prd_psey = 1.02, 0.35

            # elif ax_ro == 4:
            #     mfcl,mecl = colors[7], 'k'
            #     prdct = 'E'
            #     prd_psex,prd_psey = 1.02, 0.45
        #***********************************************           

            # Set the axes limits
            ax[ax_ro, ax_col].set_xlim(x_limits[ax_col])
            ax[ax_ro, ax_col].set_ylim(x_limits[ax_col]) 

            # fomart axes tick labels to be readerble
            ax[ax_ro,ax_col].yaxis.set_major_formatter(FuncFormatter(make_scientific_formatter(dec_form)))
            ax[ax_ro,ax_col].xaxis.set_major_formatter(FuncFormatter(make_scientific_formatter(dec_form)))        

            xvar,yvar = df2plot[ii[1][1]], df2plot[ii[1][0]] # the x and y data

            # best_pos = find_best_position(ax[ax_ro, ax_col], xvar, yvar)

            # compute metrics
            # br = round(bias_ratio(xvar,yvar),3)
            # nrmse = round(nrmsqe(xvar,yvar),2)
            # pcor = round(p_corr(xvar,yvar),2)
            # kge_val = round(kge2012(xvar,yvar,'kge)[-1],2)
            mape = round(he.mape(yvar,xvar,remove_neg=True),2) # mean abs % error

            # plot scatter
            ax[ax_ro,ax_col].plot(xvar, yvar,markersize=3, marker='o', linestyle='none',
                                markerfacecolor=mfcl, markeredgecolor=mecl, markeredgewidth=0.2)    

            # Ensure the same limits for x and y axes
            # ax[ax_ro, ax_col].autoscale(False)  # Disable autoscaling
            # ax[ax_ro, ax_col].set_aspect('equal', 'box')  # Set equal scaling by changing aspect ratio
            lims = [
                np.min([ax[ax_ro, ax_col].get_xlim()[0], ax[ax_ro, ax_col].get_ylim()[0]]),  # min of both axes
                np.max([ax[ax_ro, ax_col].get_xlim()[1], ax[ax_ro, ax_col].get_ylim()[1]]),  # max of both axes
            ]
            ax[ax_ro, ax_col].set_xlim(lims)
            ax[ax_ro, ax_col].set_ylim(lims)

            # Now plot the 1:1 line
            ax[ax_ro, ax_col].plot(lims, lims, 'r--', linewidth=2.5, alpha=0.75, zorder=10)  # 1:1 line

            # add the metrics to the plot as text
            # txt = "RMSE = {0:.2f}\nMAPE = {1:.2f}\nCC = {2:.2f}".format(*[rmse,mape,pcor])  
            # txt = f"RMSE = {rmse / 1e6:.2f}M\nMAPE = {mape:.2f}\nCC = {pcor:.2f}".format(*[rmse,mape,pcor]) 
            txt = f"{mape:.2f} %".format(*[mape]) 

            ax[ax_ro,ax_col].text(0.53, 0.15, txt, horizontalalignment='left', 
                verticalalignment='center', transform = ax[ax_ro,ax_col].transAxes, 
                fontsize=15,) # bbox=dict(boxstyle='round', facecolor='silver', alpha=0.5)

        # Hide x-tick labels for all but the bottom row of the subplot in each column
            if ax_ro < 3:
                for label in ax[ax_ro, ax_col].get_xticklabels():
                    label.set_visible(False)
            
            if [ax_ro,ax_col] in [[0,3],[1,3],[2,3],[3,3], [4,3]]:
                ax[ax_ro,ax_col].text(prd_psex,prd_psey,prdct,fontsize=20,
                rotation='vertical', transform=ax[ax_ro,ax_col].transAxes) # ,fontweight='bold'

            if [ax_ro,ax_col] in [[0,0],[0,1],[0,2],[0,3]]:
                ax[ax_ro,ax_col].set_title(ttle,fontsize=20)     

    for a in ax.flatten():
        a.tick_params(axis='x', rotation=45)
        a.minorticks_on()
        a.tick_params(which='both', direction='in', top=True, right=True, bottom=True, left=True,labelsize=13)
        a.grid(which='major', linestyle='--', linewidth='0.5', color='grey')  

    f.text(0.08, 0.5, 'Daily grid box count of estimated surface cover type', 
           ha='center', va='center', rotation='vertical', fontsize=20)

    f.text(0.5,0.05, 'Daily grid box count of GMASI surface cover type', 
           ha='center', va='center', rotation='horizontal', fontsize=20)

#---------------------------------------------------------------
def plot_me(arr2plt,cmap):

    c_map = cm.get_cmap(cmap) #.gist_rainbow#

    c_map.set_bad('white')

    ax = plt.axes()

    ax_plt = ax.imshow(arr2plt, extent=img_extent,interpolation='nearest', cmap=c_map)  
    #fig = ax.imshow(arr2plt, extent=img_extent,interpolation='nearest', cmap=c_map)
    ax.grid(c = 'grey',ls = '--', lw = 0.35)
    ax.set_xlabel('Longitude',fontsize=15)
    ax.set_ylabel('Latitude',fontsize=15)
    ax.set_yticklabels([-80,-40,0,40,80])
    plt.tick_params(axis='both',  labelsize=15)
    cbar = plt.colorbar(ax_plt,orientation='horizontal',shrink=0.8,ax=ax,ticks = [-3,-2,-1,0,1,2,3])
    cbar.ax.tick_params(labelsize=12)

#---------------------------------------------   
def plot_bar(df,xytcklb_sze,tx_lbsze,plt_tle):

    # Create a bar plot for the specified columns
    # fig, ax = plt.subplots(figsize=(12, 6),dpi=500)
    ax = df.plot(kind='bar', figsize=(10, 6),colormap='RdYlGn_r') # 

    # Apply the custom y-axis label formatting
    ax.yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))
    # Rotate x-axis labels by 45 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=xytcklb_sze)
    ax.set_yticklabels(ax.get_yticklabels(),  ha='right', fontsize=xytcklb_sze)
    ax.grid(which='major',axis='both')

    # Annotate each bar with its value (rotated by 45 degrees, font color red, font weight normal)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height / 1e6:.1f}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom' if height >= 0 else 'top',
                    xytext=(0, 6) if height >= 0 else (0, -6), textcoords='offset points', 
                    rotation=45, color='k', weight='normal', fontsize=tx_lbsze,) 
    plt.ylabel('Count',fontsize=15)
    ax.set_xlabel('')
    # Access the default legend and set the font size
    ax.legend(loc='best',fontsize=xytcklb_sze, ncol=2)
    # Set plot title
    plt.title(plt_tle, fontsize=16)
#---------------------------------------------------------------
def plot_2d_hit_miss_map(cat_arr,area,clrmp,nbs):    

    if area == 'global':
        img_extent = (-180, 180, -90, 90)
        lblsze = 12
        yclse = -0.2
        # nro,ncol = 1,2
        wspce = 0.000005
        hspce = 0.08
    elif area == 'nh':
        img_extent = [-180,180,45,90]
        lblsze = 10
        yclse = -0.13
        # nro,ncol = 2,1
        wspce = 0.08
        hspce = 0.001
    elif area == 'sh':
        img_extent = [-180,180,-90,-45]
        lblsze = 10
        yclse = -0.13
        # nro,ncol = 2,1
        wspce = 0.08
        hspce = 0.000005

    minss,maxs = 0,365
    # Define color bins and create a color map with 90-100% as white for "Hits"
    bins_hits = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    colors_hits = ['#0000ff', '#1e90ff', '#00ffff', '#00ff7f', '#7fff00', 
                   '#ffff00', '#ff7f00', '#ff4500', '#ff0000', 'white']
    cmap_hits = mcolors.ListedColormap(colors_hits)
    norm_hits = mcolors.BoundaryNorm(bins_hits, cmap_hits.N)

    # Create a color map with 0-10% as white for "Miss"
    colors_miss = ['white', '#1e90ff', '#00ffff', '#00ff7f', '#7fff00', 
                   '#ffff00', '#ff7f00', '#ff4500', '#ff0000', '#ff0000']
    cmap_miss = mcolors.ListedColormap(colors_miss)
    norm_miss = mcolors.BoundaryNorm(bins_hits, cmap_miss.N)


    fig, axes = plt.subplots(3,2, figsize=(12, 4),sharey=True,sharex=True, dpi=1000,)
    plt.subplots_adjust(hspace=hspce,wspace=wspce)

    # ERA5 only
    axes[0,0].imshow(cat_arr[0][1],cmap =cmap_hits, extent=img_extent,norm=norm_hits)
    axes[0,0].set_title('Hits', fontsize =15)

    axes[0,1].imshow(cat_arr[0][2],cmap =cmap_miss, extent=img_extent,norm=norm_miss)
    axes[0,1].text(1.02, 0.15,cat_arr[0][0],fontsize=10, rotation='horizontal', 
                   transform=axes[0,1].transAxes,fontweight='bold')
    #-------------------------------------------------------
    
    # ERA5 with climatology
    axes[1,0].imshow(cat_arr[1][1],cmap = cmap_hits, extent=img_extent,norm=norm_hits)    

    axes[1,1].imshow(cat_arr[1][2],cmap = cmap_miss, extent=img_extent,norm=norm_miss)
    axes[1,1].text(1.02, 0.15,cat_arr[1][0],fontsize=10, rotation='horizontal', 
                   transform=axes[1,1].transAxes,fontweight='bold')
    #-------------------------------------------------------
    
    # ERA5 with climatology: corrected 
    axes[2,0].imshow(cat_arr[2][1],cmap = cmap_hits, extent=img_extent,norm=norm_hits)    

    axes[2,1].imshow(cat_arr[2][2],cmap = cmap_miss, extent=img_extent,norm=norm_miss)
    axes[2,1].text(1.02, 0.15,cat_arr[2][0],fontsize=10, rotation='horizontal', 
                   transform=axes[2,1].transAxes,fontweight='bold')
    #-------------------------------------------------------
    
    # Cimatology 
    axes[3,0].imshow(cat_arr[3][1],cmap = cmap_hits, extent=img_extent,norm=norm_hits)    

    axes[3,1].imshow(cat_arr[3][2],cmap = cmap_miss, extent=img_extent,norm=norm_miss)
    axes[3,1].text(1.02, 0.15,cat_arr[3][0],fontsize=10, rotation='horizontal', 
                   transform=axes[3,1].transAxes,fontweight='bold')               

    #-------------------------------------------------------

    # Add colorbar for "Hits"
    cbar_hits = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_hits, norm=norm_hits),
                            ax=axes[:, 0], orientation='horizontal', pad=0.1)
    cbar_hits.set_ticks((bins_hits[:-1] + np.diff(bins_hits) / 2))
    cbar_hits.set_ticklabels(['0-10', '10-20', '20-30', '30-40', '40-50',
                            '50-60', '60-70', '70-80', '80-90', '90-100'])
    cbar_hits.ax.tick_params(size=0)
    cbar_hits.set_label('Hits [%]', rotation=0, labelpad=-40)

    # Add colorbar for "Miss"
    cbar_miss = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_miss, norm=norm_miss),
                            ax=axes[:, 1], orientation='horizontal', pad=0.1)
    cbar_miss.set_ticks((bins_hits[:-1] + np.diff(bins_hits) / 2))
    cbar_miss.set_ticklabels(['0-10', '10-20', '20-30', '30-40', '40-50',
                            '50-60', '60-70', '70-80', '80-90', '90-100'])
    cbar_miss.ax.tick_params(size=0)
    cbar_miss.set_label('Miss [%]', rotation=0, labelpad=-40)

#-----------------------------------------------------------------------------------
def plot_snow_free_land_data(plot_df, year=None, endswith="snfr_px_cnt", ylabel="Pixel Count", ttle="Snow Free"):
    """
    Plot all columns that end with a specific suffix in the dataframe.
    Plots the Snow Free Land Pixel Count Over Time for the winter months (Dec, Jan, Feb) of a specific year or all years.

    Parameters:
    plot_df (DataFrame): The DataFrame containing the pixel count data.
    year (int, optional): The year for which to plot the data. If None, plots data for all years.
    endswith (str): The suffix of the columns to plot.
    ylabel (str): The label for the y-axis.
    ttle (str): The title of the plot.
    """
    colors = ['orange', 'g', 'm', 'b', 'k']
    models = ['ML-E', 'ML-EC', 'ML-ECC', 'CLIM', 'GMASI']
    lws = [1, 1.5, 3, 1, 1]
    lss = ['-', '-.', ':', '-', '--']

    if year:
        filtered_data = plot_df[(plot_df.index.year == year - 1) & (plot_df.index.month == 12) |
                                (plot_df.index.year == year) & (plot_df.index.month.isin([1, 2]))]
    else:
        # Use all data
        filtered_data = plot_df

    # Plot the data
    snfr_columns = [col for col in filtered_data.columns if col.endswith(endswith)]
    fig, ax = plt.subplots(figsize=(12, 6))
    for model, color, lw, ls in zip(models, colors, lws, lss):
        column = f'{model}_{endswith}'
        if column in snfr_columns:
            ax.plot(filtered_data.index, filtered_data[column], label=model, color=color, lw=lw, ls=ls)

    ax.set_title(f"{ttle} Land Over Time - Winter {year}" if year else f"{ttle} Land Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend(title="Methods")
    ax.grid(True)
    plt.show()
#-----------------------------------------------------------------------------------
def plot_time_series(area_extent_df):
    # Define colors for each model or dataset
    colors = ['orange', 'g', 'm', 'b',]

    # Convert index to datetime if it isn't already
    area_extent_df.index = pd.to_datetime(area_extent_df.index)

    # Creating subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True, dpi=1000,
                             gridspec_kw={'width_ratios': [0.98]*2},)
    plt.subplots_adjust(bottom=0.2)

    # Plotting
    variables = ['wtr', 'snfr', 'snc', 'ice']
    titles = ['Water', 'Snow free', 'Snow cover', 'Ice']
    lws = [1,1.5,3,1]
    lss = ['--','-.',':',':']
    for i, ax in enumerate(axes.flatten()):
        var = variables[i]
        for model, lab, color,lw, ls in zip(['ml_e', 'ml_ec', 'ml_ecc','climatology',], 
                                ['ML-E','ML-EC','ML-ECC','CLIM',], colors,lws,lss):
            ax.plot(area_extent_df.index, area_extent_df[f'{model}_gmasi_{var}_diff'], 
                    label=lab, color=color,lw=lw,ls=ls)

        # Set titles and adjust axes
        ax.set_title(titles[i],fontsize=15)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(True)

        # Adding seasonal shading
        for year in range(area_extent_df.index.year.min(), area_extent_df.index.year.max() + 1):
            winter_start = pd.Timestamp(year=year, month=12, day=1)
            winter_end = pd.Timestamp(year=year + 1, month=2, day=28)
            summer_start = pd.Timestamp(year=year, month=6, day=1)
            summer_end = pd.Timestamp(year=year, month=8, day=31)

            # color the Jan, Feb winter months of min year
            if year == area_extent_df.index.year.min():
                winter_start_ = pd.Timestamp(year=year, month=1, day=1)
                winter_end_ = pd.Timestamp(year=year , month=2, day=28)
                ax.axvspan(winter_start_, winter_end_, color='lightblue', alpha=0.3)  # Winter

            ax.axvspan(winter_start, winter_end, color='lightblue', alpha=0.3)  # Winter
            ax.axvspan(summer_start, summer_end, color='lightgrey', alpha=0.3)  # Summer

    for ax in axes.flatten():    

        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, 
                    right=True, bottom=True, left=True)
        ax.grid(which='major', linestyle='--', linewidth='0.5', color='grey')

    # Labels and legend
    fig.text(0.5, 0.1, 'Year', ha='center', va='center',fontsize=18)
    fig.text(0.06, 0.5, 'Percent bias in area  extent [%]', va='center', rotation='vertical',fontsize=15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), 
               ncol=4,columnspacing=1,frameon=False, fontsize=15)

    # plt.show()
#----------------------------------------------------------
# Figures 6 and 7
def plot_hemisphere_comparison(arr2_plt,hem):
  
    # Color configurations for "Miss"
    # Color configurations for "Miss"
    bins_hits = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    colors_miss_grey = ['#ffffff', '#e6e6e6', '#cccccc', '#b3b3b3', '#999999', 
                        '#808080', '#666666', '#4d4d4d', '#333333', '#1a1a1a', '#000000']
    cmap_miss = mcolors.ListedColormap(colors_miss_grey)
    norm_miss = mcolors.BoundaryNorm(bins_hits, cmap_miss.N)

    fig, axes = plt.subplots(4, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()},
                             dpi=1000)
    plt.subplots_adjust(bottom=0.02, hspace=0.01, wspace=0.12)  # Adjusted spacing

    extents = {
        'NH': [-180, 180, 30, 90],  # Northern Hemisphere extent
        'SH': [-180, 180, -90, -30]  # Southern Hemisphere extent
    }

    yticks = {
        'NH': range(30, 91, 20),  # Y-ticks for Northern Hemisphere
        'SH': range(-90, -29, 20)  # Y-ticks for Southern Hemisphere
    }

    for i, (product, summer_data, winter_data) in enumerate(arr2_plt):
        for j, data in enumerate([summer_data, winter_data]):
            ax = axes[i, j]
            hemisphere = hem
            extent = extents[hemisphere]
            ytcks = yticks[hemisphere]

            img = ax.imshow(data, cmap=cmap_miss, norm=norm_miss, extent=extent, transform=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.grid(which='major', linestyle='--', linewidth='0.25', color='grey')

            ax.set_xticks(range(-180, 181, 45))
            ax.set_yticks(ytcks)
            ax.tick_params(axis='x', direction='in', labelbottom=(i == 3))  # Label bottom only for the last row
            ax.tick_params(axis='y', direction='in', labelleft=True)  # Label left only for the first column

            if i == 0:
                ax.set_title('(a)' if j == 0 else '(b)', fontsize=20, fontweight='normal',va='center')
            
        # Place the product name to the far right of each row
        axes[i, 1].text(1.03, 0.5, product, fontsize=15, rotation='vertical', transform=axes[i, 1].transAxes, 
                        fontweight='normal', va='center')

    # Add one colorbar for both plots
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_miss, norm=norm_miss), shrink=0.8, extend='both',
                        ax=axes.ravel().tolist(), orientation='horizontal', pad=0.1, aspect=30)
    cbar.set_label('Percentage mismatch [%]', fontsize=15)
    bins_midpoints = (np.array(bins_hits[:-1]) + np.array(bins_hits[1:])) / 2
    cbar.set_ticks(bins_midpoints)
    cbar.set_ticklabels([f'{low}-{high}' for low, high in zip(bins_hits[:-1], bins_hits[1:])], fontsize=14)
    gc.collect()
    # plt.show()
#------------------------------------------------------------------------------------

def return_px_based_hit_miss_percent(sim_arr,obs_arr):
    hit_arr = make_2d_hit_miss_map(sim_arr,obs_arr,'hit')
    miss_arr = make_2d_hit_miss_map(sim_arr,obs_arr,'miss')
    total_count_per_px = hit_arr + miss_arr

    hit_percent = (hit_arr/total_count_per_px)*100
    miss_percent = (miss_arr/total_count_per_px)*100

    return hit_percent, miss_percent

#---------------------------------------------------------------------------------------
def plot_percent_hit_mis(arrs2plot,ext,ytcks):
    # function plot 2d array using descrete colormap

    # bins_hits = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # colors_hits = ['#0000ff', '#1e90ff', '#00ffff', '#00ff7f', '#7fff00', 
    #             '#ffff00', '#ff7f00', '#ff4500', '#ff0000', 'white']
    # cmap_hits = mcolors.ListedColormap(colors_hits)
    # norm_hits = mcolors.BoundaryNorm(bins_hits, cmap_hits.N)

    # # Create a color map with 0-10% as white for "Miss"
    # colors_miss = ['white', '#1e90ff', '#00ffff', '#00ff7f', '#7fff00', '#ffff00', 
    #             '#ff7f00', '#ff4500', '#ff0000', '#ff0000']
    # cmap_miss = mcolors.ListedColormap(colors_miss)
    # norm_miss = mcolors.BoundaryNorm(bins_hits, cmap_miss.N)
    #-------------------------------------------------

    # bins_hits = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # # Use a built-in sequential colormap for "Hits"
    # cmap_hits = plt.cm.YlGnBu
    # norm_hits = mcolors.BoundaryNorm(bins_hits, cmap_hits.N)

    # # Use the reverse of a built-in sequential colormap for "Miss"
    # cmap_miss = plt.cm.Blues_r
    # norm_miss = mcolors.BoundaryNorm(bins_hits, cmap_miss.N)
    #-------------------------------------------------

    # Define the bins for percentage ranges
    # bins_hits = np.linspace(0, 100, 11)

    # colors_hits = ['#00008B', '#0000CD', '#4169E1', '#6495ED', '#87CEEB', 
    #            '#ADD8E6', '#B0E0E6', '#E0FFFF', '#F0FFFF', '#FFFFFF', '#FFFFFF'] 

    # # Define colors for "Miss" that are the reverse of "Hits"
    # colors_miss = colors_hits[::-1]

    # # Create the colormaps
    # cmap_hits = mcolors.ListedColormap(colors_hits)
    # cmap_miss = mcolors.ListedColormap(colors_miss)

    # # Create the normalizations
    # norm_hits = mcolors.BoundaryNorm(bins_hits, cmap_hits.N)
    # norm_miss = mcolors.BoundaryNorm(bins_hits, cmap_miss.N)
    #-------------------------------------------------

    # num_bins = 11  # For 0-10%, 10-20%, ..., 90-100%
    # bins = np.linspace(0, 100, num_bins)

    # # Get the colors from the YlGnBu colormap
    # colors = plt.cm.YlGnBu(np.linspace(0, 1, num_bins - 2))
    # # Append white to the end for the highest values
    # colors_hits = np.vstack((colors, [(1,1,1,1), (1,1,1,1)]))

    # # For "Miss" we reverse the order, starting with white
    # colors_miss = np.vstack(([(1,1,1,1), (1,1,1,1)], colors[::-1]))

    # # Create the custom colormaps
    # cmap_hits = mcolors.LinearSegmentedColormap.from_list('custom_hits', colors_hits, N=num_bins)
    # cmap_miss = mcolors.LinearSegmentedColormap.from_list('custom_miss', colors_miss, N=num_bins)

    # # Create the normalization objects
    # norm_hits = mcolors.BoundaryNorm(bins, cmap_hits.N)
    # norm_miss = mcolors.BoundaryNorm(bins, cmap_miss.N)
    #-------------------------------------------------
    # Create a grayscale colormap for "Hits" where black represents the lowest percentage range (0-10%)
    bins_hits = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    colors_hits_grey = ['#000000', '#1a1a1a', '#333333', '#4d4d4d', '#666666', 
                        '#808080', '#999999', '#b3b3b3', '#cccccc', '#e6e6e6', '#ffffff']
    cmap_hits = mcolors.ListedColormap(colors_hits_grey)
    norm_hits = mcolors.BoundaryNorm(bins_hits, cmap_hits.N)

    # Create a grayscale colormap for "Miss" where black represents the highest percentage range (90-100%)
    colors_miss_grey = ['#ffffff', '#e6e6e6', '#cccccc', '#b3b3b3', '#999999', 
                        '#808080', '#666666', '#4d4d4d', '#333333', '#1a1a1a', '#000000']
    cmap_miss = mcolors.ListedColormap(colors_miss_grey)
    norm_miss = mcolors.BoundaryNorm(bins_hits, cmap_miss.N)
    #-------------------------------------------------

    fig, axes = plt.subplots(4,2, figsize=(20, 5),sharey=True,sharex=True,dpi=1000,)
    plt.subplots_adjust(bottom=0.0000001,hspace=0.2,wspace=0.005) # 

    # ERA5 only
    axes[0,0].imshow(arrs2plot[0][1],cmap =cmap_hits, extent=ext,norm=norm_hits)

    axes[0,1].imshow(arrs2plot[0][2],cmap =cmap_miss, extent=ext,norm=norm_miss)
    axes[0,1].text(1.02, 0.25,arrs2plot[0][0],fontsize=10, rotation='vertical', 
                    transform=axes[0,1].transAxes,fontweight='bold')
    #-------------------------------------------------------

    # ERA5 with climatology
    axes[1,0].imshow(arrs2plot[1][1],cmap = cmap_hits, extent=ext,norm=norm_hits)    

    axes[1,1].imshow(arrs2plot[1][2],cmap = cmap_miss, extent=ext,norm=norm_miss)
    axes[1,1].text(1.02, 0.25,arrs2plot[1][0],fontsize=10, rotation='vertical', 
                    transform=axes[1,1].transAxes,fontweight='bold')
    #-------------------------------------------------------

    # ERA5 with climatology: corrected 
    axes[2,0].imshow(arrs2plot[2][1],cmap = cmap_hits, extent=ext,norm=norm_hits)    

    axes[2,1].imshow(arrs2plot[2][2],cmap = cmap_miss, extent=ext,norm=norm_miss)
    axes[2,1].text(1.02, 0.04,arrs2plot[2][0],fontsize=10, rotation='vertical', 
                    transform=axes[2,1].transAxes,fontweight='bold')
    #-------------------------------------------------------

    # Cimatology 
    axes[3,0].imshow(arrs2plot[3][1],cmap = cmap_hits, extent=ext,norm=norm_hits)    

    axes[3,1].imshow(arrs2plot[3][2],cmap = cmap_miss, extent=ext,norm=norm_miss)
    axes[3,1].text(1.02, 0.25,arrs2plot[3][0],fontsize=10, rotation='vertical', 
                    transform=axes[3,1].transAxes,fontweight='bold')       

    # E 
    # axes[4,0].imshow(arrs2plot[4][1],cmap = cmap_hits, extent=ext,norm=norm_hits)    

    # axes[4,1].imshow(arrs2plot[4][2],cmap = cmap_miss, extent=ext,norm=norm_miss)
    # axes[4,1].text(1.02, 0.025,arrs2plot[4][0],fontsize=10, rotation='vertical', 
    #                 transform=axes[4,1].transAxes,fontweight='bold')         

    #-------------------------------------------------------
    # Create the colorbar for "Hits"
    cbar_hits = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_hits, norm=norm_hits),
                            ax=axes[:, 0], # Applies the colorbar to the axes on the first column
                            orientation='horizontal',
                            fraction=0.04, # Adjusts the length of the colorbar
                            pad=0.08, # Adjusts the distance between the colorbar and the plots
                            aspect=40,# Adjusts the aspect ratio of the colorbar
                            shrink=0.95) # Adjusts the length of the colorbar relative to the plot

    cbar_hits.set_label('Fraction of match [%]', fontsize=15)
    bins_midpoints = (np.array(bins_hits[:-1]) + np.array(bins_hits[1:])) / 2
    #-------------------------------------------------

    # Corrected setting of ticks and labels for "Hits" colorbar
    cbar_hits.set_ticks(bins_midpoints)
    cbar_hits.ax.set_xticklabels(['0-10', '10-20', '20-30', '30-40', '40-50', 
                                '50-60', '60-70', '70-80', '80-90', '90-100'], fontsize=12)

    # Create the colorbar for "Miss"
    cbar_miss = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_miss, norm=norm_miss),
                            ax=axes[:, 1], # Applies the colorbar to the axes on the second column
                            orientation='horizontal',
                            fraction=0.04, # Adjusts the length of the colorbar
                            pad=0.08, # Adjusts the distance between the colorbar and the plots
                            aspect=40, # Adjusts the aspect ratio of the colorbar 
                            shrink=0.95) # Adjusts the length of the colorbar relative to the plot

    cbar_miss.set_label('Fraction of mismatch [%]', fontsize=15)
    # Corrected setting of ticks and labels for "Hits" colorbar
    cbar_miss.set_ticks(bins_midpoints)
    cbar_miss.ax.set_xticklabels(['0-10', '10-20', '20-30', '30-40', '40-50', 
                                '50-60', '60-70', '70-80', '80-90', '90-100'], fontsize=12)
#-------------------------------------------------
    
    for a in axes.flatten():
        
        a.minorticks_on()
        a.tick_params(which='major', axis='both', direction='in', length=2.5, top=True, right=True,
                       bottom=True, left=True, labelsize=12)

        a.tick_params(which='minor', axis='both', direction='in', length=1, top=True, right=True, 
                      bottom=True, left=True)
        a.grid(which='major', linestyle='--', linewidth='0.15', color='grey') 

        a.set_xticks(range(-180, 181, 45))  # x-axis from -180 to 180 at intervals of 30
        a.set_yticks(ytcks)    # y-axis from -90 to 90 at intervals of 5       
