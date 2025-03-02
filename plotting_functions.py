#%%
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,LogNorm, Normalize, ListedColormap
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.dates as mdates

# Import packages
#%%
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
def plot_time_series_4x1(area_extent_df,plt_term, plt_type):
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
