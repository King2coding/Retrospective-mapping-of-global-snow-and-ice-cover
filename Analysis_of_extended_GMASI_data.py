#%%
"""
This code analysed 20 years of snow-ice data which includes data from the extended period 
(1980-1987) and GMASI Autosnow data to see if the extended data captured seasonal patterns 
already existing in the alriginal data
"""

#%%
# import packages
import warnings
warnings.filterwarnings('ignore')

import gc
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from util_functions import *
from eval_functions import *
from plotting_functions import *

import rasterio
import xarray as xr
from rasterio import transform


#%%
# define paths

path_to_extended_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_extended_mid1987_1980'
path_to_gmasi_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_in_geotif'

path_to_put_plots = r'/home/kkumah/Autosnow_extending/results/plots/save_plots_Dec_2024'

path_to_put_intermediate_files = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/extending_autosnow_estimated/intermediate_files'

#%%
# define global variables
integer_list = [0, 1, 2, 3]
classes = ['Water','Snow free land', 'Snow covered land','Ice']

# Define the range for filtering
gmasi_start_yeardoy = 1987182  # July 1, 1987
gmasi_end_yeardoy = 2000366    # December 31, 2000 (handles leap years)

# Select files within the range
all_gmasi_files = [
    os.path.join(path_to_gmasi_data, a)
    for a in os.listdir(path_to_gmasi_data)
    if gmasi_start_yeardoy <= int(a.split('_')[4]) <= gmasi_end_yeardoy
]

all_extended_files = [os.path.join(path_to_extended_data, a) 
                      for a in os.listdir(path_to_extended_data)
                      if a.endswith('.nc')]

# Combine the two lists
all_files = all_gmasi_files + all_extended_files

# Sort the combined list based on extracted YYYYDDD
sorted_files = sorted(all_files, key=lambda x: extract_yeardoy(x))

# read one autosnow file and store it metadata
metafile = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_in_geotif/gmasi_snowice_reproc_v003_1987241_0.1deg_wgs.tif'

with rasterio.open(metafile) as dt: 

    nh_row = transform.rowcol(dt.transform,-180,30)

    sh_row = transform.rowcol(dt.transform,-180,-30)   

    meta_autosnow = dt.meta

    meta_crs = meta_autosnow['crs']

    meta_trns = meta_autosnow['transform']

# Define pixel size
pixel_width = 0.1  # Longitude resolution (degrees)
pixel_height = 0.1  # Latitude resolution (degrees)

# Global width in pixels (longitude range)
lon_range = 360  # -180° to 180°
pixels_per_lon = int(lon_range / pixel_width)  # Total number of pixels in longitude

# Define Northern Hemisphere (NH) Transform: 90°N to 30°N
upper_left_x_NH = -180.0  # Upper-left longitude
upper_left_y_NH = 90.0    # Upper-left latitude
pixels_per_lat_NH = int((90 - 30) / pixel_height)  # Total latitude rows from 90°N to 30°N

transform_NH = transform.from_origin(upper_left_x_NH, upper_left_y_NH, pixel_width, pixel_height)

# Define Southern Hemisphere (SH) Transform: 30°S to 90°S
upper_left_x_SH = -180.0  # Upper-left longitude
upper_left_y_SH = -30.0   # Upper-left latitude
pixels_per_lat_SH = int((90 - 30) / pixel_height)  # Total latitude rows from 30°S to 90°S

transform_SH = transform.from_origin(upper_left_x_SH, upper_left_y_SH, pixel_width, -pixel_height)

# Print the results
print("Northern Hemisphere Transform:", transform_NH)
print("Southern Hemisphere Transform:", transform_SH)

# For dimensions (height and width)
print(f"NH Dimensions: {pixels_per_lat_NH} rows x {pixels_per_lon} columns")
print(f"SH Dimensions: {pixels_per_lat_SH} rows x {pixels_per_lon} columns")

gc.collect()

cde_run_dte = str(date.today().strftime('%Y%m%d'))

  
#%%
area_extent_NH = pd.DataFrame()
area_extent_SH = pd.DataFrame()

area_extent_ras = pd.DataFrame()
area_extent_pol = pd.DataFrame()

count = 0
# begin process
for fle in sorted_files:
    yr_DOY = str(extract_yeardoy(fle))
    yr,doY = yr_DOY[:4], yr_DOY[-3:]
    # Create a datetime object
    date_time = pd.to_datetime(f'{yr}-{doY}', format='%Y-%j')

    # read the file
    file_path, filename = os.path.split(fle)
    filename, ext = os.path.splitext(filename)
    colname = 'Extended' if filename.startswith('UofA') else 'GMASI'
    snow_ice_data = read_processed_files(file_path,filename.split('_'),ext)

    y_shp,x_shp = snow_ice_data.shape

    snow_ice_data_nh = snow_ice_data[0:nh_row[0],:].astype(np.int16)

    snow_ice_data_sh = snow_ice_data[sh_row[0]:y_shp,:].astype(np.int16)

    # calcualte_total_area(area_extent_pol, snow_ice_data, path_to_put_intermediate_files, 
    #                      yr_DOY, date_time, colname + '_global', integer_list, 
    #                      meta_crs, meta_trns)
    
    calcualte_total_area(area_extent_pol, snow_ice_data_nh, path_to_put_intermediate_files, 
                         yr_DOY, date_time, colname + '_NH', integer_list, 
                         meta_crs, transform_NH)
    
    calcualte_total_area(area_extent_pol, snow_ice_data_sh, path_to_put_intermediate_files, 
                         yr_DOY, date_time, colname + '_SH', integer_list, 
                         meta_crs, transform_SH)

    snow_ice_data = xr.open_dataarray(fle)
    snow_ice_data = snow_ice_data.rename({dim: new_name for dim, new_name in {'y': 'lat', 'x': 'lon'}.items() if dim in snow_ice_data.dims})
    snow_ice_data_nh = snow_ice_data.sel(lat=slice(90,30))
    snow_ice_data_sh = snow_ice_data.sel(lat=slice(-30,-90))

    compute_area_from_raster(area_extent_ras, date_time, snow_ice_data_nh, 
                             0.1, colname + '_NH')
    
    compute_area_from_raster(area_extent_ras, date_time, snow_ice_data_sh, 
                             0.1, colname + '_SH')
    
    count += 1
    if count % 50 == 0:
        print(str(count) + ' files are read so far')
    
gc.collect()

#%%

# Step 1: Extract relevant columns
snow_cols = ['Extended_NH_snc_total_area', 'GMASI_NH_snc_total_area']
ice_cols = ['Extended_NH_ice_total_area', 'GMASI_NH_ice_total_area']

# Extract data
snow_extended = area_extent_pol.loc[:'1987-06-30', snow_cols[0]]
snow_gmasi = area_extent_pol.loc['1987-07-01':, snow_cols[1]]

ice_extended = area_extent_pol.loc[:'1987-06-30', ice_cols[0]]
ice_gmasi = area_extent_pol.loc['1987-07-01':, ice_cols[1]]

# Step 2: Combine 1987 data
snow_combined = pd.concat([snow_extended, snow_gmasi])
ice_combined = pd.concat([ice_extended, ice_gmasi])

# Step 3: Mark data sources
snow_sources = ['Extended'] * len(snow_extended) + ['GMASI'] * len(snow_gmasi)
ice_sources = ['Extended'] * len(ice_extended) + ['GMASI'] * len(ice_gmasi)

# Step 4: Plot
#%%
fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True, dpi=1000)
plt.subplots_adjust(hspace=0.2)

# Define colors for seasons
winter_color = 'lightblue'
summer_color = 'lightgrey'

# Snow-Covered Area Plot
for source in ['Extended', 'GMASI']:
    mask = [src == source for src in snow_sources]
    axes[0].plot(snow_combined.index[mask], snow_combined.values[mask], 
                 label=source, lw=2, color='orange' if source == 'Extended' else 'blue')

axes[0].set_title('Snow cover', fontsize=18)
axes[0].set_ylabel('Area Extent [sq. km]', fontsize=18)
axes[0].grid(which='both', linestyle='--', linewidth=0.5)

# Ice-Covered Area Plot
for source in ['Extended', 'GMASI']:
    mask = [src == source for src in ice_sources]
    axes[1].plot(ice_combined.index[mask], ice_combined.values[mask], 
                 label=source, lw=2, color='orange' if source == 'Extended' else 'blue')

axes[1].set_title('Ice', fontsize=18)
axes[1].set_ylabel('Area Extent [sq. km]', fontsize=18)
axes[1].grid(which='both', linestyle='--', linewidth=0.5)

# Add seasonal shading
for ax in axes:
    for year in range(snow_combined.index.year.min(), snow_combined.index.year.max() + 1):
        # Winter: Dec 1 (year) to Feb 28 (year + 1)
        winter_start = pd.Timestamp(year=year, month=12, day=1)
        winter_end = pd.Timestamp(year=year + 1, month=2, day=28)
        ax.axvspan(winter_start, winter_end, color=winter_color, alpha=0.3)
        
        # Summer: Jun 1 to Aug 31
        summer_start = pd.Timestamp(year=year, month=6, day=1)
        summer_end = pd.Timestamp(year=year, month=8, day=31)
        ax.axvspan(summer_start, summer_end, color=summer_color, alpha=0.3)
        
        # January-February for the first year
        if year == snow_combined.index.year.min():
            winter_start_ = pd.Timestamp(year=year, month=1, day=1)
            winter_end_ = pd.Timestamp(year=year, month=2, day=28)
            ax.axvspan(winter_start_, winter_end_, color=winter_color, alpha=0.3)

# Shared x-axis label
fig.text(0.5, 0.04, 'Year', ha='center', va='center', fontsize=18)

# Scientific notation for y-axis
for ax in axes:
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    
    # Increase font size of scientific notation
    ax.yaxis.offsetText.set_fontsize(18)

    # Add ticks on all sides
    ax.tick_params(axis='both', which='major', direction='in', 
                   labelsize=18, length=8, width=1.5, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1, top=True, right=True)
    ax.minorticks_on()

# Add single legend outside the plot
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), 
               fontsize=18, ncol=2, frameon=False)

plt.tight_layout()
# plt.show()
svnem_csv = '_'.join(['Extended_data_analysis',cde_run_dte])+ '.png'
plt.savefig(os.path.join(path_to_put_plots, svnem_csv), bbox_inches='tight')

plt.close()