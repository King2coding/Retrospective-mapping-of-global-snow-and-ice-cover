#%%
# import packages
import warnings
warnings.filterwarnings('ignore')
# import datetime

from pyproj import CRS
import rioxarray as rxr
from rasterio.warp import Resampling
from datetime import datetime, timedelta, date
from scipy import stats
from scipy.ndimage import distance_transform_edt, generic_filter

import os
import pickle
import pandas as pd
import numpy as np
import gc

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,LogNorm, Normalize, ListedColormap
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.dates as mdates
from matplotlib.cm import ScalarMappable
import matplotlib
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

from util_functions import *
from eval_functions import *
from plotting_functions import *
import HydroErr as he

import rasterio
from rasterio import transform
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor


#%%
# define path to data
path_to_autosnow_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_in_geotif'

path_to_estimated_autosnw = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_estimated_1988_1991'

path_to_clim_only_autosnw_estimated = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/extending_autosnow_estimated/all_data_1992_2022_airTemp_subsetted_climatology_estimate'

path_to_era5_based_snowice = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/extending_autosnow_estimated/global_snow_icer_cover_estimated_using_era5_snow_and_seaice_cover'

path_to_put_df = r'/home/kkumah/Projects/Autosnow_extending/results/dfs/save_df_Feb2025'

path_to_put_plots = r'/home/kkumah/Projects/Autosnow_extending/results/plots/save_plots_Feb2025'

path_to_put_ancillary = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/ancillary_hit_miss_maps'

path_to_put_intermediate_files = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/intermediates'

path_to_rutgers_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/Rutgers_24km_NH_SCE'

path_to_era5_vars = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/era5_daily_data_for_extending_autosnow'

dir_of_diff_est_meth = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/extending_autosnow_estimated'
#%%
# decalre global variables
all_autosnow_files = [os.path.join(path_to_autosnow_data,a) for a in os.listdir(path_to_autosnow_data) if (int(a.split('_')[4][:4]) >=1988) and (int(a.split('_')[4][:4]) <=1991)]

classes = ['Water','Snow free land', 'Snow covered land','Ice']

# read one autosnow file and store it metadata
metafile = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_in_geotif/gmasi_snowice_reproc_v003_1987241_0.1deg_wgs.tif'

with rasterio.open(metafile) as dt:

    nh_row = transform.rowcol(dt.transform,-180,30)

    nh_row60 = transform.rowcol(dt.transform,-180,60)

    sh_row = transform.rowcol(dt.transform,-180,-30)

    sh_row60 = transform.rowcol(dt.transform,-180,-60)

    nh_row_90 = transform.rowcol(dt.transform,-180,0)

    meta_autosnow = dt.meta

    meta_crs = meta_autosnow['crs']

    meta_trns = meta_autosnow['transform']

    meta_shp = dt.read(1).shape

img_extent = (-180, 180, -90, 90)

plt_extent = [-180,180,-90,90]

plt_extent_nh = [-180,180,45,90]

plt_extent_sh = [-180,180,-90,-45]

gc.collect()

#-------------------------------------------------------

integer_list = [0, 1, 2, 3]

cde_run_dte = str(date.today().strftime('%Y%m%d'))

remove_elem = ['RF_estimated_autosnow_using','gmasi_snowice_reproc',
               'corrected_RF_estimated_autosnow_using_alldataclim',
               'ERA5_seaice-snowcover']

colors = ['orange', 'k', 'crimson', 'cyan','b', 'lime','m','g','r']
#-------------------------------------------------------

# land sea mask
land_sea_mask = r'/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/ancillary_imerg_data/GPM_IMERG_LandSeaMask.2.nc4'
lsm_arr = xr.open_dataset(land_sea_mask).landseamask.data
# transpose the data to get longitude on the x axis, and flip vertically
# so that latitude is displayed south to north as it should be
lsm_arr = np.flip(lsm_arr.transpose(), axis=0)
lsm = np.where(lsm_arr < 25, 1, 0)
#-------------------------------------------------------
rutgers_data = 'G10035-rutgers-nh-24km-weekly-sce-v01r00-19800826-20220905_01_wgs.nc' # rutgers sce data

ds_rutgers = xr.open_dataset(os.path.join(path_to_rutgers_data,rutgers_data),decode_coords="all")

# crs = ds_rutgers.snow_cover_extent.rio.crs.to_string()

crfs_dest = '+proj=longlat +datum=WGS84 +no_defs +type=crs'

cc = CRS.from_string(crfs_dest)#from_authority(code=4326,auth_name='EPSG')

ds_rutgers.rio.write_crs(cc,inplace=True)

times = list(pd.to_datetime(ds_rutgers.time.values))
#------------------------------------------------------------
mesh_xy = np.meshgrid(np.arange(-180, 180, 0.1), np.arange(90, -90, -0.1))

lons,lats = mesh_xy[0],mesh_xy[1]

antartica_msk = (lats <= -66.5) & (lsm == 1) # antartica lands mask

# Tropical and equatorial regions: 25°S to 25°N, assumed snow free
tropical_mask = (lats >= -25) & (lats <= 25) & (lsm == 1)

# get the land water mask from Rutgers data
# lw_msk = np.flipud(ds_rutgers.land.values)
"""
0: water
1: snow-free land
2: snow-covered land
3: ice
"""


#%%
# define fucntions
def extract_method_surface_type(index):
    parts = index.split('_')
    method = '_'.join([parts[0],parts[1]])
    surface_type = [i for i in parts if i in ['wtr', 'snfr', 'snc', 'ice']][0]
    return method, surface_type
#---------------------------------------------

# define a plot function
import matplotlib.cm as cm
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
# Function to format y-axis labels
def scientific_notation_formatter(x, pos):
    """
    Format y-axis labels using scientific notation with a base of 1e6.
    """
    return f'{x / 1e6:.0f}M'
#---------------------------------------------

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
#----------------------------------------------
# fcuntion to make big 1d data from list of arrays
def make_nd_arr(array_lst):

    # Create a 3D array by stacking the 2D arrays along the third axis
    arr3d = np.stack(array_lst, axis=2).astype(np.int8)    
    arr1d = arr3d.reshape(-1)
    del(arr3d)
    return arr1d
#---------------------------------------------------
def process_in_chunks(arr, chunk_size):
    # Placeholder for the processed parts
    processed_parts = []
    
    # Calculate the number of chunks
    num_chunks = np.ceil(arr.shape[0] / chunk_size).astype(int)
    
    for i in range(num_chunks):
        # Calculate start and end indices of the current chunk
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, arr.shape[0])
        
        # Extract the chunk
        chunk = arr[start_idx:end_idx]
        
        # Filter out NaN rows in the chunk and convert dtype
        filtered_chunk = chunk[~np.isnan(chunk).any(axis=1)].astype(np.int8)
        
        # Store the processed chunk
        processed_parts.append(filtered_chunk)

        gc.collect()
    
    # Combine processed parts back into a single array
    return np.concatenate(processed_parts, axis=0)
#--------------------------------------------------
def cat_evaluate(lst_of_arrays, prdct):

    orig_arrs = lst_of_arrays[0]

    # Loop through the list and modify each array in place
    for i in range(len(orig_arrs)):
        orig_arrs[i] = orig_arrs[i].astype(np.int8)
    del(i)

    rf_arrs = lst_of_arrays[1]
    for i in range(len(rf_arrs)):
        rf_arrs[i] = rf_arrs[i].astype(np.int8)
    del(i)

    orig_1d = make_nd_arr(orig_arrs)
    rf_1d = make_nd_arr(rf_arrs)

    gc.collect()

    arr_stck = np.column_stack((orig_1d,rf_1d))
    arr_stck = arr_stck.astype(np.int8)
    
    # arr_stck = arr_stck[~np.isnan(arr_stck).any(axis=1)]

    processed_arr_stck = process_in_chunks(arr_stck, 100000)

    df_obs_pred = pd.DataFrame(processed_arr_stck,columns=['test','pred']) # ,dtype=np.int8

    # Convert dtypes after creation
    df_obs_pred = df_obs_pred.astype(np.int8)  # Converting all columns to int8

    # get the count per class
    n_per_class_in_y = pd.DataFrame(df_obs_pred['test'].value_counts())
    n_per_class_in_y.sort_index(inplace=True)
    n_per_class_in_y['Class']  = classes # per it with the class names

    del(orig_1d, rf_1d,arr_stck)

    gc.collect()

    #---------------------#------------------

    # here we evaluate per class
    lst_cat_stats = []
    for i in enumerate(classes):

        tval = float(i[0])

        class_val = i[1]

        lst_cat_stats.append(binary_cat_metrics(df_obs_pred,'test','pred',tval,class_val))

    dfs_cat_stats = pd.concat(lst_cat_stats,axis=0)

    dfs_cat_stats = dfs_cat_stats.merge(n_per_class_in_y[['count', 'Class']],right_on='Class',
                                        left_on=dfs_cat_stats.index)

    dfs_cat_stats['label'] = dfs_cat_stats.index

    dfs_cat_stats.index = dfs_cat_stats['Class']

    dfs_cat_stats.drop(columns='Class',inplace=True)

    dfs_cat_stats = pd.DataFrame(dfs_cat_stats.filter(items=['count','Hits', 'Miss', 'label',
                                                             'POD', 'FAR', 'Bias', 'CSI']))
    dfs_cat_stats['Product'] = prdct
    dfs_cat_stats = pd.DataFrame(dfs_cat_stats.filter(items=['Product','count','Hits', 'Miss', 'label',
                                                             'POD', 'FAR', 'Bias','CSI']))

    del(df_obs_pred)

    return dfs_cat_stats
#---------------------------------------------------------------------
# make bar plot
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
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------

# compute hit miss count per pixel
# def make_2d_hit_miss_map(pred_arr_lst,observ_arr_lst,kind):
#     pred_3d_arr = np.dstack(pred_arr_lst)
#     obs_3d_arr = np.dstack(observ_arr_lst)

#     # Extract slices along the third axis (depth) and apply the calculate_hits_miss function
#     hits_miss_per_pixel = np.apply_along_axis(
#                         lambda x: calculate_hits_miss(x[0], x[1], integer_list,kind),
#                         axis=2,
#                         arr=np.stack([pred_3d_arr, obs_3d_arr], axis=2)
#                         )
    
#     return hits_miss_per_pixel
#--------------------------------------------------------------------------

def calculate_seasonal_means(df, columns, season):
    """
    Calculate the mean of specified columns in a dataframe for a given season.

    Args:
    df (pd.DataFrame): Input dataframe with datetime index.
    columns (list): List of column names to calculate the mean.
    season (str): Season to filter by ('winter', 'spring', 'summer', 'autumn').

    Returns:
    pd.Series: Mean of the specified columns for the given season.
    """
    # Season month mapping
    season_months = {
        'winter': [12, 1, 2],
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'autumn': [9, 10, 11]
    }
    
    # Check if the season is valid
    if season not in season_months:
        raise ValueError(f"Invalid season '{season}'. Choose from 'winter', 'spring', 'summer', 'autumn'.")
    
    # Filter the dataframe for the months corresponding to the season
    mask = df.index.month.isin(season_months[season])
    season_df = df.loc[mask]
    
    # Calculate mean of the specified columns
    mean_values = season_df[columns].mean()

    std_values = season_df[columns].std()
    
    return mean_values, std_values

#--------------------------------------------------------------------------
def make_2d_hit_miss_map(pred_arr_lst,observ_arr_lst,kind):

    arr_yshp,arr_xshp = pred_arr_lst[0].shape[0], pred_arr_lst[0].shape[1]

    pred_3d_arr = np.dstack(pred_arr_lst)
    obs_3d_arr = np.dstack(observ_arr_lst)

    lst_row_col_idx = []

    for row in range(0,arr_yshp):

        for col in range(0,arr_xshp):

            id_tuple = tuple((row,col))

            lst_row_col_idx.append(id_tuple)
    
    out_arr = np.empty(pred_arr_lst[0].shape,dtype=np.int32)

    for i in lst_row_col_idx:

        pred_lst = pred_3d_arr[i[0],i[1],:]
        obs_lst = obs_3d_arr[i[0],i[1],:]

        out_arr[i[0],i[1]] = calculate_hits_miss(pred_lst,obs_lst,integer_list,kind)

    del(pred_3d_arr,obs_3d_arr,arr_xshp,arr_yshp,lst_row_col_idx,
        row,col,id_tuple,pred_lst,obs_lst)
    
    gc.collect()

    return out_arr
#--------------------------------------------------------------------------

def custom_formatter(x, pos):
    if float(x).is_integer():
        return int(x)  
    else:
        return f'{x:.0f}'
#--------------------------------------------------------------------------

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

#----------------------------------------------------------

def find_relevant_monday(input_date_str):
    # Convert input string to datetime object
    input_date = datetime.strptime(input_date_str, '%Y-%m-%d').date()
    
    # Calculate the day of the week (0=Monday, 6=Sunday)
    day_of_week = input_date.weekday()
    
    # If the date is already a Monday, return it directly
    if day_of_week == 0:
        return input_date
    else:
        # Find the next Monday
        days_until_next_monday = 7 - day_of_week
        next_monday = input_date + timedelta(days=days_until_next_monday)
        return next_monday

#------------------------------------------
def find_monday_of_same_week_past_years(input_date_str, past_years):
    input_date = datetime.strptime(input_date_str, '%Y-%m-%d')
    week_number = input_date.isocalendar()[1]
    
    mondays_of_same_week = []
    for year in past_years:
        # Find the first Monday of the year
        year_start = datetime(year, 1, 1)
        first_monday = year_start + timedelta(days=(7-year_start.weekday()) % 7)
        # Calculate the Monday of the same week number
        monday_of_same_week = first_monday + timedelta(weeks=week_number-1)
        mondays_of_same_week.append(monday_of_same_week.date())
    
    return mondays_of_same_week
#------------------------------------------------

def get_aggregated_dates(input_date_str):
    # Parse the input date
    input_date = datetime.strptime(input_date_str, '%Y-%m-%d')
    
    # Check if the input date is a Monday
    if input_date.weekday() != 0:
        raise ValueError("The input date must be a Monday.")
    
    # Calculate the start date as the Tuesday before the last week
    start_date = input_date - timedelta(days=6)  # Previous Tuesday
    
    # Generate the list of dates for the aggregation period
    aggregated_dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    
    return aggregated_dates

#----------------------------------------------------------
def day_of_year(date_str):
    # Parse the input string to a datetime object
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    # Get the day of the year
    day_of_year = date_obj.timetuple().tm_yday
    return day_of_year
#---------------------------------------------

def correct_pixel(values):
    # Center pixel is values[4] in a 3x3 kernel
    # center_pixel = values[4]
    # if center_pixel == 3 and np.any(values[[0, 1, 2, 3, 5, 6, 7, 8]] == 1):  # Check for land in the immediate vicinity
    #     return 2  # Reclassify as snow-covered land
    # return center_pixel
    center_pixel = values[4]
    surrounding_land_count = np.sum(np.array(values) == 1)
    surrounding_water_count = np.sum(np.array(values) == 0)
    
    # Only reclassify ice (3) to snow-covered land (2) if there's significant land presence
    if center_pixel == 3 and surrounding_land_count > 4:
        return 2  # Reclassify as snow-covered land
    elif center_pixel == 3 and surrounding_water_count >= 5:
        return 3  # Keep as ice to preserve small water bodies covered with ice
    return center_pixel
#---------------------------------------------
# fucntion to make nd array
def make_nd_array(bkst_arrs,yshp,):
    '''
    requires 
    bkst_arrs = list of arrays
    yshp = number of rows in array, i.e. array.shape[0]
    '''

    arr_3d = np.dstack(bkst_arrs)        

    arr_1d = arr_3d.reshape((yshp*x_shp*arr_3d.shape[2]))

    del(arr_3d)

    gc.collect()

    return arr_1d

#----------------------------------------------
# define function to geodata frame
def get_gdf(array,flnme,reg):
    vec_flnme = os.path.join(path_to_put_intermediate_files,os.path.basename(flnme).replace('.tif',reg + '.shp'))
    array = array.astype(np.int16).copy()    
    gdffrme = array_to_vector(array, vec_flnme, integer_list, meta_crs.to_string(),meta_trns,)  
    return gdffrme

#----------------------------------------------
# Define a function to determine the position based on the data density
def find_best_position(ax, x_data, y_data):
    # Get the axis bounds
    x_bounds = ax.get_xlim()
    y_bounds = ax.get_ylim()
    
    # Define grid size and create bins within the axis bounds
    grid_size = 10
    x_bins = np.linspace(*x_bounds, grid_size)
    y_bins = np.linspace(*y_bounds, grid_size)
    
    # Create a 2D histogram of the data to find the density of points
    hist, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=(x_bins, y_bins))
    
    # Find the index of the bin with the lowest count
    min_density_idx = np.unravel_index(np.argmin(hist, axis=None), hist.shape)
    
    # Get the position for the text
    # We choose the upper edge of the bin with the lowest count
    x_pos = x_edges[min_density_idx[1]]
    y_pos = y_edges[min_density_idx[0]]
    
    # Adjust the position to be within the bounds and visible
    x_pos = max(min(x_pos, x_bounds[1] * 0.95), x_bounds[0] * 1.05)
    y_pos = max(min(y_pos, y_bounds[1] * 0.95), y_bounds[0] * 1.05)
    
    return x_pos, y_pos
#----------------------------------------------

# define function to do scatter plot
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
        
#-----------------------------------------------------------------------------------------
def count_metric_compute(extent_df,cl_elm,met_elm):
    
    met_df_lst = [] 
    prdc_lst,class_lst,cc_lst,kge_lst,rmse_lst,nrmse_lst,mape_lst,rb_lst = [],[],[],[],[],[],[],[]

    for m in met_elm:      

        prdc_to_ana,ana_elem = m[0],m[1]   

        for e in ana_elem:       

            sim_column,obs_column = e[0],e[1]

            clss = list(set(sim_column.split('_')).intersection(set(cl_elm)))[0]

            if clss == 'wtr':
                clss_cal = 'Water'
            elif clss == 'snfr':
                clss_cal = 'Snow free'
            elif clss == 'snc':
                clss_cal = 'Snow cover'
            elif clss == 'ice':
                clss_cal = 'Ice'

            sim_dat, obs_dat = extent_df[sim_column], extent_df[obs_column]

            cc_val = round(p_corr(obs_dat,sim_dat),2)
            kge_val = round(kge2012(obs_dat,sim_dat)[-1],2)
            rmse_val = round(rmsqe(obs_dat,sim_dat),2)
            rb_val = round(relative_bias(obs_dat,sim_dat)*100,2)
            nrmse_val = round(nrmsqe(obs_dat,sim_dat),2)
            mape_val = round(he.mape(sim_dat, obs_dat,remove_neg=True),2) # mean abs % error

            prdc_lst.append(prdc_to_ana)
            class_lst.append(clss_cal)
            cc_lst.append(cc_val)
            rmse_lst.append(rmse_val)
            rb_lst.append(rb_val)
            nrmse_lst.append(nrmse_val)
            kge_lst.append(kge_val)
            mape_lst.append(mape_val)

    extent_metrics_df = pd.DataFrame({'Class': class_lst, 'Product': prdc_lst, 
                                      'CC': cc_lst, 'KGE': kge_lst, 'RMSE': rmse_lst, 
                                      'RB':rb_lst, 'NRMSE': nrmse_lst,'MAPE': mape_lst})
    met_df_lst.append(extent_metrics_df)

    all_met_df = pd.concat(met_df_lst,axis=0)

    all_met_df.sort_values(by='Class',inplace=True,ascending=False)

    return all_met_df

#------------------------------------------------------------------
def cat_metrcs_computed(sym_lst,org_lst):
    cat_df_lst = []

    for s in sym_lst:

        cat_prdct,sim_lst = s[0],s[1]

        print(cat_prdct)

        cat_eval_lst_sim = [org_lst, sim_lst]

        cat_eval_df = cat_evaluate(cat_eval_lst_sim,cat_prdct)

        cat_df_lst.append(cat_eval_df)
        
    all_cat_df = pd.concat(cat_df_lst,axis=0)
    all_cat_df.sort_values(by='Class',ascending=False,inplace=True)
    return all_cat_df

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

#------------------------------------------------------------------------------------
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

#----------------------------------------------------------------------------------------
def process_file(file_path):
# for l in sorted(all_autosnow_files):   
#     file_path = l

    yr_DOY = os.path.basename(file_path).split('_')[4]

    yr  = int(os.path.basename(file_path).split('_')[4][:4])#int(l[:4])

    doy_ = os.path.basename(file_path).split('_')[4][-3:]

    doY = int(os.path.basename(file_path).split('_')[4][-3:]) #int(l[-3:])

    doy = os.path.basename(file_path).split('_')[4][-3:]

    dt = datetime(yr, 1, 1) + timedelta(doY - 1) # datetime.

    dt_sve = dt.strftime('%Y%m%d') 

    # Create a datetime object
    date_time = pd.to_datetime(f'{yr}-{doY}', format='%Y-%j')

    mnth = pd.to_datetime(dt_sve).month
    #------------------------------------------------------
    # read files
    # estimates 
    path_ = os.path.join(dir_of_diff_est_meth,'ML-E-approach_based_estimates')
    ml_e_estimated_arr = read_processed_files(path_, ['ML-E',yr_DOY,'0.1deg_wgs'],'.nc')
    
    path_ = os.path.join(dir_of_diff_est_meth,'ML-EC-approach_based_estimates')

    ml_ec_estimated_arr = read_processed_files(path_, ['ML-EC',yr_DOY,'0.1deg_wgs'],'.nc')
    
    path_ = os.path.join(dir_of_diff_est_meth,'ML-ECC-approach_based_estimates')
    
    ml_ecc_estimated_arr = read_processed_files(path_, ['ML-ECC',yr_DOY,'0.1deg_wgs'],'.nc')
    
    path_ = os.path.join(dir_of_diff_est_meth,'CLIM-approach_based_estimates') 
    nmeprt='CLIM-approach_estimate_based_on_1992_2022_Autosnow_clim_subsetted_by_airTemp'
    climatology_estimated_arr = read_processed_files(path_, [nmeprt, yr_DOY,'0.1deg_wgs'],'.nc')
    
    # the original autosnow data
    gmais_dat = xr.open_dataarray(file_path) 
    gmasi_dat_array = gmais_dat.data[0,:,:]
    gmasi_dat_array = np.where(gmasi_dat_array > 3,np.nan,gmasi_dat_array)
    y_shp,x_shp = gmasi_dat_array.shape[0],gmasi_dat_array.shape[1]    
        
    date_time_ = date_time    

    print(date_time_)       

    # calculate hit/miss in %
    get_percent_hitmiss(hit_miss_df, gmasi_dat_array, ml_e_estimated_arr, 'ML-E', date_time, integer_list)
    get_percent_hitmiss(hit_miss_df, gmasi_dat_array, ml_ec_estimated_arr, 'ML-EC', date_time, integer_list)
    get_percent_hitmiss(hit_miss_df, gmasi_dat_array, ml_ecc_estimated_arr, 'ML-ECC', date_time, integer_list)
    get_percent_hitmiss(hit_miss_df, gmasi_dat_array, climatology_estimated_arr, 'CLIM', date_time, integer_list)
      
    #------------------------------------------------------

    # do hit/miss in % per class
    df_hit_miss_per_class(hit_miss_df,gmasi_dat_array, ml_e_estimated_arr, date_time, 'ML-E')
    df_hit_miss_per_class(hit_miss_df,gmasi_dat_array, ml_ec_estimated_arr, date_time, 'ML-EC')
    df_hit_miss_per_class(hit_miss_df,gmasi_dat_array, ml_ecc_estimated_arr, date_time, 'ML-ECC')
    df_hit_miss_per_class(hit_miss_df,gmasi_dat_array, climatology_estimated_arr, date_time, 'CLIM')   

    #------------------------------------------------------

    # append estimates and original to baskets
    ml_e_glb_bskt.append(ml_e_estimated_arr)

    ml_ec_glb_bskt.append(ml_ec_estimated_arr)

    climatology_glb_bskt.append(climatology_estimated_arr)

    ml_ecc_glb_bskt.append(ml_ecc_estimated_arr)

    
    gmasi_glb_bskt.append(gmasi_dat_array) 
    gc.collect()
    #------------------------------------------------------
    # get regional 60 degree lat N data
    ml_e_arr_60_nh = ml_e_estimated_arr[0:nh_row60[0],:].astype(np.int16)

    ml_ec_arr_60_nh = ml_ec_estimated_arr[0:nh_row60[0],:].astype(np.int16)

    ml_ecc_arr_60_nh = ml_ecc_estimated_arr[0:nh_row60[0],:].astype(np.int16)

    clim_arr_60_nh = climatology_estimated_arr[0:nh_row60[0],:].astype(np.int16)

    gmasi_arr_60_nh = gmasi_dat_array[0:nh_row60[0],:].astype(np.int16)
    #------------------------------------------------------

    # segregate the data into different lists according NH, SH and seasons
    ml_e_arr_nh = ml_e_estimated_arr[0:nh_row[0],:].astype(np.int16)
    ml_e_nh.append(ml_e_arr_nh)

    ml_e_arr_sh = ml_e_estimated_arr[sh_row[0]:y_shp,:].astype(np.int16)
    ml_e_sh.append(ml_e_arr_sh)
    #-------------------

    gmasi_arr_nh = gmasi_dat_array[0:nh_row[0],:].astype(np.int16)
    gmasi_nh.append(gmasi_arr_nh)

    gmasi_arr_sh = gmasi_dat_array[sh_row[0]:y_shp,:].astype(np.int16)
    gmasi_sh.append(gmasi_arr_sh)
    #-------------------

    ml_ec_arr_nh = ml_ec_estimated_arr[0:nh_row[0],:].astype(np.int16)
    ml_ec_nh.append(ml_ec_arr_nh)

    ml_ec_arr_sh = ml_ec_estimated_arr[sh_row[0]:y_shp,:].astype(np.int16)
    ml_ec_sh.append(ml_ec_arr_sh)
    #-------------------

    climatology_arr_nh = climatology_estimated_arr[0:nh_row[0],:].astype(np.int16)
    climatology_nh.append(climatology_arr_nh)

    climatology_arr_sh = climatology_estimated_arr[sh_row[0]:y_shp,:].astype(np.int16)
    climatology_sh.append(climatology_arr_sh)

    #-------------------

    ml_ecc_arr_nh = ml_ecc_estimated_arr[0:nh_row[0],:].astype(np.int16)
    ml_ecc_nh.append(ml_ecc_arr_nh)

    ml_ecc_arr_sh = ml_ecc_estimated_arr[sh_row[0]:y_shp,:].astype(np.int16)
    ml_ecc_sh.append(ml_ecc_arr_sh)    
    
#------------------------------------------------------
    # do the extent comparison at the seasonal level
    # NH
    
    ml_e_cnt_nh = count_class_pixels(ml_e_arr_nh)
    ml_ec_cnt_nh = count_class_pixels(ml_ec_arr_nh)
    ml_ecc_cnt_nh = count_class_pixels(ml_ecc_arr_nh)
    climatology_cnt_nh = count_class_pixels(climatology_arr_nh)
    gmasi_cnt_nh = count_class_pixels(gmasi_arr_nh)
    # e_cnt_nh = count_class_pixels(e_arr_nh)    

    populate_df_count(nh_px_cnt, ml_e_cnt_nh, date_time, 'ML-E')
    populate_df_count(nh_px_cnt, ml_ec_cnt_nh, date_time, 'ML-EC')
    populate_df_count(nh_px_cnt, ml_ecc_cnt_nh, date_time, 'ML-ECC')
    populate_df_count(nh_px_cnt, climatology_cnt_nh, date_time, 'CLIM')
    populate_df_count(nh_px_cnt, gmasi_cnt_nh, date_time, 'GMASI')      

    #---------------------------------------------------
    # SH
    

    ml_e_cnt_sh = count_class_pixels(ml_e_arr_sh)
    ml_ec_cnt_sh = count_class_pixels(ml_ec_arr_sh)
    ml_ecc_cnt_sh = count_class_pixels(ml_ecc_arr_sh)
    climatology_cnt_sh = count_class_pixels(climatology_arr_sh)
    gmasi_cnt_sh = count_class_pixels(gmasi_arr_sh)
    
    populate_df_count(sh_px_cnt, ml_e_cnt_sh, date_time, 'ML-E')
    populate_df_count(sh_px_cnt, ml_ec_cnt_sh, date_time, 'ML-EC')
    populate_df_count(sh_px_cnt, ml_ecc_cnt_sh, date_time, 'ML-ECC')
    populate_df_count(sh_px_cnt, climatology_cnt_sh, date_time, 'CLIM')
    populate_df_count(sh_px_cnt, gmasi_cnt_sh, date_time, 'GMASI')       
    #----------------------------------------------------

    # append based on season and hemisphere
    season_nh = find_season(mnth,'Northern')
    if season_nh is 'Winter':        
        nh_winter_ml_e.append(ml_e_arr_nh) 
        nh_winter_gmasi.append(gmasi_arr_nh)      
        nh_winter_ml_ec.append(ml_ec_arr_nh)  
        nh_winter_climatology.append(climatology_arr_nh) 
        nh_winter_ml_ecc.append(ml_ecc_arr_nh)        

        ml_e_winter_cnt_nh = count_class_pixels(ml_e_arr_nh)
        ml_ec_winter_cnt_nh = count_class_pixels(ml_ec_arr_nh)
        ml_ecc_winter_cnt_nh = count_class_pixels(ml_ecc_arr_nh)
        climatology_winter_cnt_nh = count_class_pixels(climatology_arr_nh)
        gmasi_winter_cnt_nh = count_class_pixels(gmasi_arr_nh)

        populate_df_count(nh_winter_px_cnt, ml_e_winter_cnt_nh, date_time, 'ML-E')
        populate_df_count(nh_winter_px_cnt, ml_ec_winter_cnt_nh, date_time, 'ML-EC')
        populate_df_count(nh_winter_px_cnt, ml_ecc_winter_cnt_nh, date_time, 'ML-ECC')
        populate_df_count(nh_winter_px_cnt, climatology_winter_cnt_nh, date_time, 'CLIM')
        populate_df_count(nh_winter_px_cnt, gmasi_winter_cnt_nh, date_time, 'GMASI')              

        #----------------------------------------------------

    elif season_nh is 'Spring':        
        nh_spring_ml_e.append(ml_e_arr_nh)
        nh_spring_gmasi.append(gmasi_arr_nh)   
        nh_spring_ml_ec.append(ml_ec_arr_nh) 
        nh_spring_climatology.append(climatology_arr_nh) 
        nh_spring_ml_ecc.append(ml_ecc_arr_nh)        

        ml_e_spring_cnt_nh = count_class_pixels(ml_e_arr_nh)
        ml_ec_spring_cnt_nh = count_class_pixels(ml_ec_arr_nh)
        ml_ecc_spring_cnt_nh = count_class_pixels(ml_ecc_arr_nh)
        climatology_spring_cnt_nh = count_class_pixels(climatology_arr_nh)
        gmasi_spring_cnt_nh = count_class_pixels(gmasi_arr_nh)

        populate_df_count(nh_spring_px_cnt, ml_e_spring_cnt_nh, date_time, 'ML-E')
        populate_df_count(nh_spring_px_cnt, ml_ec_spring_cnt_nh, date_time, 'ML-EC')
        populate_df_count(nh_spring_px_cnt, ml_ecc_spring_cnt_nh, date_time, 'ML-ECC')
        populate_df_count(nh_spring_px_cnt, climatology_spring_cnt_nh, date_time, 'CLIM')
        populate_df_count(nh_spring_px_cnt, gmasi_spring_cnt_nh, date_time, 'GMASI')                      

        #----------------------------------------------------

    elif season_nh is 'Summer':
        nh_summer_ml_e.append(ml_e_arr_nh)
        nh_summer_gmasi.append(gmasi_arr_nh)
        nh_summer_ml_ec.append(ml_ec_arr_nh)
        nh_summer_climatology.append(climatology_arr_nh)
        nh_summer_ml_ecc.append(ml_ecc_arr_nh)        

        ml_e_summer_cnt_nh = count_class_pixels(ml_e_arr_nh)
        ml_ec_summer_cnt_nh = count_class_pixels(ml_ec_arr_nh)
        ml_ecc_summer_cnt_nh = count_class_pixels(ml_ecc_arr_nh)
        climatology_summer_cnt_nh = count_class_pixels(climatology_arr_nh)
        gmasi_summer_cnt_nh = count_class_pixels(gmasi_arr_nh)

        populate_df_count(nh_summer_px_cnt, ml_e_summer_cnt_nh, date_time, 'ML-E')
        populate_df_count(nh_summer_px_cnt, ml_ec_summer_cnt_nh, date_time, 'ML-EC')
        populate_df_count(nh_summer_px_cnt, ml_ecc_summer_cnt_nh, date_time, 'ML-ECC')
        populate_df_count(nh_summer_px_cnt, climatology_summer_cnt_nh, date_time, 'CLIM')
        populate_df_count(nh_summer_px_cnt, gmasi_summer_cnt_nh, date_time, 'GMASI')              

        #----------------------------------------------------

    elif season_nh is 'Autumn':
        nh_autumn_ml_e.append(ml_e_arr_nh)
        nh_autumn_gmasi.append(gmasi_arr_nh)
        nh_autumn_ml_ec.append(ml_ec_arr_nh)
        nh_autumn_climatology.append(climatology_arr_nh)
        nh_autumn_ml_ecc.append(ml_ecc_arr_nh)
        
        ml_e_autumn_cnt_nh = count_class_pixels(ml_e_arr_nh)
        ml_ec_autumn_cnt_nh = count_class_pixels(ml_ec_arr_nh)
        ml_ecc_autumn_cnt_nh = count_class_pixels(ml_ecc_arr_nh)
        climatology_autumn_cnt_nh = count_class_pixels(climatology_arr_nh)
        gmasi_autumn_cnt_nh = count_class_pixels(gmasi_arr_nh) 

        populate_df_count(nh_autumn_px_cnt, ml_e_autumn_cnt_nh, date_time, 'ML-E')
        populate_df_count(nh_autumn_px_cnt, ml_ec_autumn_cnt_nh, date_time, 'ML-EC')
        populate_df_count(nh_autumn_px_cnt, ml_ecc_autumn_cnt_nh, date_time, 'ML-ECC')
        populate_df_count(nh_autumn_px_cnt, climatology_autumn_cnt_nh, date_time, 'CLIM')
        populate_df_count(nh_autumn_px_cnt, gmasi_autumn_cnt_nh, date_time, 'GMASI')   

        #----------------------------------------------------

    season_sh = find_season(mnth,'Southern')
    if season_sh is 'Winter':
        sh_winter_ml_e.append(ml_e_arr_sh)
        sh_winter_gmasi.append(gmasi_arr_sh)
        sh_winter_ml_ec.append(ml_ec_arr_sh)
        sh_winter_climatology.append(climatology_arr_sh)
        sh_winter_ml_ecc.append(ml_ecc_arr_sh)
        
        ml_e_winter_cnt_sh = count_class_pixels(ml_e_arr_sh)
        ml_ec_winter_cnt_sh = count_class_pixels(ml_ec_arr_sh)
        ml_ecc_winter_cnt_sh = count_class_pixels(ml_ecc_arr_sh)
        climatology_winter_cnt_sh = count_class_pixels(climatology_arr_sh)
        gmasi_winter_cnt_sh = count_class_pixels(gmasi_arr_sh)

        populate_df_count(sh_winter_px_cnt, ml_e_winter_cnt_sh, date_time, 'ML-E')
        populate_df_count(sh_winter_px_cnt, ml_ec_winter_cnt_sh, date_time, 'ML-EC')
        populate_df_count(sh_winter_px_cnt, ml_ecc_winter_cnt_sh, date_time, 'ML-ECC')
        populate_df_count(sh_winter_px_cnt, climatology_winter_cnt_sh, date_time, 'CLIM')
        populate_df_count(sh_winter_px_cnt, gmasi_winter_cnt_sh, date_time, 'GMASI')        

        #----------------------------------------------------

    elif season_sh is 'Spring':
        sh_spring_ml_e.append(ml_e_arr_sh)
        sh_spring_gmasi.append(gmasi_arr_sh)
        sh_spring_ml_ec.append(ml_ec_arr_sh)
        sh_spring_climatology.append(climatology_arr_sh)
        sh_spring_ml_ecc.append(ml_ecc_arr_sh)
                
        ml_e_spring_cnt_sh = count_class_pixels(ml_e_arr_sh)
        ml_ec_spring_cnt_sh = count_class_pixels(ml_ec_arr_sh)
        ml_ecc_spring_cnt_sh = count_class_pixels(ml_ecc_arr_sh)
        climatology_spring_cnt_sh = count_class_pixels(climatology_arr_sh)
        gmasi_spring_cnt_sh = count_class_pixels(gmasi_arr_sh)

        populate_df_count(sh_spring_px_cnt, ml_e_spring_cnt_sh, date_time, 'ML-E')
        populate_df_count(sh_spring_px_cnt, ml_ec_spring_cnt_sh, date_time, 'ML-EC')
        populate_df_count(sh_spring_px_cnt, ml_ecc_spring_cnt_sh, date_time, 'ML-ECC')
        populate_df_count(sh_spring_px_cnt, climatology_spring_cnt_sh, date_time, 'CLIM')
        populate_df_count(sh_spring_px_cnt, gmasi_spring_cnt_sh, date_time, 'GMASI') 

        #----------------------------------------------------

    elif season_sh is 'Summer':
        sh_summer_ml_e.append(ml_e_arr_sh)
        sh_summer_gmasi.append(gmasi_arr_sh)  
        sh_summer_climatology.append(climatology_arr_sh)    
        sh_summer_ml_ec.append(ml_ec_arr_sh)  
        sh_summer_ml_ecc.append(ml_ecc_arr_sh) 
        
        ml_e_summer_cnt_sh = count_class_pixels(ml_e_arr_sh)
        ml_ec_summer_cnt_sh = count_class_pixels(ml_ec_arr_sh)
        ml_ecc_summer_cnt_sh = count_class_pixels(ml_ecc_arr_sh)
        climatology_summer_cnt_sh = count_class_pixels(climatology_arr_sh)
        gmasi_summer_cnt_sh = count_class_pixels(gmasi_arr_sh)        
        
        populate_df_count(sh_summer_px_cnt, ml_e_summer_cnt_sh, date_time, 'ML-E')
        populate_df_count(sh_summer_px_cnt, ml_ec_summer_cnt_sh, date_time, 'ML-EC')
        populate_df_count(sh_summer_px_cnt, ml_ecc_summer_cnt_sh, date_time, 'ML-ECC')
        populate_df_count(sh_summer_px_cnt, climatology_summer_cnt_sh, date_time, 'CLIM')
        populate_df_count(sh_summer_px_cnt, gmasi_summer_cnt_sh, date_time, 'GMASI')     

        #----------------------------------------------------
    elif season_sh is 'Autumn':
        sh_autumn_ml_e.append(ml_e_arr_sh)
        sh_autumn_gmasi.append(gmasi_arr_sh)
        sh_autumn_ml_ec.append(ml_ec_arr_sh) 
        sh_autumn_climatology.append(climatology_arr_sh) 
        sh_autumn_ml_ecc.append(ml_ecc_arr_sh)
        
        ml_e_autumn_cnt_sh = count_class_pixels(ml_e_arr_sh)
        ml_ec_autumn_cnt_sh = count_class_pixels(ml_ec_arr_sh)
        ml_ecc_autumn_cnt_sh = count_class_pixels(ml_ecc_arr_sh)
        climatology_autumn_cnt_sh = count_class_pixels(climatology_arr_sh)
        gmasi_autumn_cnt_sh = count_class_pixels(gmasi_arr_sh)
        
        populate_df_count(sh_autumn_px_cnt, ml_e_autumn_cnt_sh, date_time, 'ML-E')
        populate_df_count(sh_autumn_px_cnt, ml_ec_autumn_cnt_sh, date_time, 'ML-EC')
        populate_df_count(sh_autumn_px_cnt, ml_ecc_autumn_cnt_sh, date_time, 'ML-ECC')
        populate_df_count(sh_autumn_px_cnt, climatology_autumn_cnt_sh, date_time, 'CLIM')
        populate_df_count(sh_autumn_px_cnt, gmasi_autumn_cnt_sh, date_time, 'GMASI')
       
    #------------------------------------------------------
    
    # compare the extent (km^2) per autosnow class estimated by the model and original
        # RF ERA5 based estimates yr_DOY

    calcualte_total_area(area_extent, ml_e_estimated_arr, path_to_put_intermediate_files, 
                         yr_DOY, date_time, 'ML-E', integer_list, meta_crs, meta_trns)
    
    calcualte_total_area(area_extent, ml_ec_estimated_arr, path_to_put_intermediate_files, 
                         yr_DOY, date_time, 'ML-EC', integer_list, meta_crs, meta_trns)
    
    calcualte_total_area(area_extent, ml_ecc_estimated_arr, path_to_put_intermediate_files, 
                         yr_DOY, date_time, 'ML-ECC', integer_list, meta_crs, meta_trns)
    
    calcualte_total_area(area_extent, climatology_estimated_arr, path_to_put_intermediate_files, 
                         yr_DOY, date_time, 'CLIM', integer_list, meta_crs, meta_trns)
    
    calcualte_total_area(area_extent, gmasi_dat_array, path_to_put_intermediate_files, 
                         yr_DOY, date_time, 'GMASI', integer_list, meta_crs, meta_trns)
    
    #------------------------------------------------------
    # calculate area for regional analysis 60N
    calcualte_total_area(area_ext_reg_60n,ml_e_arr_60_nh, path_to_put_intermediate_files,
                         yr_DOY, date_time, 'ML-E', integer_list, meta_crs, meta_trns)
    
    calcualte_total_area(area_ext_reg_60n,ml_ec_arr_60_nh, path_to_put_intermediate_files,
                         yr_DOY, date_time, 'ML-EC', integer_list, meta_crs, meta_trns)
    
    calcualte_total_area(area_ext_reg_60n,ml_ecc_arr_60_nh, path_to_put_intermediate_files,
                         yr_DOY, date_time, 'ML-ECC', integer_list, meta_crs, meta_trns)
    
    calcualte_total_area(area_ext_reg_60n,clim_arr_60_nh, path_to_put_intermediate_files,
                         yr_DOY, date_time, 'CLIM', integer_list, meta_crs, meta_trns)
    
    calcualte_total_area(area_ext_reg_60n,gmasi_arr_60_nh, path_to_put_intermediate_files,
                         yr_DOY, date_time, 'GMASI', integer_list, meta_crs, meta_trns)
    #------------------------------------------------------

    ml_e_cnt = count_class_pixels(ml_e_estimated_arr.astype(np.int16)) 
    ml_ec_cnt = count_class_pixels(ml_ec_estimated_arr.astype(np.int16)) 
    ml_ecc_cnt = count_class_pixels(ml_ecc_estimated_arr.astype(np.int16))
    clim_cnt = count_class_pixels(climatology_estimated_arr.astype(np.int16))
    gmasi_cnt = count_class_pixels(gmasi_dat_array.astype(np.int16))

    populate_df_count(px_cnt, ml_e_cnt, date_time, 'ML-E')
    populate_df_count(px_cnt, ml_ec_cnt, date_time, 'ML-EC')
    populate_df_count(px_cnt, ml_ecc_cnt, date_time, 'ML-ECC')
    populate_df_count(px_cnt, clim_cnt, date_time, 'CLIM')
    populate_df_count(px_cnt, gmasi_cnt, date_time, 'GMASI')          

    # if count % 100 == 0:
    #     print(str(count) + ' at ' + str(date_time_))  

#-----------------------------------------------------------
def parallel_process_files_(file_paths):
    results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        future_to_file = {executor.submit(process_file, file_path): file_path for file_path in file_paths}
        
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)  # Collect results for aggregation
            except Exception as exc:
                print(f'{file_path} generated an exception: {exc}')
    
    return results

def parallel_process_files(file_paths, max_workers=20):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    """
    Parallel process a list of file paths using ProcessPoolExecutor.
    
    Parameters:
    - file_paths (list): List of file paths to process.
    - max_workers (int): Number of worker processes to spawn.

    Returns:
    - results (list): List of results from processing each file.
    """
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, file_path): file_path for file_path in file_paths}
        
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)  # Collect results for aggregation
            except Exception as exc:
                print(f'{file_path} generated an exception: {exc}')
    
    return results

#%%
# area_ext_reg_60n_cpy = area_ext_reg_60n.copy()

print('begin reading data and compiling evaluation data')
ml_e_glb_bskt = []
ml_ec_glb_bskt = []
# e_glb_bskt = []
climatology_glb_bskt = []
gmasi_glb_bskt = []
ml_ecc_glb_bskt = []
#------------------------------

ml_e_nh, ml_e_sh = [], []

ml_ec_nh, ml_ec_sh = [],[]
climatology_nh, climatology_sh = [],[]
# e_nh, e_sh = [],[]
gmasi_nh, gmasi_sh = [], []

ml_ecc_nh, ml_ecc_sh = [],[]
#------------------------------
nh_winter_ml_e, nh_spring_ml_e, nh_summer_ml_e, nh_autumn_ml_e = [], [], [], []
sh_winter_ml_e, sh_spring_ml_e, sh_summer_ml_e, sh_autumn_ml_e = [], [], [], []

nh_winter_gmasi, nh_spring_gmasi, nh_summer_gmasi, nh_autumn_gmasi = [], [], [], []
sh_winter_gmasi, sh_spring_gmasi, sh_summer_gmasi, sh_autumn_gmasi = [], [], [], []

nh_winter_ml_ec, nh_spring_ml_ec, nh_summer_ml_ec, nh_autumn_ml_ec = [], [], [], []
sh_winter_ml_ec, sh_spring_ml_ec, sh_summer_ml_ec, sh_autumn_ml_ec = [], [], [], []

nh_winter_climatology, nh_spring_climatology, nh_summer_climatology, nh_autumn_climatology = [], [], [], []
sh_winter_climatology, sh_spring_climatology, sh_summer_climatology, sh_autumn_climatology = [], [], [], []

nh_winter_ml_ecc, nh_spring_ml_ecc, nh_summer_ml_ecc, nh_autumn_ml_ecc = [], [], [], []
sh_winter_ml_ecc, sh_spring_ml_ecc, sh_summer_ml_ecc, sh_autumn_ml_ecc = [], [], [], []

# nh_winter_e, nh_spring_e, nh_summer_e, nh_autumn_e = [], [], [], []
# sh_winter_e, sh_spring_e, sh_summer_e, sh_autumn_e = [], [], [], []

#---------------------------------------------------------------------
count = 0
# area_ext_reg_60n_cpy = area_ext_reg_60n.copy()

hit_miss_df = pd.DataFrame()
area_extent = pd.DataFrame()
area_ext_reg_60n = pd.DataFrame()
px_cnt = pd.DataFrame()
nh_px_cnt = pd.DataFrame()
sh_px_cnt = pd.DataFrame()

nh_winter_px_cnt = pd.DataFrame()
nh_spring_px_cnt = pd.DataFrame()
nh_summer_px_cnt = pd.DataFrame()
nh_autumn_px_cnt = pd.DataFrame()

sh_winter_px_cnt = pd.DataFrame()
sh_spring_px_cnt = pd.DataFrame()
sh_summer_px_cnt = pd.DataFrame()
sh_autumn_px_cnt = pd.DataFrame()

print('Main parallel execution has begun')
files = sorted(all_autosnow_files) # List of file paths
aggregated_results = parallel_process_files(files)
print('done!')

svnem_csv = '_'.join(['daily_percent_mismatch_analysis',cde_run_dte])+ '.csv'
hit_miss_df.to_csv(os.path.join(path_to_put_df,svnem_csv))
#%%
# print('begin evaluation')    
# # the differences in the total pixel count per class classified by the different models and origina (GMASI) data
# px_cnt['ml_e_gmasi_wtr_diff'] = px_cnt['ml_e_wtr_px_cnt'] - px_cnt['gmasi_wtr_px_cnt']# ml_e_wtr_px_cnt

# px_cnt['ml_e_gmasi_snfr_diff'] = px_cnt['ml_e_snfr_px_cnt'] - px_cnt['gmasi_snfr_px_cnt']

# px_cnt['ml_e_gmasi_snc_diff'] = px_cnt['ml_e_snc_px_cnt'] - px_cnt['gmasi_snc_px_cnt']

# px_cnt['ml_e_gmasi_ice_diff'] = px_cnt['ml_e_ice_px_cnt'] - px_cnt['gmasi_ice_px_cnt']

# #------------------------------------------------------
# px_cnt['ml_ec_gmasi_wtr_diff'] = px_cnt['ml_ec_wtr_px_cnt'] - px_cnt['gmasi_wtr_px_cnt']

# px_cnt['ml_ec_gmasi_snfr_diff'] = px_cnt['ml_ec_snfr_px_cnt'] - px_cnt['gmasi_snfr_px_cnt']

# px_cnt['ml_ec_gmasi_snc_diff'] = px_cnt['ml_ec_snc_px_cnt'] - px_cnt['gmasi_snc_px_cnt']

# px_cnt['ml_ec_gmasi_ice_diff'] = px_cnt['ml_ec_ice_px_cnt'] - px_cnt['gmasi_ice_px_cnt']

# #------------------------------------------------------
# px_cnt['climatology_gmasi_wtr_diff'] = px_cnt['climatology_wtr_px_cnt'] - px_cnt['gmasi_wtr_px_cnt']

# px_cnt['climatology_gmasi_snfr_diff'] = px_cnt['climatology_snfr_px_cnt'] - px_cnt['gmasi_snfr_px_cnt']

# px_cnt['climatology_gmasi_snc_diff'] = px_cnt['climatology_snc_px_cnt'] - px_cnt['gmasi_snc_px_cnt']

# px_cnt['climatology_gmasi_ice_diff'] = px_cnt['climatology_ice_px_cnt'] - px_cnt['gmasi_ice_px_cnt']

# #------------------------------------------------------
# px_cnt['ml_ecc_gmasi_wtr_diff'] = px_cnt['ml_ecc_wtr_px_cnt'] - px_cnt['gmasi_wtr_px_cnt']

# px_cnt['ml_ecc_gmasi_snfr_diff'] = px_cnt['ml_ecc_snfr_px_cnt'] - px_cnt['gmasi_snfr_px_cnt']

# px_cnt['ml_ecc_gmasi_snc_diff'] = px_cnt['ml_ecc_snc_px_cnt'] - px_cnt['gmasi_snc_px_cnt']

# px_cnt['ml_ecc_gmasi_ice_diff'] = px_cnt['ml_ecc_ice_px_cnt'] - px_cnt['gmasi_ice_px_cnt']

#------------------------------------------------------
# px_cnt['e_gmasi_wtr_diff'] = px_cnt['e_wtr_px_cnt'] - px_cnt['gmasi_wtr_px_cnt']

# px_cnt['e_gmasi_snfr_diff'] = px_cnt['e_snfr_px_cnt'] - px_cnt['gmasi_snfr_px_cnt']

# px_cnt['e_gmasi_snc_diff'] = px_cnt['e_snc_px_cnt'] - px_cnt['gmasi_snc_px_cnt']

# px_cnt['e_gmasi_ice_diff'] = px_cnt['e_ice_px_cnt'] - px_cnt['gmasi_ice_px_cnt']

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the differences in area extent (per class) by the different model and GMASI duirng the vaidation period


area_extent_computed = get_area_extent_diffs(area_extent)

area_extent_computed_60n = get_area_extent_diffs(area_ext_reg_60n)

# area_extent['ml_e_gmasi_wtr_diff'] = ((area_extent['ml_e_wtr_total_area'] - area_extent['gmasi_wtr_total_area'])/area_extent['gmasi_wtr_total_area'])*100

# area_extent['ml_e_gmasi_snfr_diff'] = ((area_extent['ml_e_snfr_total_area'] - area_extent['gmasi_snfr_total_area'])/area_extent['gmasi_snfr_total_area'])*100

# area_extent['ml_e_gmasi_snc_diff'] = ((area_extent['ml_e_snc_total_area'] - area_extent['gmasi_snc_total_area'])/area_extent['gmasi_snc_total_area'])*100

# area_extent['ml_e_gmasi_ice_diff'] = ((area_extent['ml_e_ice_total_area'] - area_extent['gmasi_ice_total_area'])/area_extent['gmasi_ice_total_area'])*100

# #------------------------------------------------------
# area_extent['ml_ec_gmasi_wtr_diff'] = ((area_extent['ml_ec_wtr_total_area'] - area_extent['gmasi_wtr_total_area'])/area_extent['gmasi_wtr_total_area'])*100

# area_extent['ml_ec_gmasi_snfr_diff'] = ((area_extent['ml_ec_snfr_total_area'] - area_extent['gmasi_snfr_total_area'])/area_extent['gmasi_snfr_total_area'])*100

# area_extent['ml_ec_gmasi_snc_diff'] = ((area_extent['ml_ec_snc_total_area'] - area_extent['gmasi_snc_total_area'])/area_extent['gmasi_snc_total_area'])*100

# area_extent['ml_ec_gmasi_ice_diff'] = ((area_extent['ml_ec_ice_total_area'] - area_extent['gmasi_ice_total_area'])/area_extent['gmasi_ice_total_area'])*100

# #------------------------------------------------------
# area_extent['climatology_gmasi_wtr_diff'] = ((area_extent['climatology_wtr_total_area'] - area_extent['gmasi_wtr_total_area'])/area_extent['gmasi_wtr_total_area'])*100

# area_extent['climatology_gmasi_snfr_diff'] = ((area_extent['climatology_snfr_total_area'] - area_extent['gmasi_snfr_total_area'])/area_extent['gmasi_snfr_total_area'])*100

# area_extent['climatology_gmasi_snc_diff'] = ((area_extent['climatology_snc_total_area'] - area_extent['gmasi_snc_total_area'])/area_extent['gmasi_snc_total_area'])*100

# area_extent['climatology_gmasi_ice_diff'] = ((area_extent['climatology_ice_total_area'] - area_extent['gmasi_ice_total_area'])/area_extent['gmasi_ice_total_area'])*100

# #------------------------------------------------------
# area_extent['ml_ecc_gmasi_wtr_diff'] = ((area_extent['ml_ecc_wtr_total_area'] - area_extent['gmasi_wtr_total_area'])/area_extent['gmasi_wtr_total_area'])*100

# area_extent['ml_ecc_gmasi_snfr_diff'] = ((area_extent['ml_ecc_snfr_total_area'] - area_extent['gmasi_snfr_total_area'])/area_extent['gmasi_snfr_total_area'])*100

# area_extent['ml_ecc_gmasi_snc_diff'] = ((area_extent['ml_ecc_snc_total_area'] - area_extent['gmasi_snc_total_area'])/area_extent['gmasi_snc_total_area'])*100

# area_extent['ml_ecc_gmasi_ice_diff'] = ((area_extent['ml_ecc_ice_total_area'] - area_extent['gmasi_ice_total_area'])/area_extent['gmasi_ice_total_area'])*100

#------------------------------------------------------
# area_extent['e_gmasi_wtr_diff'] = ((area_extent['e_wtr_total_area'] - area_extent['gmasi_wtr_total_area'])/area_extent['gmasi_wtr_total_area'])*100

# area_extent['e_gmasi_snfr_diff'] = ((area_extent['e_snfr_total_area'] - area_extent['gmasi_snfr_total_area'])/area_extent['gmasi_snfr_total_area'])*100

# area_extent['e_gmasi_snc_diff'] = ((area_extent['e_snc_total_area'] - area_extent['gmasi_snc_total_area'])/area_extent['gmasi_snc_total_area'])*100

# area_extent['e_gmasi_ice_diff'] = ((area_extent['e_ice_total_area'] - area_extent['gmasi_ice_total_area'])/area_extent['gmasi_ice_total_area'])*100

print('running disk management')
# remove intermediate files    
[os.remove(os.path.join(path_to_put_intermediate_files,x)) for x in os.listdir(path_to_put_intermediate_files) \
                        if any(x.startswith(rm) for rm in remove_elem)]

[os.remove(os.path.join(path_to_clim_only_autosnw_estimated,x)) for x in os.listdir(path_to_clim_only_autosnw_estimated) \
                        if any(x.endswith(rm) for rm in ['cpg','dbf','shp','prj','shx'])]

[os.remove(os.path.join(path_to_estimated_autosnw,x)) for x in os.listdir(path_to_estimated_autosnw) \
                                if any(x.endswith(rm) for rm in ['cpg','dbf','shp','prj','shx'])]

# svnem_csv = '_'.join(['area_extent_analysis',cde_run_dte])+ '.csv'
# area_extent.to_csv(os.path.join(path_to_put_df,svnem_csv))

# svnem_csv = '_'.join(['grid_population_analysis',cde_run_dte])+ '.csv'
# px_cnt.to_csv(os.path.join(path_to_put_df,svnem_csv))
#%%
print('*************** begin match/missmatch calculations ***********************')

# do pixel-wise hit/miss calculate
# non seasonal computation
# nh_hits_2d_ml_e, nh_miss_2d_ml_e = return_px_based_hit_miss_percent(ml_e_nh, gmasi_nh,)

# nh_hits_2d_ml_ec, nh_miss_2d_ml_ec = return_px_based_hit_miss_percent(ml_ec_nh, gmasi_nh)

# nh_hits_2d_ml_ecc, nh_miss_2d_ml_ecc = return_px_based_hit_miss_percent(ml_ecc_nh,gmasi_nh)

# nh_hits_2d_climatology, nh_miss_2d_climatology = return_px_based_hit_miss_percent(climatology_nh, gmasi_nh)
# #-------------------------------------

# sh_hits_2d_ml_e, sh_miss_2d_ml_e = return_px_based_hit_miss_percent(ml_e_sh, gmasi_sh,)

# sh_hits_2d_ml_ec, sh_miss_2d_ml_ec = return_px_based_hit_miss_percent(ml_ec_sh, gmasi_sh)

# sh_hits_2d_ml_ecc, sh_miss_2d_ml_ecc = return_px_based_hit_miss_percent(ml_ecc_sh, gmasi_sh)

# sh_hits_2d_climatology, sh_miss_2d_climatology = return_px_based_hit_miss_percent(climatology_sh,gmasi_sh)

# non_seas_arr2_plt = [('ML-E',nh_miss_2d_ml_e,sh_miss_2d_ml_e), 
#                      ('ML-EC',nh_miss_2d_ml_ec, sh_miss_2d_ml_ec),
#                      ('ML-ECC',nh_miss_2d_ml_ecc, sh_miss_2d_ml_ecc),               
#                      ('CLIM',nh_miss_2d_climatology, sh_miss_2d_climatology)]

# plot_hemisphere_comparison(non_seas_arr2_plt)
# svenme = '_'.join(['percent_hit_miss_non_seasonal',cde_run_dte]) + '.png'
# plt.savefig(os.path.join(path_to_put_plots,svenme),bbox_inches='tight')
# plt.close()

#--------------------------------------------------------------------------------------------
# NH 
# nh_summer_hits_2d_ml_e, nh_summer_miss_2d_ml_e = return_px_based_hit_miss_percent(nh_summer_ml_e, nh_summer_gmasi,)

# nh_summer_hits_2d_ml_ec, nh_summer_miss_2d_ml_ec = return_px_based_hit_miss_percent(nh_summer_ml_ec, nh_summer_gmasi)

# nh_summer_hits_2d_ml_ecc, nh_summer_miss_2d_ml_ecc = return_px_based_hit_miss_percent(nh_summer_ml_ecc, nh_summer_gmasi)

# nh_summer_hits_2d_climatology, nh_summer_miss_2d_climatology = return_px_based_hit_miss_percent(nh_summer_climatology,
#                                                                                               nh_summer_gmasi)

# nh_winter_hits_2d_ml_e, nh_winter_miss_2d_ml_e = return_px_based_hit_miss_percent(nh_winter_ml_e, nh_winter_gmasi,)

# nh_winter_hits_2d_ml_ec, nh_winter_miss_2d_ml_ec = return_px_based_hit_miss_percent(nh_winter_ml_ec, nh_winter_gmasi)

# nh_winter_hits_2d_ml_ecc, nh_winter_miss_2d_ml_ecc = return_px_based_hit_miss_percent(nh_winter_ml_ecc, nh_winter_gmasi)

# nh_winter_hits_2d_climatology, nh_winter_miss_2d_climatology = return_px_based_hit_miss_percent(nh_winter_climatology,
#                                                                                                 nh_winter_gmasi)

# nh_arr2_plt = [('ML-E',nh_summer_miss_2d_ml_e, nh_winter_miss_2d_ml_e), 
#                ('ML-EC',nh_summer_miss_2d_ml_ec, nh_winter_miss_2d_ml_ec),
#                ('ML-ECC', nh_summer_miss_2d_ml_ecc, nh_winter_miss_2d_ml_ecc),                      
#                ('CLIM',nh_summer_miss_2d_climatology, nh_winter_miss_2d_climatology)]

# plot_hemisphere_comparison(nh_arr2_plt,'NH')

# svenme = '_'.join(['nh_percent_hit_miss',cde_run_dte]) + '.png'
# plt.savefig(os.path.join(path_to_put_plots,svenme),bbox_inches='tight')
# plt.close()

#--------------------------------------------------------------------------------------------

# SH 
# sh_summer_hits_2d_ml_e, sh_summer_miss_2d_ml_e = return_px_based_hit_miss_percent(sh_summer_ml_e, sh_summer_gmasi,)

# sh_summer_hits_2d_ml_ec, sh_summer_miss_2d_ml_ec = return_px_based_hit_miss_percent(sh_summer_ml_ec, sh_summer_gmasi)

# sh_summer_hits_2d_ml_ecc, sh_summer_miss_2d_ml_ecc = return_px_based_hit_miss_percent(sh_summer_ml_ecc, sh_summer_gmasi)

# sh_summer_hits_2d_climatology, sh_summer_miss_2d_climatology = return_px_based_hit_miss_percent(sh_summer_climatology,
#                                                                                               sh_summer_gmasi)
#-------------------------------------
# SH
# sh_winter_hits_2d_ml_e, sh_winter_miss_2d_ml_e = return_px_based_hit_miss_percent(sh_winter_ml_e,sh_winter_gmasi,)

# sh_winter_hits_2d_ml_ec, sh_winter_miss_2d_ml_ec = return_px_based_hit_miss_percent(sh_winter_ml_ec,sh_winter_gmasi)

# sh_winter_hits_2d_ml_ecc, sh_winter_miss_2d_ml_ecc = return_px_based_hit_miss_percent(sh_winter_ml_ecc, sh_winter_gmasi)

# sh_winter_hits_2d_climatology, sh_winter_miss_2d_climatology = return_px_based_hit_miss_percent(sh_winter_climatology,
#                                                                                                 sh_winter_gmasi)

# sh_arr2_plt = [('ML-E',sh_summer_miss_2d_ml_e, sh_winter_miss_2d_ml_e), 
#                    ('ML-EC',sh_summer_miss_2d_ml_ec, sh_winter_miss_2d_ml_ec),
#                    ('ML-ECC',sh_summer_miss_2d_ml_ecc, sh_winter_miss_2d_ml_ecc),                    
#                    ('CLIM',sh_summer_miss_2d_climatology, sh_winter_miss_2d_climatology)]

# plot_hemisphere_comparison(sh_arr2_plt,'SH')
# svenme = '_'.join(['sh_percent_hit_miss',cde_run_dte]) + '.png'
# plt.savefig(os.path.join(path_to_put_plots,svenme),bbox_inches='tight')
# plt.close()

# print('done with hemispheric pixel wise percent mismatch/disagreement computations')
#%%

# resample the differences over a month and year
col2resample = ['ml_e_gmasi_wtr_diff', 'ml_e_gmasi_snfr_diff', 'ml_e_gmasi_snc_diff', 'ml_e_gmasi_ice_diff',
                'ml_ec_gmasi_wtr_diff', 'ml_ec_gmasi_snfr_diff', 'ml_ec_gmasi_snc_diff','ml_ec_gmasi_ice_diff',
                'climatology_gmasi_wtr_diff', 'climatology_gmasi_snfr_diff', 'climatology_gmasi_snc_diff', 'climatology_gmasi_ice_diff', 
                'ml_ecc_gmasi_wtr_diff', 'ml_ecc_gmasi_snfr_diff', 'ml_ecc_gmasi_snc_diff','ml_ecc_gmasi_ice_diff',
                ] 
# 'e_gmasi_wtr_diff', 'e_gmasi_snfr_diff', 'e_gmasi_snc_diff','e_gmasi_ice_diff',


filter_items = [clmn for clmn in area_extent.columns if 'diff' in clmn]
['ml_e_gmasi_wtr_diff','ml_ec_gmasi_wtr_diff', 'ml_ecc_gmasi_wtr_diff', 'climatology_gmasi_wtr_diff', 
               'ml_e_gmasi_snfr_diff','ml_ec_gmasi_snfr_diff','ml_ecc_gmasi_snfr_diff','climatology_gmasi_snfr_diff', 
               'ml_e_gmasi_snc_diff','ml_ec_gmasi_snc_diff','ml_ecc_gmasi_snc_diff','climatology_gmasi_snc_diff', 
               'ml_e_gmasi_ice_diff','ml_ec_gmasi_ice_diff','ml_ecc_gmasi_ice_diff','climatology_gmasi_ice_diff', ]
# 'e_gmasi_wtr_diff', 'e_gmasi_snfr_diff', 'e_gmasi_snc_diff', 'e_gmasi_ice_diff'
#--------------------------------------------------------------------------------------------

area_extent_diff_mnth = area_extent[col2resample].resample('M').mean()

area_extent_diff_yr = area_extent[col2resample].resample('Y').mean()

# calculate the average error per class per year
# area_extent_anom_yearly_avg = area_extent.groupby(area_extent.index.year)[col2resample].mean() # 
# area_extent_anom_yearly_avg = pd.DataFrame(area_extent_anom_yearly_avg.filter(items=filter_items))
# area_extent_anom_yearly_avg = area_extent_anom_yearly_avg.transpose()
# Table 5
area_extent_anom_yearly_avg = pd.DataFrame(area_extent_computed[filter_items].mean(),
                                           columns=['mean'])
svnem_csv = '_'.join(['average_errors_in_land_cover_extent_during_val_period',cde_run_dte])+ '.csv'
area_extent_anom_yearly_avg.to_csv(os.path.join(path_to_put_df,svnem_csv))

svnem_csv = '_'.join(['average_errors_in_land_cover_extent_during_val_by_season',cde_run_dte])+ '.csv'
area_extent_computed['month'] = area_extent_computed.index.month
area_extent_computed['season'] = area_extent_computed['month'].apply(get_season)
area_extent_seasonal_mean = area_extent.groupby('season')[filter_items].mean()
area_extent_seasonal_mean.to_csv(os.path.join(path_to_put_df,svnem_csv))

# Table S2
svnem_csv = '_'.join(['average_errors_in_land_cover_extent_during_val_by_year_60N',cde_run_dte])+ '.csv'
area_ext_60n_anom_yearl_avg = pd.DataFrame(area_extent_computed_60n[filter_items].mean()
                                           ,columns=['mean'])

svnem_csv = '_'.join(['average_errors_in_land_cover_extent_during_val_by_season_60N',cde_run_dte])+ '.csv'
area_ext_reg_60n['month'] = area_extent_computed_60n.index.month
area_ext_reg_60n['season'] = area_extent_computed_60n['month'].apply(get_season)
area_ext_reg_60n_seasonal_mean = area_extent_computed_60n.groupby('season')[filter_items].mean()
area_ext_reg_60n_seasonal_mean.to_csv(os.path.join(path_to_put_df,svnem_csv))

#----------------------------------------------------------------------------------------
print('make time series plot of area extent - Figure 8')

# Call the function with your data
# Figure 8
# plot_time_series(area_extent)

plot_time_series_4x1(area_extent_computed)
svenme = '_'.join(['percentage_bias_in_extent_anomaly_global',cde_run_dte]) + '.png'
plt.savefig(os.path.join(path_to_put_plots,svenme),bbox_inches='tight')
plt.close()

plot_time_series_4x1(area_extent_computed_60n)
svenme = '_'.join(['percentage_bias_in_extent_anomaly_60N',cde_run_dte]) + '.png'
plt.savefig(os.path.join(path_to_put_plots,svenme),bbox_inches='tight')
plt.close()

print('done')
#--------------------------------------------------------------------------------------------
mean_area_extent_biases = pd.DataFrame(index=filter_items,columns=['Mean', 'Winter_mean', 'Winter_std',
                                                                    'Spring_mean', 'Spring_std',
                                                                     'Summer_mean', 'Summer_std',
                                                                     'Autumn_mean', 'Autumn_std'])
    
winter_sts = calculate_seasonal_means(area_extent,filter_items,'winter')
spring_sts = calculate_seasonal_means(area_extent,filter_items,'spring')
summer_sts = calculate_seasonal_means(area_extent,filter_items,'summer')
autumn_sts = calculate_seasonal_means(area_extent,filter_items,'autumn')

mean_area_extent_biases.loc[:,'Mean'] = area_extent[filter_items].mean()
mean_area_extent_biases.loc[:,'Winter_mean'] = winter_sts[0]
mean_area_extent_biases.loc[:,'Winter_std'] = winter_sts[1]
mean_area_extent_biases.loc[:,'Spring_mean'] = spring_sts[0]
mean_area_extent_biases.loc[:,'Spring_std'] = spring_sts[1]
mean_area_extent_biases.loc[:,'Summer_mean'] = summer_sts[0]
mean_area_extent_biases.loc[:,'Summer_std'] = summer_sts[1]
mean_area_extent_biases.loc[:,'Autumn_mean'] = autumn_sts[0]
mean_area_extent_biases.loc[:,'Autumn_std'] = autumn_sts[1]

svnem_csv = '_'.join(['mean_area_extent_biases',cde_run_dte])+ '.csv'
mean_area_extent_biases.to_csv(os.path.join(path_to_put_df,svnem_csv))

# make bar plot
# get data - this will be changed later
data_area_mean = pd.read_csv(os.path.join(path_to_put_df,svnem_csv),index_col=0)
# os.path.join(path_to_put_df,'mean_area_extent_biases_20240519.csv')
# Ensure the index is of type string
data_area_mean.index = data_area_mean.index.astype(str)
# Plotting seasonal barplots
# Extract Method and Surface Cover Type from the first column

data_area_mean['Method'], data_area_mean['Surface cover type'] = zip(*data_area_mean.index.map(extract_method_surface_type))

# Rename the methods and surface cover types
data_area_mean['Method'] = data_area_mean['Method'].replace({
    'ml_e': 'ML-E', 
    'ml_ec': 'ML-EC', 
    'ml_ecc': 'ML-ECC', 
    'climatology_gmasi': 'CLIM'
})

data_area_mean['Surface cover type'] = data_area_mean['Surface cover type'].replace({
    'wtr': 'Water', 
    'snfr': 'Snow free', 
    'snc': 'Snow cover', 
    'ice': 'Ice'
})

# Reset the index to use the columns properly
data_area_mean.reset_index(drop=True, inplace=True)

# Display the final DataFrame structure before plotting
print("\nFinal DataFrame:")
print(data_area_mean.head())


# Plotting seasonal barplots
# Figure 9
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
surface_types = data_area_mean['Surface cover type'].unique()
methods = data_area_mean['Method'].unique()
svenme = '_'.join(['mean_area_extent_biases_bar_plot',cde_run_dte]) + '.png'

# Define the colors for the methods
color_mapping = {'ML-E': 'orange', 'ML-EC': 'g', 'ML-ECC': 'm', 'CLIM': 'b'}

fig, axs = plt.subplots(2, 2, figsize=(14, 12), sharex=True, dpi=1000)
axs = axs.flatten()

bar_width = 0.2
index = np.arange(len(surface_types))

for i, season in enumerate(seasons):
    ax = axs[i]
    for j, method in enumerate(methods):
        method_data = data_area_mean[data_area_mean['Method'] == method]
        means = [method_data[method_data['Surface cover type'] == st][f'{season}_mean'].values[0] for st in surface_types]
        stds = [method_data[method_data['Surface cover type'] == st][f'{season}_std'].values[0] for st in surface_types]

        # Use the defined color for the method
        color = color_mapping[method]
        
        bar = ax.bar(index + j * bar_width, means, bar_width, yerr=stds, capsize=5, label=method, color=color) # 
        
        # Annotate the std dev on top of the bars
        # for k in range(len(surface_types)):
        #     ax.annotate(f'{stds[k]:.2f}', 
        #                 xy=(index[k] + j * bar_width, means[k]), 
        #                 xytext=(0, 3),  # 3 points vertical offset
        #                 textcoords="offset points",
        #                 ha='center', va='bottom',
        #                 )

    ax.set_title(f'{season}', fontsize=15)
    # if i >= 2:  # Only set x-axis labels for the bottom subplots
    #     ax.set_xlabel('Surface Cover Type')
    ax.set_ylabel('Mean PB in area extent estimate [%]', fontsize=15)
    ax.set_xticks(index + bar_width * (len(methods) - 1) / 2)
    ax.set_xticklabels(surface_types)
    ax.minorticks_on()
    ax.tick_params(which='major', axis= 'both', direction='in',length=5, 
                    top=True, right=True, bottom=True, left=True,labelsize=15)  # Adjust major tick length
    ax.tick_params(which='minor', axis= 'both', direction='in', length=2.5, 
                    top=True, right=True, bottom=True, left=True)   # Adjust minor tick length
    ax.grid(which='major', ls='--', lw=0.35, color='grey') 
ax.legend(frameon=False,fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(path_to_put_plots,svenme))
plt.close()

#--------------------------------------------------------------------------------------------

# svnem_csv = '_'.join(['hit_miss_df',cde_run_dte])+ '.csv'

# hit_miss_df_yearly_aveg = hit_miss_df.groupby(hit_miss_df.index.year)[]
# hit_miss_df.to_csv(os.path.join(path_to_put_df,svnem_csv))

# svnem_csv = '_'.join(['area_extent',cde_run_dte])+ '.csv'
# area_extent.to_csv(os.path.join(path_to_put_df,svnem_csv))

# svnem_csv = '_'.join(['px_cnt',cde_run_dte])+ '.csv'
# px_cnt.to_csv(os.path.join(path_to_put_df,svnem_csv))

# svnem_csv = '_'.join(['nh_px_cnt',cde_run_dte])+ '.csv'
# nh_px_cnt.to_csv(os.path.join(path_to_put_df,svnem_csv))

# svnem_csv = '_'.join(['sh_px_cnt',cde_run_dte])+ '.csv'
# sh_px_cnt.to_csv(os.path.join(path_to_put_df,svnem_csv))

# dfs_sve = {
#     'hit_miss_df': hit_miss_df,
#     'area_extent': area_extent,
#     'px_cnt': px_cnt,
#     'nh_px_cnt': nh_px_cnt,
#     'sh_px_cnt': sh_px_cnt,
#     'nh_winter_px_cnt': nh_winter_px_cnt,
#     'nh_spring_px_cnt': nh_spring_px_cnt,
#     'nh_summer_px_cnt': nh_summer_px_cnt,
#     'nh_autumn_px_cnt': nh_autumn_px_cnt,
#     'sh_winter_px_cnt': sh_winter_px_cnt,
#     'sh_spring_px_cnt': sh_spring_px_cnt,
#     'sh_summer_px_cnt': sh_summer_px_cnt,
#     'sh_autumn_px_cnt': sh_autumn_px_cnt
#     }

# # Iterate through the dictionary, saving each DataFrame to a CSV file
# for name, df in dfs_sve.items():
#     sve_nme_csv = '_'.join([name, cde_run_dte]) + '.csv'
#     df.to_csv(os.path.join(path_to_put_df,sve_nme_csv))

# area_extent = pd.read_csv(os.path.join(path_to_put_df,'area_extent_20240327.csv'),index_col=0)
# area_extent.index = pd.to_datetime(area_extent.index)
#%%
print('plot scatter of area extent by orig and estimated land surfce cover type base don different methods')
ml_e = [['ml_e_wtr_px_cnt','gmasi_wtr_px_cnt'], ['ml_e_snfr_px_cnt','gmasi_snfr_px_cnt'],
        ['ml_e_snc_px_cnt','gmasi_snc_px_cnt',], ['ml_e_ice_px_cnt','gmasi_ice_px_cnt',]]

ml_ec = [['ml_ec_wtr_px_cnt','gmasi_wtr_px_cnt'], ['ml_ec_snfr_px_cnt','gmasi_snfr_px_cnt'],
         ['ml_ec_snc_px_cnt','gmasi_snc_px_cnt',], ['ml_ec_ice_px_cnt','gmasi_ice_px_cnt',]]

climatology = [['climatology_wtr_px_cnt','gmasi_wtr_px_cnt'], ['climatology_snfr_px_cnt','gmasi_snfr_px_cnt'],
               ['climatology_snc_px_cnt','gmasi_snc_px_cnt',], ['climatology_ice_px_cnt','gmasi_ice_px_cnt',]]

ml_ecc = [['ml_ecc_wtr_px_cnt','gmasi_wtr_px_cnt'], ['ml_ecc_snfr_px_cnt','gmasi_snfr_px_cnt'],
          ['ml_ecc_snc_px_cnt','gmasi_snc_px_cnt',], ['ml_ecc_ice_px_cnt','gmasi_ice_px_cnt',]]

# e = [['e_wtr_px_cnt','gmasi_wtr_px_cnt'], ['e_snfr_px_cnt','gmasi_snfr_px_cnt'],
#      ['e_snc_px_cnt','gmasi_snc_px_cnt',], ['e_ice_px_cnt','gmasi_ice_px_cnt',]]

cl2plt = [ml_e, ml_ec, ml_ecc, climatology] # , e
#-----------------------------------

# scatter compare combined case of per class area extents
scattter_compare(px_cnt,cl2plt,2)
svenme = '_'.join(['scatter_compare_px_cnt_cmb',cde_run_dte]) + '.png'
plt.savefig(os.path.join(path_to_put_plots,svenme))
plt.close()

#-----------------------------------
# scatter compare combined case of per class area extents
scattter_compare(nh_px_cnt,cl2plt,2)
svenme = '_'.join(['nh_scatter_compare_px_cnt_cmb',cde_run_dte]) + '.png'
plt.savefig(os.path.join(path_to_put_plots,svenme))
plt.close()

#-----------------------------------
# scatter compare combined case of per class area extents
scattter_compare(sh_px_cnt,cl2plt,2)
svenme = '_'.join(['sh_scatter_compare_px_cnt_cmb',cde_run_dte]) + '.png'
plt.savefig(os.path.join(path_to_put_plots,svenme))
plt.close()

#-----------------------------------
# # scatter compare per class area extents in NH winter
# scattter_compare(nh_winter_px_cnt,cl2plt,2)
# svenme = '_'.join(['scatter_compare_px_cnt_nh_winter',cde_run_dte]) + '.png'
# plt.savefig(os.path.join(path_to_put_plots,svenme))
# plt.close()

# #-----------------------------------
# # scatter compare per class area extents in NH spring
# scattter_compare(nh_spring_px_cnt,cl2plt,2)
# svenme = '_'.join(['scatter_compare_px_cnt_nh_spring',cde_run_dte]) + '.png'
# plt.savefig(os.path.join(path_to_put_plots,svenme))
# plt.close()

# #-----------------------------------
# # scatter compare per class area extents in NH summer
# scattter_compare(nh_summer_px_cnt,cl2plt,2)
# svenme = '_'.join(['scatter_compare_px_cnt_nh_summer',cde_run_dte]) + '.png'
# plt.savefig(os.path.join(path_to_put_plots,svenme))
# plt.close()

# #-----------------------------------
# # scatter compare per class area extents in NH autumn
# scattter_compare(nh_autumn_px_cnt,cl2plt,2)
# svenme = '_'.join(['scatter_compare_px_cnt_nh_autumn',cde_run_dte]) + '.png'
# plt.savefig(os.path.join(path_to_put_plots,svenme))
# plt.close()

# #-----------------------------------
# # scatter compare per class area extents in SH winter
# scattter_compare(sh_winter_px_cnt,cl2plt,2)
# svenme = '_'.join(['scatter_compare_px_count_sh_winter',cde_run_dte]) + '.png'
# plt.savefig(os.path.join(path_to_put_plots,svenme))
# plt.close()

# #-----------------------------------
# # scatter compare per class area extents in SH spring
# scattter_compare(sh_spring_px_cnt,cl2plt,2)
# svenme = '_'.join(['scatter_compare_px_cnt_sh_spring',cde_run_dte]) + '.png'
# plt.savefig(os.path.join(path_to_put_plots,svenme))
# plt.close()

# #-----------------------------------
# # scatter compare per class area extents in SH summer
# scattter_compare(sh_summer_px_cnt,cl2plt,2)
# svenme = '_'.join(['scatter_compare_px_cnt_sh_summer',cde_run_dte]) + '.png'
# plt.savefig(os.path.join(path_to_put_plots,svenme))
# plt.close()

# #-----------------------------------
# # scatter compare per class area extents in SH autumn
# scattter_compare(sh_autumn_px_cnt,cl2plt,2)
# svenme = '_'.join(['scatter_compare_px_cnt_sh_autumn',cde_run_dte]) + '.png'
# plt.savefig(os.path.join(path_to_put_plots,svenme))
# plt.close()

print('done!')
# #-----------------------------------

print('compute error metrics based on land surface cover type mapped by the different methods and GMASI')
cls_elem = ['wtr','snfr','snc','ice']
metrics_elements = [('ML-E', ml_e), ('ML-EC', ml_ec), 
                    ('ML-ECC', ml_ecc), ('CLIM', climatology)]

dfs_to_compute_met = [('global_px_cnt_met',px_cnt), 
                      
                      ('nh_px_cnt_met',nh_px_cnt), ('sh_px_cnt_met',sh_px_cnt),
                      
                      ('nh_winter_px_cnt_met',nh_winter_px_cnt),('nh_spring_px_cnt',nh_spring_px_cnt), 
                      ('nh_summer_px_cnt_met',nh_summer_px_cnt), ('nh_autumn_px_cnt_met',nh_autumn_px_cnt),

                      ('sh_winter_px_cnt_met',sh_winter_px_cnt), ('sh_spring_px_cnt_met',sh_spring_px_cnt),
                      ('sh_summer_px_cnt_met',sh_summer_px_cnt), ('sh_autumn_px_cnt_met',sh_autumn_px_cnt)]
#--------------------------------------

for cp in dfs_to_compute_met:

    nmeprt,df2compute = cp[0],cp[1]

    svnem_csv = '_'.join([nmeprt,cde_run_dte]) + '.csv'

    extent_met_computed = count_metric_compute(df2compute,cls_elem,metrics_elements)

    extent_met_computed.to_csv(os.path.join(path_to_put_df,svnem_csv))

print('********************** done ********************')
#%%
print('begin cat evaluation')
print('**********************************************************')
clms2plt = ['count', 'Hits', 'Miss', 'False alarms']

nh_sim_lst_to_compute_cat_stats = [('ML-E',ml_e_nh), 
                                   ('ML-EC',ml_ec_nh),
                                   ('ML-ECC',ml_ecc_nh), 
                                   ('CLIM', climatology_nh),] # ('E', e_nh)

nh_winter_sim_lst_to_compute_cat_stats = [('ML-E',nh_winter_ml_e), 
                                          ('ML-EC',nh_winter_ml_ec),
                                          ('ML-ECC',nh_winter_ml_ecc),                                           
                                          ('CLIM', nh_winter_climatology),] # ('E', nh_winter_e),

nh_spring_sim_lst_to_compute_cat_stats = [('ML-E',nh_spring_ml_e), 
                                          ('ML-EC',nh_spring_ml_ec),
                                          ('ML-ECC',nh_spring_ml_ecc),                                           
                                          ('CLIM', nh_spring_climatology)] # ('E',nh_spring_e),

nh_summer_sim_lst_to_compute_cat_stats = [('ML-E',nh_summer_ml_e), 
                                          ('ML-EC',nh_summer_ml_ec),
                                          ('ML-ECC',nh_summer_ml_ecc),                                          
                                          ('CLIM', nh_summer_climatology)] # ('E',nh_summer_e), 

nh_autumn_sim_lst_to_compute_cat_stats = [('ML-E',nh_autumn_ml_e), 
                                          ('ML-EC',nh_autumn_ml_ec),
                                          ('ML-ECC',nh_autumn_ml_ecc),                                         
                                          ('CLIM', nh_autumn_climatology)]  #('E',nh_autumn_e), 

#------------------------------------------------
sh_sim_lst_to_compute_cat_stats = [('ML-E',ml_e_sh), 
                                   ('ML-EC',ml_ec_sh),
                                   ('ML-ECC',ml_ecc_sh),                                    
                                   ('CLIM', climatology_sh)] # ('E',ml_e_sh), 

sh_winter_sim_lst_to_compute_cat_stats = [('ML-E',sh_winter_ml_e), 
                                          ('ML-EC',sh_winter_ml_ecc),
                                          ('ML-ECC',sh_winter_ml_ecc),                                           
                                          ('CLIM', sh_winter_climatology)] # ('E',sh_winter_e),

sh_spring_sim_lst_to_compute_cat_stats = [('ML-E',sh_spring_ml_e), 
                                          ('ML-EC',sh_spring_ml_ec),
                                          ('ML-ECC',sh_spring_ml_ecc),                                         
                                          ('CLIM', sh_spring_climatology)] #  ('E',sh_spring_e),  

sh_summer_sim_lst_to_compute_cat_stats = [('ML-E',sh_summer_ml_e), 
                                          ('ML-EC',sh_summer_ml_ec),
                                          ('ML-ECC',sh_summer_ml_ecc),                                           
                                          ('CLIM', sh_summer_climatology)] # ('E',sh_summer_e),

sh_autumn_sim_lst_to_compute_cat_stats = [('ML-E',sh_autumn_ml_e), 
                                          ('ML-EC',sh_autumn_ml_ec),
                                          ('ML-ECC',sh_autumn_ml_ecc),                                           
                                          ('CLIM', sh_autumn_climatology)] # ('E',sh_autumn_e), 

# #------------------------------------------------

nh_sim_lst_to_compute = [('nh_cmb',nh_sim_lst_to_compute_cat_stats,gmasi_nh), 
                         ('nh_winter',nh_winter_sim_lst_to_compute_cat_stats,nh_winter_gmasi),
                         ('nh_spring',nh_spring_sim_lst_to_compute_cat_stats,nh_spring_gmasi), 
                         ('nh_summer',nh_summer_sim_lst_to_compute_cat_stats,nh_summer_gmasi),
                         ('nh_autumn',nh_autumn_sim_lst_to_compute_cat_stats,nh_autumn_gmasi)]

for q in nh_sim_lst_to_compute:

    svneme,s_lst,o_lst = q[0],q[1],q[2]

    svnem_csv = '_'.join([svneme,cde_run_dte]) + '.csv'

    cat_df_out = cat_metrcs_computed(s_lst,o_lst)

    cat_df_out.to_csv(os.path.join(path_to_put_df,svnem_csv))
del(q,svnem_csv)
gc.collect()

print('done with NH')
# #------------------------------------------------
# SH
sh_sim_lst_to_compute = [('sh_cmb',sh_sim_lst_to_compute_cat_stats,gmasi_sh), 
                         ('sh_winter',sh_winter_sim_lst_to_compute_cat_stats,sh_winter_gmasi),
                         ('sh_spring',sh_spring_sim_lst_to_compute_cat_stats,sh_spring_gmasi), 
                         ('sh_summer',sh_summer_sim_lst_to_compute_cat_stats,sh_summer_gmasi),
                         ('sh_autumn',sh_autumn_sim_lst_to_compute_cat_stats,sh_autumn_gmasi)]

for q in sh_sim_lst_to_compute:

    svneme,s_lst,o_lst = q[0],q[1],q[2]

    svnem_csv = '_'.join([svneme,cde_run_dte]) + '.csv'

    cat_df_out = cat_metrcs_computed(s_lst,o_lst)

    cat_df_out.to_csv(os.path.join(path_to_put_df,svnem_csv))
del(q,svnem_csv)

print('done with NH')

#%%
print('-----------------------begin time series plots-----------------------------------------')
# make time series plot of hit miss
fig,axes = plt.subplots(1, 5,figsize=(25,5), sharex=True,dpi=1000, 
                        gridspec_kw={'width_ratios': [0.98]*5},) # 
plt.subplots_adjust(bottom=0.2)
lw= 1

axes[0].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-ECC-GMASI-miss'],ls=':',c=colors[6], lw=3,)
axes[0].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-E-GMASI-miss'],ls='--',c=colors[0], lw=lw,)
axes[0].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-EC-GMASI-miss'],ls='-.',c=colors[7], lw=1.5)
axes[0].plot(hit_miss_df.index,hit_miss_df.loc[:,'Climatology-GMASI-miss'],ls=':',c=colors[4], lw=lw)

axes[0].set_ylim(0, 5)
axes[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 12]))
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# axes[1,0].legend(loc='best',frameon=False,fontsize=9)
axes[0].set_title('Combined',fontsize=20)
axes[0].set_ylabel('Daily percentage mismatch [%]',fontsize=15)
# Get current y-axis tick locations
y_ticks = axes[0].get_yticks()
# Round the tick values to desired precision, here rounding to nearest whole number
# Convert the tick values to integers
int_ticks = [int(val) for val in y_ticks]
# Set new, rounded y-axis tick labels
axes[0].set_yticklabels(int_ticks)

axes[1].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-E-wter-miss'],ls='--',c=colors[0], lw=lw,)
axes[1].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-ECC-wter-miss'],ls=':',c=colors[6], lw=3,)
axes[1].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-EC-wter-miss'],ls='-.',c=colors[7], lw=1.5,)
axes[1].plot(hit_miss_df.index,hit_miss_df.loc[:,'Climatology-wter-miss'],ls=':',c=colors[4], lw=lw,)
axes[1].set_title('Water', fontsize=20)

axes[1].set_ylim(0, 3) # 15
# axes[1,1].set_yticklabels([])
axes[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[12]))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

axes[2].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-E-snflnd-miss'],ls='--',c=colors[0], lw=lw,)
axes[2].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-ECC-snflnd-miss'],ls=':',c=colors[6], lw=3,)
axes[2].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-EC-snflnd-miss'],ls='-.',c=colors[7], lw=1.5,)
axes[2].plot(hit_miss_df.index,hit_miss_df.loc[:,'Climatology-snflnd-miss'],ls=':',c=colors[4], lw=lw,)
axes[2].set_title('Snow free', fontsize=20)

axes[2].set_ylim(0, 10)
# axes[1,2].set_yticklabels([])
axes[2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[12]))
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

axes[3].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-E-snclnd-miss'],ls='--',c=colors[0], lw=lw)
axes[3].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-ECC-snclnd-miss'],ls=':',c=colors[6], lw=3,)
axes[3].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-EC-snclnd-miss'],ls='-.',c=colors[7], lw=1.5)
axes[3].plot(hit_miss_df.index,hit_miss_df.loc[:,'Climatology-snclnd-miss'],ls=':',c=colors[4], lw=lw)
axes[3].set_title('Snow cover', fontsize=20)

axes[3].set_ylim(0, 10)
# axes[1,3].set_yticklabels([])
axes[3].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[12]))
axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

axes[4].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-E-ice-miss'],ls='--',c=colors[0], lw=lw, label='ML-E')
axes[4].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-ECC-ice-miss'],ls=':',c=colors[6],lw=3, label='ML-ECC')
axes[4].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-EC-ice-miss'],ls='-.',c=colors[7], lw=1.5, label='ML-EC')
axes[4].plot(hit_miss_df.index,hit_miss_df.loc[:,'Climatology-ice-miss'],ls=':',c=colors[4], lw=lw, label='CLIM')

axes[4].set_ylim(0, 15)
# axes[1,4].set_yticklabels([])
axes[4].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[12]))
axes[4].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# axes[1,4].xticks(rotation=45, ha='right')
axes[4].set_title('Ice', fontsize=20)

for ax in axes.flatten():
    
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', labelsize=15,
                   top=True, right=True, bottom=True, left=True)
    ax.grid(which='major', linestyle='--', linewidth='0.5', color='grey')  

fig.text(0.5,0.1, 'Year', ha='center', va='center', rotation='horizontal', fontsize=20)

# Since all lines are the same across subplots, take the handles and labels from the first subplot
handles, labels = axes[0].get_legend_handles_labels()

legend = fig.legend(handles=handles, labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),                       
                    ncol=5, fontsize=20,frameon=False) # (0.8, 0.5)
# Set legend text color to black for better visibility
for text in legend.get_texts():
    text.set_color('black')

svenme = '_'.join(['timeseries_hit-miss',cde_run_dte]) + '.png'
plt.savefig(os.path.join(path_to_put_plots,svenme),bbox_inches='tight')
plt.close()

gc.collect()

# #----------------------------------------------------------------
# Adjust layout for vertically stacked subplots
# Figure 5
fig, axes = plt.subplots(5, 1, figsize=(15, 10), dpi=1000, sharex=True,  # 5 rows, 1 column
                         gridspec_kw={'height_ratios': [1] * 5})  # Equal height for all subplots
plt.subplots_adjust(hspace=0.27, bottom=0.09, left=0.1)  # Increase space between subplots

lw = 1

# Combined
axes[0].plot(hit_miss_df.index, hit_miss_df['ML-ECC-GMASI-miss'], ls=':', c=colors[6], lw=3)
axes[0].plot(hit_miss_df.index, hit_miss_df['ML-E-GMASI-miss'], ls='-', c=colors[0], lw=lw)
axes[0].plot(hit_miss_df.index, hit_miss_df['ML-EC-GMASI-miss'], ls='-.', c=colors[7], lw=1.5)
axes[0].plot(hit_miss_df.index, hit_miss_df['Climatology-GMASI-miss'], ls='-', c=colors[4], lw=lw)
axes[0].set_title('Combined', fontsize=18, loc='left')
axes[0].set_ylim(0, 3.5)
axes[0].set_yticks([0, 1.5, 3.5])
# Water
axes[1].plot(hit_miss_df.index, hit_miss_df['ML-E-wter-miss'], ls='-', c=colors[0], lw=lw)
axes[1].plot(hit_miss_df.index, hit_miss_df['ML-ECC-wter-miss'], ls=':', c=colors[6], lw=3)
axes[1].plot(hit_miss_df.index, hit_miss_df['ML-EC-wter-miss'], ls='-.', c=colors[7], lw=1.5)
axes[1].plot(hit_miss_df.index, hit_miss_df['Climatology-wter-miss'], ls='-', c=colors[4], lw=lw)
axes[1].set_title('Water', fontsize=18, loc='left')
axes[1].set_ylim(0, 3)
axes[1].set_yticks([0, 1.5, 3])

# Snow free
axes[2].plot(hit_miss_df.index, hit_miss_df['ML-E-snflnd-miss'], ls='-', c=colors[0], lw=lw)
axes[2].plot(hit_miss_df.index, hit_miss_df['ML-ECC-snflnd-miss'], ls=':', c=colors[6], lw=3)
axes[2].plot(hit_miss_df.index, hit_miss_df['ML-EC-snflnd-miss'], ls='-.', c=colors[7], lw=1.5)
axes[2].plot(hit_miss_df.index, hit_miss_df['Climatology-snflnd-miss'], ls='-', c=colors[4], lw=lw)
axes[2].set_title('Snow Free', fontsize=18, loc='left')
axes[2].set_ylim(0, 9)
axes[2].set_yticks([0, 5.0, 10.0])

# Snow cover
axes[3].plot(hit_miss_df.index, hit_miss_df['ML-E-snclnd-miss'], ls='-', c=colors[0], lw=lw)
axes[3].plot(hit_miss_df.index, hit_miss_df['ML-ECC-snclnd-miss'], ls=':', c=colors[6], lw=3)
axes[3].plot(hit_miss_df.index, hit_miss_df['ML-EC-snclnd-miss'], ls='-.', c=colors[7], lw=1.5)
axes[3].plot(hit_miss_df.index, hit_miss_df['Climatology-snclnd-miss'], ls='-', c=colors[4], lw=lw)
axes[3].set_title('Snow Cover', fontsize=18, loc='left')
axes[3].set_ylim(0, 10)
axes[3].set_yticks([0, 5.0, 10.0])
# Ice
axes[4].plot(hit_miss_df.index, hit_miss_df['ML-E-ice-miss'], ls='-', c=colors[0], lw=lw, label='ML-E')
axes[4].plot(hit_miss_df.index, hit_miss_df['ML-ECC-ice-miss'], ls=':', c=colors[6], lw=3, label='ML-ECC')
axes[4].plot(hit_miss_df.index, hit_miss_df['ML-EC-ice-miss'], ls='-.', c=colors[7], lw=1.5, label='ML-EC')
axes[4].plot(hit_miss_df.index, hit_miss_df['Climatology-ice-miss'], ls='-', c=colors[4], lw=lw, label='CLIM')
axes[4].set_title('Ice', fontsize=18, loc='left')
axes[4].set_ylim(0, 13)
axes[4].set_yticks([0, 7.0, 14.0])

# General settings
for ax in axes:
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', labelsize=15, top=True, right=True)
    ax.grid(which='major', linestyle='--', linewidth=0.5, color='grey')
    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', labelrotation=0)

# Add shared labels
fig.text(0.06, 0.5, 'Daily Percentage Mismatch [%]', ha='center', va='center', rotation='vertical', fontsize=18)
fig.text(0.5, 0.035, 'Year', ha='center', va='center', rotation='horizontal', fontsize=18)

# Add legend
handles, labels = axes[4].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), 
           ncol=5, fontsize=18, frameon=False)

# Save the figure
svenme = '_'.join(['timeseries_hit-miss', cde_run_dte]) + '_vertical.png'
plt.savefig(os.path.join(path_to_put_plots, svenme), bbox_inches='tight')
plt.close()


#--------------------------------------------------------------------
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
    # ['orange', 'k', 'crimson', 'cyan', 'b', 'lime', 'm', 'g', 'r']  # Color list

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

plot_daily_percentage_mismatch(hit_miss_df)

gc.collect()
#%%

for d in ['1988005','1988076','1990005']:
    
    yrdoy = d

    dtme = pd.to_datetime(f'{yrdoy[:4]}-{yrdoy[4:]}', format='%Y-%j')

    # define and check for existence of files to read
        # RF estimated using only ERA5
    rfmp_read_nme = os.path.join(path_to_estimated_autosnw,'_'.join(['RF_estimated_autosnow_using_only_ERA5_data',
                                                                     yrdoy,'0.1deg_wgs']) + '.tif')
    # RF estimated using Climatology
    climrfmp_read_nme = os.path.join(path_to_estimated_autosnw,'_'.join(['RF_estimated_autosnow_using_alldataclim',
                                                                         yrdoy,'0.1deg_wgs']) + '.tif')
    # RF estimated using temp dependent Climatology
    temp_climrfmp_read_nme = os.path.join(path_to_clim_only_autosnw_estimated,'_'.join(['alldata_1992_2022_clim_subsetted_by_airTemp',
                                                                                        yrdoy,'0.1deg_wgs']) + '.tif')
    
    orig_dat_red_nme = os.path.join(path_to_autosnow_data,'_'.join(['gmasi_snowice_reproc_v003',yrdoy,
                                                                    '0.1deg_wgs']) + '.tif')
    
    cor_climrfmp_read_nme = os.path.join(path_to_estimated_autosnw,'_'.join(['corrected_RF_estimated_autosnow_using_alldataclim',
                                                                         yrdoy,'0.1deg_wgs']) + '.tif')
    #------------------------------------------------------

    # read  data    
    rf_autsnw_estimated = xr.open_dataarray(rfmp_read_nme)
    rf_autsnw_estimated_arr = rf_autsnw_estimated.data[0,:,:]
    rf_autsnw_estimated_arr = np.where(rf_autsnw_estimated_arr > 3,np.nan,rf_autsnw_estimated_arr)

    climrf_autsnw_estimated = xr.open_dataarray(climrfmp_read_nme)
    climrf_autsnw_estimated_arr = climrf_autsnw_estimated.data[0,:,:]
    climrf_autsnw_estimated_arr = np.where(climrf_autsnw_estimated_arr > 3,np.nan,climrf_autsnw_estimated_arr)

    temp_climrf_autsnw_estimated = xr.open_dataarray(temp_climrfmp_read_nme)
    temp_climrf_autsnw_estimated_arr = temp_climrf_autsnw_estimated.data[0,:,:]
    temp_climrf_autsnw_estimated_arr = np.where(temp_climrf_autsnw_estimated_arr > 3,np.nan,temp_climrf_autsnw_estimated_arr)

    cor_climrf_autsnw_estimated = xr.open_dataarray(cor_climrfmp_read_nme)
    cor_climrf_autsnw_estimated_arr = cor_climrf_autsnw_estimated.data[0,:,:]
    cor_climrf_autsnw_estimated_arr = np.where(cor_climrf_autsnw_estimated_arr > 3,np.nan,cor_climrf_autsnw_estimated_arr)

    # the original autosnow data
    dat_auto = xr.open_dataarray(orig_dat_red_nme) 
    dat_auto_array = dat_auto.data[0,:,:]
    dat_auto_array = np.where(dat_auto_array > 3,np.nan,dat_auto_array)
    y_shp,x_shp = dat_auto_array.shape[0],dat_auto_array.shape[1] 

    #------------------------------------------------------

    rf_orig_wtr_hit = np.logical_and(rf_autsnw_estimated_arr == 0, dat_auto_array == 0)
    rf_orig_snfr_hit = np.where(((rf_autsnw_estimated_arr == 1) & (dat_auto_array == 1)),1,np.nan)
    rf_orig_snc_hit = np.where(((rf_autsnw_estimated_arr == 2) & (dat_auto_array == 2)),2,np.nan)
    rf_orig_ice_hit = np.where(((rf_autsnw_estimated_arr == 3) & (dat_auto_array == 3)),3,np.nan)
    rf_orig_snfr_snc_anom_hit1 = np.where(((dat_auto_array == 1) & (rf_autsnw_estimated_arr == 2)),1,np.nan)
    rf_orig_snfr_snc_anom_hit2 = np.where(((dat_auto_array == 2) & (rf_autsnw_estimated_arr == 1)),1,np.nan)
    rf_orig_ice_wtr_anom_hit1 = np.where(((dat_auto_array == 0) & (rf_autsnw_estimated_arr == 3)),1,np.nan)
    rf_orig_ice_wtr_anom_hit2 = np.where(((dat_auto_array == 3) & (rf_autsnw_estimated_arr == 0)),1,np.nan)
    #------------------------------------------------------

    climrf_orig_wtr_hit = np.logical_and(climrf_autsnw_estimated_arr == 0, dat_auto_array == 0)
    climrf_orig_snfr_hit = np.where(((climrf_autsnw_estimated_arr == 1) & (dat_auto_array == 1)),1,np.nan)
    climrf_orig_snc_hit = np.where(((climrf_autsnw_estimated_arr == 2) & (dat_auto_array == 2)),2,np.nan)
    climrf_orig_ice_hit = np.where(((climrf_autsnw_estimated_arr == 3) & (dat_auto_array == 3)),3,np.nan)
    climrf_orig_snfr_snc_anom_hit1 = np.where(((dat_auto_array == 1) & (climrf_autsnw_estimated_arr == 2)),1,np.nan)
    climrf_orig_snfr_snc_anom_hit2 = np.where(((dat_auto_array == 2) & (climrf_autsnw_estimated_arr== 1)),1,np.nan)
    climrf_orig_ice_wtr_anom_hit1 = np.where(((dat_auto_array == 0) & (climrf_autsnw_estimated_arr == 3)),1,np.nan)
    climrf_orig_ice_wtr_anom_hit2 = np.where(((dat_auto_array == 3) & (climrf_autsnw_estimated_arr == 0)),1,np.nan)
    #------------------------------------------------------

    temp_climrf_orig_wtr_hit = np.logical_and(temp_climrf_autsnw_estimated_arr == 0, dat_auto_array == 0)
    temp_climrf_orig_snfr_hit = np.where(((temp_climrf_autsnw_estimated_arr == 1) & (dat_auto_array == 1)),1,np.nan)
    temp_climrf_orig_snc_hit = np.where(((temp_climrf_autsnw_estimated_arr == 2) & (dat_auto_array == 2)),2,np.nan)
    temp_climrf_orig_ice_hit = np.where(((temp_climrf_autsnw_estimated_arr == 3) & (dat_auto_array == 3)),3,np.nan)
    temp_climrf_orig_snfr_snc_anom_hit1 = np.where(((dat_auto_array == 1) & (temp_climrf_autsnw_estimated_arr == 2)),1,np.nan)
    temp_climrf_orig_snfr_snc_anom_hit2 = np.where(((dat_auto_array == 2) & (temp_climrf_autsnw_estimated_arr == 1)),1,np.nan)
    temp_climrf_orig_ice_wtr_anom_hit1 = np.where(((dat_auto_array == 0) & (temp_climrf_autsnw_estimated_arr == 3)),1,np.nan)
    temp_climrf_orig_ice_wtr_anom_hit2 = np.where(((dat_auto_array== 3) & (temp_climrf_autsnw_estimated_arr  == 0)),1,np.nan)

    cor_climrf_orig_wtr_hit = np.logical_and(cor_climrf_autsnw_estimated_arr == 0, dat_auto_array == 0)
    cor_climrf_orig_snfr_hit = np.where(((cor_climrf_autsnw_estimated_arr == 1) & (dat_auto_array == 1)),1,np.nan)
    cor_climrf_orig_snc_hit = np.where(((cor_climrf_autsnw_estimated_arr == 2) & (dat_auto_array == 2)),2,np.nan)
    cor_climrf_orig_ice_hit = np.where(((cor_climrf_autsnw_estimated_arr == 3) & (dat_auto_array == 3)),3,np.nan)
    cor_climrf_orig_snfr_snc_anom_hit1 = np.where(((dat_auto_array == 1) & (cor_climrf_autsnw_estimated_arr == 2)),1,np.nan)
    cor_climrf_orig_snfr_snc_anom_hit2 = np.where(((dat_auto_array == 2) & (cor_climrf_autsnw_estimated_arr == 1)),1,np.nan)
    cor_climrf_orig_ice_wtr_anom_hit1 = np.where(((dat_auto_array == 0) & (cor_climrf_autsnw_estimated_arr == 3)),1,np.nan)
    cor_climrf_orig_ice_wtr_anom_hit2 = np.where(((dat_auto_array== 3) & (cor_climrf_autsnw_estimated_arr  == 0)),1,np.nan)
    #------------------------------------------------------

    dat_auto_array_wtr = dat_auto_array == 0
    dat_auto_array_snfr_hit = np.where((dat_auto_array == 1),1,np.nan)
    dat_auto_array_snc_hit = np.where((dat_auto_array == 2),2,np.nan)
    dat_auto_array_ice_hit = np.where((dat_auto_array == 3),3,np.nan)
    #------------------------------------------------------

    wtr_cmap = ListedColormap(['blue'])
    ice_cmap = ListedColormap(['yellow'])
    sfr_cmap = ListedColormap(['green'])
    snc_cmap = ListedColormap(['white'])
    snfr_snc_anom_cmap1 = ListedColormap(['black'])
    snfr_snc_anom_cmap2 = ListedColormap(['cyan'])
    ice_wtr_anom_cmap1 = ListedColormap(['red'])
    ice_wtr_anom_cmap2 = ListedColormap(['orange'])
    #------------------------------------------------------
    legend_labels = [
        Patch(color='blue', label='Water'),
        Patch(color='yellow', label='Ice'),
        Patch(color='green', label='Snow free'),
        Patch(color='white', label='Snow cover'),
        Patch(color='cyan', label='GMASI: Snow cover, Model: Snow free'),
        Patch(color='black',label= 'GMASI: Snow free, Model: Snow cover'),
        Patch(color='orange', label= 'GMASI: Ice, Model: Water'),
        Patch(color='red', label='GMASI: Water, Model: Ice')
    ]
    #------------------------------------------------------

    fg,ax = plt.subplots(3,2, figsize = (20,12), sharey=True,tight_layout=True, dpi=1000,) # 
    # plt.tight_layout(pad=0.4, w_pad=0.0001)
    plt.subplots_adjust(top=0.5) # ,
    # gridspec_kw={'width_ratios': [0.98]*2}, ax[0,0]

    ax[0,0].imshow(dat_auto_array_wtr,cmap=wtr_cmap, extent=plt_extent)
    ax[0,0].imshow(dat_auto_array_snfr_hit,cmap=sfr_cmap,interpolation='nearest', extent=plt_extent)
    ax[0,0].imshow(dat_auto_array_snc_hit,cmap=snc_cmap,interpolation='nearest', extent=plt_extent)
    ax[0,0].imshow(dat_auto_array_ice_hit,cmap=ice_cmap,interpolation='nearest', extent=plt_extent)
    ax[0,0].set_title('GMASI', ha='center', va='top', fontsize=17)
    # Add text to the upper-left corner of the entire figure
    # ax[0,0].text(0.02, 0.98, 'Original', ha='left', va='top', 
    #         fontsize=12,transform=ax[0,0].transAxes)
    for label in ax[0,0].get_xticklabels():
        label.set_visible(False)
    #-------------------------------------

    ax[0,1].imshow(rf_orig_wtr_hit,cmap=wtr_cmap, extent=plt_extent)
    ax[0,1].imshow(rf_orig_snfr_hit,cmap=sfr_cmap,interpolation='nearest', extent=plt_extent)
    ax[0,1].imshow(rf_orig_snc_hit,cmap=snc_cmap,interpolation='nearest', extent=plt_extent)
    ax[0,1].imshow(rf_orig_snfr_snc_anom_hit1,cmap=snfr_snc_anom_cmap1,interpolation='nearest', extent=plt_extent)
    ax[0,1].imshow(rf_orig_snfr_snc_anom_hit2,cmap=snfr_snc_anom_cmap2,interpolation='nearest', extent=plt_extent)
    ax[0,1].imshow(rf_orig_ice_wtr_anom_hit1,cmap=ice_wtr_anom_cmap1,interpolation='nearest', extent=plt_extent)
    ax[0,1].imshow(rf_orig_ice_wtr_anom_hit2,cmap=ice_wtr_anom_cmap2,interpolation='nearest', extent=plt_extent)
    ax[0,1].imshow(rf_orig_ice_hit,cmap=ice_cmap,interpolation='nearest', extent=plt_extent)

    ax[0,1].set_title('ML-E', ha='center', va='top', fontsize=17)
    # Add text to the upper-left corner of the entire figure
    # ax[0,1].text(0.02, 0.98, 'ERA5 only', ha='left', va='top', 
    #         fontsize=12,transform=ax[0,1].transAxes)
    for label in ax[0,1].get_xticklabels():
        label.set_visible(False)
    #-------------------------------------
    ax[1,0].imshow(climrf_orig_wtr_hit,cmap=wtr_cmap, extent=plt_extent)
    ax[1,0].imshow(climrf_orig_snfr_hit,cmap=sfr_cmap,interpolation='nearest', extent=plt_extent)
    ax[1,0].imshow(climrf_orig_snc_hit,cmap=snc_cmap,interpolation='nearest', extent=plt_extent)
    ax[1,0].imshow(climrf_orig_snfr_snc_anom_hit1,cmap=snfr_snc_anom_cmap1,interpolation='nearest', extent=plt_extent)
    ax[1,0].imshow(climrf_orig_snfr_snc_anom_hit2,cmap=snfr_snc_anom_cmap2,interpolation='nearest', extent=plt_extent)
    ax[1,0].imshow(climrf_orig_ice_wtr_anom_hit1,cmap=ice_wtr_anom_cmap1,interpolation='nearest', extent=plt_extent)
    ax[1,0].imshow(climrf_orig_ice_wtr_anom_hit2,cmap=ice_wtr_anom_cmap2,interpolation='nearest', extent=plt_extent)
    ax[1,0].imshow(climrf_orig_ice_hit,cmap=ice_cmap,interpolation='nearest', extent=plt_extent)
    ax[1,0].set_title('ML-EC', ha='center', va='top', fontsize=17)
    # ax[1,0].text(0.02, 0.98, 'ERA5 with climatology', ha='left', va='top', 
    #         fontsize=12,transform=ax[1,0].transAxes)

    # ax[1].text(0.02, 0.38, date_time.strftime('%Y-%m-%d'), ha='left', va='top', 
    #            fontsize=10,transform=ax[1].transAxes, color='white', )

    for label in ax[1,0].get_xticklabels():
        label.set_visible(False)
    #-------------------------------------
    ax[1,1].imshow(cor_climrf_orig_wtr_hit,cmap=wtr_cmap, extent=plt_extent)
    ax[1,1].imshow(cor_climrf_orig_snfr_hit,cmap=sfr_cmap,interpolation='nearest', extent=plt_extent)
    ax[1,1].imshow(cor_climrf_orig_snc_hit,cmap=snc_cmap,interpolation='nearest', extent=plt_extent)
    ax[1,1].imshow(cor_climrf_orig_snfr_snc_anom_hit1,cmap=snfr_snc_anom_cmap1,interpolation='nearest', extent=plt_extent)
    ax[1,1].imshow(cor_climrf_orig_snfr_snc_anom_hit2,cmap=snfr_snc_anom_cmap2,interpolation='nearest', extent=plt_extent)
    ax[1,1].imshow(cor_climrf_orig_ice_wtr_anom_hit1,cmap=ice_wtr_anom_cmap1,interpolation='nearest', extent=plt_extent)
    ax[1,1].imshow(cor_climrf_orig_ice_wtr_anom_hit2,cmap=ice_wtr_anom_cmap2,interpolation='nearest', extent=plt_extent)
    ax[1,1].imshow(cor_climrf_orig_ice_hit,cmap=ice_cmap,interpolation='nearest', extent=plt_extent)
    ax[1,1].set_title('ML-ECC', ha='center', va='top', fontsize=17)

    # ax[1,1].text(0.02, 0.38, dtme.strftime('%Y-%m-%d'), ha='left', va='top', 
    #         fontsize=12,transform=ax[1,1].transAxes, color='white', )

    #-------------------------------------
    ax[2,0].imshow(temp_climrf_orig_wtr_hit,cmap=wtr_cmap, extent=plt_extent)
    ax[2,0].imshow(temp_climrf_orig_snfr_hit,cmap=sfr_cmap,interpolation='nearest', extent=plt_extent)
    ax[2,0].imshow(temp_climrf_orig_snc_hit,cmap=snc_cmap,interpolation='nearest', extent=plt_extent)
    ax[2,0].imshow(temp_climrf_orig_snfr_snc_anom_hit1,cmap=snfr_snc_anom_cmap1,interpolation='nearest', extent=plt_extent)
    ax[2,0].imshow(temp_climrf_orig_snfr_snc_anom_hit2,cmap=snfr_snc_anom_cmap2,interpolation='nearest', extent=plt_extent)
    ax[2,0].imshow(temp_climrf_orig_ice_wtr_anom_hit1,cmap=ice_wtr_anom_cmap1,interpolation='nearest', extent=plt_extent)
    ax[2,0].imshow(temp_climrf_orig_ice_wtr_anom_hit2,cmap=ice_wtr_anom_cmap2,interpolation='nearest', extent=plt_extent)
    ax[2,0].imshow(temp_climrf_orig_ice_hit,cmap=ice_cmap,interpolation='nearest', extent=plt_extent)
    ax[2,0].set_title('CLIM', ha='center', va='top', fontsize=17)
    # ax[2,0].text(0.02, 0.98, 'Climatology', ha='left', va='top', 
    #         fontsize=12,transform=ax[2,0].transAxes)

    # ax[1,1].text(0.02, 0.38, dtme.strftime('%Y-%m-%d'), ha='left', va='top', 
    #         fontsize=12,transform=ax[1,1].transAxes, color='white', )
    
    # Remove the last subplot (ax[2,1]) to make space for the legend
    plt.delaxes(ax[2, 1])
    # Add legend outside the plot at the bottom
    legend = fg.legend(handles=legend_labels, loc='lower center', bbox_to_anchor = (0.72, 0.15),                       
                    facecolor = 'silver',ncol=2, fontsize=15) #
    # loc='upper center', bbox_to_anchor=(0.5, 0.15), (0.65, 0.1)
    # Set legend text color to black for better visibility
    for text in legend.get_texts():
        text.set_color('black')
    
    for i in range(ax.shape[0]):  # Iterate over the rows
        for j in range(ax.shape[1]):  # Iterate over the columns
            if i == 2 and j == 1:
                # Skip the last subplot which has been deleted
                continue
            a = ax[i, j]
            a.minorticks_on()
            a.tick_params(which='major', axis='both', direction='in', length=5, top=True, right=True, 
                          bottom=True, left=True, labelsize=14)
            a.tick_params(which='minor', axis='both', direction='in', length=2.5, top=True, right=True, 
                          bottom=True, left=True)
            a.grid(which='major', c='grey', ls='--', lw=0.35)
            
            # Set the ticks for x and y axis
            a.set_xticks(range(-180, 181, 30))  # x-axis from -180 to 180 at intervals of 30
            a.set_yticks(range(-90, 91, 30))    # y-axis from -90 to 90 at intervals of 30
        
    #------------------------------------------------------

    svenme = '_'.join(['estimate_vrs_orig_with_anomalies',dtme.strftime('%Y%m%d'), cde_run_dte]) + '.png'
    plt.savefig(os.path.join(path_to_put_plots,svenme),bbox_inches='tight')

    plt.close()
# #------------------------------------
print('*************** done with hit miss calculations and saved maps ***********************')




# bins_hits = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# # colors_hits = ['#0000ff', '#1e90ff', '#00ffff', '#00ff7f', '#7fff00', 
# #                '#ffff00', '#ff7f00', '#ff4500', '#ff0000', 'white']
# # cmap_hits = mcolors.ListedColormap(colors_hits)
# # norm_hits = mcolors.BoundaryNorm(bins_hits, cmap_hits.N)

# # # Create a color map with 0-10% as white for "Miss"
# # colors_miss = ['white', '#1e90ff', '#00ffff', '#00ff7f', '#7fff00', '#ffff00', 
# #                '#ff7f00', '#ff4500', '#ff0000', '#ff0000']
# # cmap_miss = mcolors.ListedColormap(colors_miss)
# # norm_miss = mcolors.BoundaryNorm(bins_hits, cmap_miss.N)

# # Create a grayscale colormap for "Hits" where black represents the lowest percentage range (0-10%)
# colors_hits_grey = ['#000000', '#1a1a1a', '#333333', '#4d4d4d', '#666666', 
#                     '#808080', '#999999', '#b3b3b3', '#cccccc', '#e6e6e6', '#ffffff']
# cmap_hits_grey = mcolors.ListedColormap(colors_hits_grey)
# norm_hits_grey = mcolors.BoundaryNorm(bins_hits, cmap_hits_grey.N)

# # Create a grayscale colormap for "Miss" where black represents the highest percentage range (90-100%)
# colors_miss_grey = ['#ffffff', '#e6e6e6', '#cccccc', '#b3b3b3', '#999999', 
#                     '#808080', '#666666', '#4d4d4d', '#333333', '#1a1a1a', '#000000']
# cmap_miss_grey = mcolors.ListedColormap(colors_miss_grey)
# norm_miss_grey = mcolors.BoundaryNorm(bins_hits, cmap_miss_grey.N)


# fig, axes = plt.subplots(4,2, figsize=(20, 5),sharey=True,sharex=True)
# plt.subplots_adjust(bottom=0.0000001,hspace=0.2,wspace=0.005) # dpi=1000, 

# # ERA5 only
# axes[0,0].imshow(nh_arr2_plt[0][1],cmap =cmap_hits_grey, extent=[-180,180,45,90],norm=norm_hits_grey)
# # axes[0,0].set_title('Hits', fontsize =15)

# axes[0,1].imshow(nh_arr2_plt[0][2],cmap =cmap_miss_grey, extent=[-180,180,45,90],norm=norm_miss_grey)
# axes[0,1].text(1.02, 0.25,nh_arr2_plt[0][0],fontsize=10, rotation='vertical', 
#                 transform=axes[0,1].transAxes,fontweight='bold')
# #-------------------------------------------------------

# # ERA5 with climatology
# axes[1,0].imshow(nh_arr2_plt[1][1],cmap = cmap_hits_grey, extent=[-180,180,45,90],norm=norm_hits_grey)    

# axes[1,1].imshow(nh_arr2_plt[1][2],cmap = cmap_miss_grey, extent=[-180,180,45,90],norm=norm_miss_grey)
# axes[1,1].text(1.02, 0.25,nh_arr2_plt[1][0],fontsize=10, rotation='vertical', 
#                 transform=axes[1,1].transAxes,fontweight='bold')
# #-------------------------------------------------------

# # ERA5 with climatology: corrected 
# axes[2,0].imshow(nh_arr2_plt[2][1],cmap = cmap_hits_grey, extent=[-180,180,45,90],norm=norm_hits_grey)    

# axes[2,1].imshow(nh_arr2_plt[2][2],cmap = cmap_miss_grey, extent=[-180,180,45,90],norm=norm_miss_grey)
# axes[2,1].text(1.02, 0.035,nh_arr2_plt[2][0],fontsize=10, rotation='vertical', 
#                 transform=axes[2,1].transAxes,fontweight='bold')
# #-------------------------------------------------------

# # Cimatology 
# axes[3,0].imshow(nh_arr2_plt[3][1],cmap = cmap_hits_grey, extent=[-180,180,45,90],norm=norm_hits_grey)    

# axes[3,1].imshow(nh_arr2_plt[3][2],cmap = cmap_miss_grey, extent=[-180,180,45,90],norm=norm_miss_grey)
# axes[3,1].text(1.02, 0.035,nh_arr2_plt[3][0],fontsize=10, rotation='vertical', 
#                 transform=axes[3,1].transAxes,fontweight='bold')               

# #-------------------------------------------------------

# # Create the colorbar for "Hits"
# cbar_hits = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_hits_grey, norm=norm_hits_grey),
#                          ax=axes[:, 0], # Applies the colorbar to the axes on the first column
#                          orientation='horizontal',
#                          fraction=0.04, # Adjusts the length of the colorbar
#                          pad=0.08, # Adjusts the distance between the colorbar and the plots
#                          aspect=40,# Adjusts the aspect ratio of the colorbar
#                          shrink=0.95) # Adjusts the length of the colorbar relative to the plot

# cbar_hits.set_label('Hits [%]', fontsize=15)
# bins_midpoints = (np.array(bins_hits[:-1]) + np.array(bins_hits[1:])) / 2

# # Corrected setting of ticks and labels for "Hits" colorbar
# cbar_hits.set_ticks(bins_midpoints)
# cbar_hits.ax.set_xticklabels(['0-10', '10-20', '20-30', '30-40', '40-50', 
#                               '50-60', '60-70', '70-80', '80-90', '90-100'], fontsize=12)

# # Create the colorbar for "Miss"
# cbar_miss = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_miss_grey, norm=norm_miss_grey),
#                          ax=axes[:, 1], # Applies the colorbar to the axes on the second column
#                          orientation='horizontal',
#                          fraction=0.04, # Adjusts the length of the colorbar
#                          pad=0.08, # Adjusts the distance between the colorbar and the plots
#                          aspect=40, # Adjusts the aspect ratio of the colorbar 
#                          shrink=0.95) # Adjusts the length of the colorbar relative to the plot

# cbar_miss.set_label('Miss [%]', fontsize=15)
# # Corrected setting of ticks and labels for "Hits" colorbar
# cbar_miss.set_ticks(bins_midpoints)
# cbar_miss.ax.set_xticklabels(['0-10', '10-20', '20-30', '30-40', '40-50', 
#                               '50-60', '60-70', '70-80', '80-90', '90-100'], fontsize=12)



# # # Add colorbar for "Hits"
# # pos = axes[3, 0].get_position()
# # cbar_ax_hits = fig.add_axes([0.125, 0.08, 0.35, 0.02])
# # cbar_hits = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_hits, norm=norm_hits),
# #                         cax=cbar_ax_hits, orientation='horizontal', pad=0.4, fraction=0.25)
# # cbar_hits.set_ticks((bins_hits[:-1] + np.diff(bins_hits) / 2))
# # cbar_hits.set_ticklabels(['0-10', '10-20', '20-30', '30-40', '40-50',
# #                         '50-60', '60-70', '70-80', '80-90', '90-100'])
# # cbar_hits.ax.tick_params(size=0)
# # cbar_hits.set_label('Hits [%]', rotation=0, labelpad=5)

# # # Add colorbar for "Miss"
# # pos = axes[3, 1].get_position()
# # cbar_ax_miss = fig.add_axes([0.545, 0.08, 0.35, 0.02])
# # cbar_miss = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_miss, norm=norm_miss),
# #                         cax=cbar_ax_miss, orientation='horizontal', pad=0.4, fraction=0.25)
# # cbar_miss.set_ticks((bins_hits[:-1] + np.diff(bins_hits) / 2))
# # cbar_miss.set_ticklabels(['0-10', '10-20', '20-30', '30-40', '40-50',
# #                         '50-60', '60-70', '70-80', '80-90', '90-100'])
# # cbar_miss.ax.tick_params(size=0)
# # cbar_miss.set_label('Miss [%]', rotation=0, labelpad=5)

# #----------------------------------------------------------------
# global
# ('E',e_glb_bskt),
# global_sim_lst_to_compute_cat_stats = [('ML-E',ml_e_glb_bskt),
#                                        ('ML-EC',ml_ec_glb_bskt),
#                                        ('ML-ECC',ml_ecc_glb_bskt),                                        
#                                        ('CLIM', climatology_glb_bskt)]

# svnem_csv = '_'.join(['global',cde_run_dte]) + '.csv'

# cat_df_out = cat_metrcs_computed(global_sim_lst_to_compute_cat_stats,gmasi_glb_bskt)

# cat_df_out.to_csv(os.path.join(path_to_put_df,svnem_csv))
# del(svnem_csv,cat_df_out)

print('done!')
#--------------------------------------------------------------------
# sh_hits_2d_rf = make_2d_hit_miss_map(rf_est_atsnw_sh,orig_atsnw_sh,'hit')
# sh_miss_2d_rf = make_2d_hit_miss_map(rf_est_atsnw_sh,orig_atsnw_sh,'miss')

# sh_hits_2d_climrf = make_2d_hit_miss_map(clim_rf_est_atsnw_sh,orig_atsnw_sh,'hit')
# sh_miss_2d_climrf = make_2d_hit_miss_map(clim_rf_est_atsnw_sh,orig_atsnw_sh,'miss')

# sh_arr2_plt = [sh_hits_2d_rf,sh_hits_2d_climrf, sh_miss_2d_rf,sh_miss_2d_climrf]

# plot_2d_hit_miss_map(sh_arr2_plt,'sh','RdYlGn_r',12)

# # save both maps
# sh_hit_svenme = '_'.join(['hits_miss_sh',cde_run_dte]) + '.png'
# plt.savefig(os.path.join(path_to_put_plots,sh_hit_svenme))
# plt.close()  
# print('done with sh')
#%%
'''
# do some zonal analysis
lons_lst = [lons.copy() for _ in range(len(rf_estimated_autosnow_bskt))] 
lats_lst = [lats.copy() for _ in range(len(rf_estimated_autosnow_bskt))] 

# make a 1d array from list of 2d precip arrays
orig_arr_1d = make_nd_array(original_autosnow_bskt,original_autosnow_bskt[0].shape[0])
rf_est_arr_1d = make_nd_array(rf_estimated_autosnow_bskt,rf_estimated_autosnow_bskt[0].shape[0])
climrf_est_arr_1d = make_nd_array(clim_rf_estimated_autosnow_bskt,clim_rf_estimated_autosnow_bskt[0].shape[0])
cor_climrf_est_arr_1d = make_nd_array(cor_clim_rf_estimated_autosnow_bskt,cor_clim_rf_estimated_autosnow_bskt[0].shape[0])
temp_clim_aarr_1d = make_nd_array(temp_clim_rf_estimated_autosnow_bskt,temp_clim_rf_estimated_autosnow_bskt[0].shape[0])

lons_arr_1d = make_nd_array(lons_lst,lons_lst[0].shape[0])
lats_arr_1d = make_nd_array(lats_lst,lats_lst[0].shape[0])

array_for_zonal = np.column_stack([orig_arr_1d, rf_est_arr_1d, climrf_est_arr_1d, 
                                  cor_climrf_est_arr_1d, temp_clim_aarr_1d,lats_arr_1d, lons_arr_1d])

array_for_zonal = array_for_zonal[~np.isnan(array_for_zonal).any(axis=1)]

array_for_zonal_df = pd.DataFrame(array_for_zonal,columns=['Original', 'ERA5 only','ERA5 + Clim', 
                                                           'cor ERA5 + Clim','Temp + Clim', 'lat', 'lon'])
gc.collect()

# for c in ['Original', 'ERA5 only','ERA5 + Clim', 'cor ERA5 + Clim','Temp + Clim']:
#     sub_zon_array_df = array_for_zonal_df.loc[:,[c,'lat','lon']]
for cs in [0,1,2,3]:

    fg,ax = plt.subplots(dpi=1000)

    if cs == 0:
        ttle = 'Water'
    elif cs == 1:
        ttle = 'Snow free land'
    elif cs == 2:
        ttle = 'Snow covered land'
    elif cs == 3:
        ttle = 'Ice'
        
    array_for_zonal_df = pd.DataFrame(array_for_zonal,columns=['Original', 'ERA5 only','ERA5 + Clim', 
                                                           'cor ERA5 + Clim','Temp + Clim', 'lat', 'lon'])

    # make a column that says that if precip is above 0.03 put 1 else 0
    array_for_zonal_df['orig_snc_cnt'] = 0
    array_for_zonal_df.loc[array_for_zonal_df['Original'] == cs,'orig_snc_cnt'] = 1

    array_for_zonal_df['era5only_snc_cnt'] = 0
    array_for_zonal_df.loc[array_for_zonal_df['ERA5 only'] == cs,'era5only_snc_cnt'] = 1

    array_for_zonal_df['era5clim_snc_cnt'] = 0
    array_for_zonal_df.loc[array_for_zonal_df['ERA5 + Clim'] == cs,'era5clim_snc_cnt'] = 1

    array_for_zonal_df['corera5clim_snc_cnt'] = 0
    array_for_zonal_df.loc[array_for_zonal_df['cor ERA5 + Clim'] == cs,'corera5clim_snc_cnt'] = 1

    array_for_zonal_df['tempclim_snc_cnt'] = 0
    array_for_zonal_df.loc[array_for_zonal_df['Temp + Clim'] == cs,'tempclim_snc_cnt'] = 1

    zonal_cnt_clss_lat =  array_for_zonal_df.groupby('lat')[['orig_snc_cnt', 'era5only_snc_cnt', 'era5clim_snc_cnt', 
                                                        'corera5clim_snc_cnt','tempclim_snc_cnt',]].sum()

    # find total number of samples per each product
    zonal_cnt_lat = array_for_zonal_df.groupby('lat')[['Original', 'ERA5 only','ERA5 + Clim', 
                                                    'cor ERA5 + Clim','Temp + Clim' ]].count()

    zonal_precip_fract_lat  = pd.DataFrame()
    zonal_precip_fract_lat['Original'] = zonal_cnt_clss_lat.loc[:,'orig_snc_cnt']/zonal_cnt_lat.loc[:,'Original']
    zonal_precip_fract_lat['ERA5 only'] = zonal_cnt_clss_lat.loc[:,'era5only_snc_cnt']/zonal_cnt_lat.loc[:,'ERA5 only']
    zonal_precip_fract_lat['ERA5 + Clim'] = zonal_cnt_clss_lat.loc[:,'era5clim_snc_cnt']/zonal_cnt_lat.loc[:,'ERA5 + Clim']
    zonal_precip_fract_lat['cor ERA5 + Clim'] = zonal_cnt_clss_lat.loc[:,'corera5clim_snc_cnt']/zonal_cnt_lat.loc[:,'cor ERA5 + Clim']
    zonal_precip_fract_lat['Temp + Clim'] = zonal_cnt_clss_lat.loc[:,'tempclim_snc_cnt']/zonal_cnt_lat.loc[:,'Temp + Clim']

    filtered_zonal_precip_fract_lat = zonal_precip_fract_lat[zonal_precip_fract_lat.index > 25]

    filtered_zonal_precip_fract_lat.plot(ax=ax)

    # for ax in axes.flatten():  
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True, bottom=True, left=True)
    ax.grid(which='major', linestyle='--', linewidth='0.5', color='grey')
    ax.set_ylabel('Fraction',fontsize=13)
    ax.set_xlabel('Latitude',fontsize=13)
    ax.set_title(ttle,fontsize=13)
    ax.set_ylim((-0.009,1.02))

    svenme = '_'.join([ttle, cde_run_dte]) + '.png'
    plt.savefig(os.path.join(path_to_put_plots,svenme))

    plt.close()

#------------------------------------------
'''
#%%
'''
# snw_pp = np.where((est_snw == 2) & (probabilities >= 0.2), est_snw,
        #         np.where(est_snw ==2, sce_mode,est_snw))

        # snw_pp_ = np.where((est_snw == 2) & (sce_mode == 2), est_snw, 
        #                 np.where(est_snw == 2, np.nan, est_snw))

        # snw_pp__ = np.where((est_snw == 2) & (snw_probab >= 0.5), 2,
        #            np.where((est_snw ==2) &  (nosnw_probab >=0.5), 1,RF_clim_est))

        # snw_pp__ = np.where((est_snw == 2) & (snw_probab >= 0.2) & (abs_diff_snw <= 20), 2,
        #            np.where((est_snw == 2) &  (nosnw_probab >=0.5) & (abs_diff_nosnw <= 20), 1,
        #            np.where(est_snw != 2),est_snw,4))

metadat = meta_autosnow.copy()
metadat.update({'width':3600,'height':1800,
                'crs': crs,
                'transform':trns})

rasterio_based_save_array_to_disk(path_to_rytgers_reproc,,metadat,sce_var)

crs_source = '+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-80 +k_0=1 +x_0=0 +y_0=0 +a=6371200 +b=6371200 +units=km' # from rutgers data

destflnme = os.path.join(path_to_rytgers_reproc,'rutgers_1988004.tif')

map_fnc.gdal_based_save_array_to_disk(destflnme,ncols_w,nrows_h,pxSIZE,
                                      x_min,y_max,crs_source,'proj4',sce_var)

dat_var = xr.open_dataarray(os.path.join(path_to_rytgers_reproc,'rutgers_1988004.tif'))
dat_var.rio.write_crs(cc, inplace=True)

data_var_wgs = dat_var.rio.reproject(cc)#,  # set the shape as the autosnow data shape
                # resampling=Resampling.mode)

# resample data spatially to autosnow spatial resolution (0.1 deg for our study case)
dat_var_res = data_var_wgs.rio.reproject(
data_var_wgs.rio.crs,
shape=meta_shp, # set the shape as the autosnow data shape
resampling=Resampling.mode,)   

sce_var = np.flipud(dat_var_res)

# Select a specific location along the spatial dimensions
data_var_1d = data_var.isel(y=0, x=0)

# Set the spatial dimensions explicitly
data_var_1d.rio.set_spatial_dims('x', 'y', inplace=True)

# Set the CRS directly on the DataArray
target_crs = 'EPSG:4326'
new_shape = (1800, 3600)
data_var.rio.write_crs(crs_source, inplace=True)

resampled_data = data_var.rio.reproject(
    dst_crs=target_crs,
    shape=new_shape,
    resampling=Resampling.mode  # You can choose other methods like 'nearest', 'cubic', etc.
)

# Reproject the DataArray
data_var_wgs = data_var.rio.reproject(cc,  # set the shape as the autosnow data shape
                resampling=Resampling.mode)

# Assuming 'snow_cover_extent' is multidimensional, you may want to select a specific dimension
# or flatten it before reprojecting.
data_var = ds_rutgers.snow_cover_extent.sel(time=gettme, method='nearest')

lonx = ds_rutgers.x.values
laty = ds_rutgers.y.values
# Select a specific location along the spatial dimensions
data_var_1d = data_var.isel(latitude=0, longitude=0)

# Set the spatial dimensions explicitly
data_var_1d.rio.set_spatial_dims('lon', 'lat', inplace=True)

# Set the CRS directly on the DataArray
data_var_1d.rio.write_crs(cc, inplace=True)

# Reproject the DataArray
data_var_wgs = data_var_1d.rio.reproject(cc)

data_var.rio.write_crs(cc, inplace=True)  # Set the CRS directly on the original DataArray

data_var_wgs = data_var.rio.reproject(cc)

data_var_wgs = data_var.rio.reproject(cc)

# Or flatten the variable
data_var_flat = ds_rutgers.snow_cover_extent.isel(time=0).stack()

# Assuming data_var_flat is a Pandas Series with a MultiIndex
data_var_flat_da = data_var_flat.to_xarray()

data_var_flat_da = xr.DataArray.from_series(data_var_flat)

# Reproject the DataArray
data_var_wgs = data_var_flat_da.rio.reproject(cc)


data_var_wgs = data_var_flat.rio.reproject(cc)



#%%%



def generate_years_list(start_year: int, end_year: int) -> list:
    """
    Generate a list of years from the start year to the end year, inclusive.
    
    Parameters:
    - start_year (int): The starting year.
    - end_year (int): The ending year.
    
    Returns:
    - list: A list of years from start_year to end_year, inclusive.
    """
    return list(range(start_year, end_year + 1))

# Example usage of the function
start_year_example = 1980
end_year_example = 1988

# Generate the list of years for the example
example_years_list = generate_years_list(start_year_example, end_year_example)

print(example_years_list)
'''
# f,ax = plt.subplots(3,4, figsize=(20,10), gridspec_kw={'wspace':0.25,'hspace':0.15}, 
#                     tight_layout=True) # sharex=True, 

# x_limits = [(min(area_extent_kmsrqd['orig_autsnw_estimated_wtr_total_area'].min(), 
#                  area_extent_kmsrqd['rf_autsnw_estimated_wtr_total_area'].min(), 
#                  area_extent_kmsrqd['climrf_autsnw_estimated_wtr_total_area'].min(),
#                  area_extent_kmsrqd['temp_climrf_autsnw_estimated_wtr_total_area'].min()),

#              max(area_extent_kmsrqd['orig_autsnw_estimated_wtr_total_area'].max(),
#                  area_extent_kmsrqd['rf_autsnw_estimated_wtr_total_area'].max(),
#                  area_extent_kmsrqd['climrf_autsnw_estimated_wtr_total_area'].max(),
#                  area_extent_kmsrqd['temp_climrf_autsnw_estimated_wtr_total_area'].max())), 
            
#              (min(area_extent_kmsrqd['orig_autsnw_estimated_snfr_total_area'].min(),
#                   area_extent_kmsrqd['rf_autsnw_estimated_snfr_total_area'].min(),
#                   area_extent_kmsrqd['climrf_autsnw_estimated_snfr_total_area'].min(),
#                   area_extent_kmsrqd['temp_climrf_autsnw_estimated_snfr_total_area'].min()), 
            
#              max(area_extent_kmsrqd['orig_autsnw_estimated_snfr_total_area'].max(),
#                  area_extent_kmsrqd['rf_autsnw_estimated_snfr_total_area'].max(),
#                  area_extent_kmsrqd['climrf_autsnw_estimated_snfr_total_area'].max(),
#                  area_extent_kmsrqd['temp_climrf_autsnw_estimated_snfr_total_area'].max())), 

#              (min(area_extent_kmsrqd['orig_autsnw_estimated_snc_total_area'].min(),
#                   area_extent_kmsrqd['rf_autsnw_estimated_snc_total_area'].min(),
#                   area_extent_kmsrqd['climrf_autsnw_estimated_snc_total_area'].min(),
#                   area_extent_kmsrqd['temp_climrf_autsnw_estimated_snc_total_area'].min()), 

#              max(area_extent_kmsrqd['orig_autsnw_estimated_snc_total_area'].max(),
#                  area_extent_kmsrqd['rf_autsnw_estimated_snc_total_area'].max(),
#                  area_extent_kmsrqd['climrf_autsnw_estimated_snc_total_area'].max(),
#                  area_extent_kmsrqd['temp_climrf_autsnw_estimated_snc_total_area'].max())), 

#              (min(area_extent_kmsrqd['orig_autsnw_estimated_ice_total_area'].min(),
#                   area_extent_kmsrqd['rf_autsnw_estimated_ice_total_area'].min(),
#                   area_extent_kmsrqd['climrf_autsnw_estimated_ice_total_area'].min(),
#                   area_extent_kmsrqd['temp_climrf_autsnw_estimated_ice_total_area'].min()), 

#              max(area_extent_kmsrqd['orig_autsnw_estimated_ice_total_area'].max(),
#                  area_extent_kmsrqd['rf_autsnw_estimated_ice_total_area'].max(),
#                  area_extent_kmsrqd['climrf_autsnw_estimated_ice_total_area'].max(),
#                  area_extent_kmsrqd['temp_climrf_autsnw_estimated_ice_total_area'].max()))] 

# # # Share axes column-wise
# # for colmn in range(4):  # Assuming 4 columns based on your x_limits
# #     for ro in range(1, 3):  # Skip the first row, start with the second
# #         ax[ro, colmn].sharex(ax[0, colmn])
# #         ax[ro, colmn].sharey(ax[0, colmn])

# for i in enumerate(cl2plt):
#     ax_ro = i[0]
#     for ii in enumerate(i[1]):
#         ax_col = ii[0]

#         if ax_col == 0:
#             ttle = 'Water'
#         elif ax_col == 1:
#             ttle = 'Snow free land'
#         if ax_col == 2:
#             ttle = 'Snow covered land'
#         if ax_col == 3:
#             ttle = 'Ice'

#         if ax_ro == 0:
#             mfcl,mecl = colors[0], 'k'
#             prdct = 'ERA5 only'
#         elif ax_ro == 1:
#             mfcl,mecl = colors[1], 'r'
#             prdct = 'ERA5 with climatology'
#         elif ax_ro == 2:
#             mfcl,mecl = colors[4], 'k'
#             prdct = 'Climatology'
        

#         # # Set the axes to log scale here
#         ax[ax_ro, ax_col].set_xlim(x_limits[ax_col])
#         ax[ax_ro, ax_col].set_ylim(x_limits[ax_col])       

#         ax[ax_ro,ax_col].yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))
#         ax[ax_ro,ax_col].xaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))        

#         xvar,yvar = area_extent_kmsrqd[ii[1][1]], area_extent_kmsrqd[ii[1][0]]       

#         # br = round(bias_ratio(xvar,yvar),3)
#         rmse = round(rmsqe(xvar,yvar),2)
#         # pcor = round(r_sqr(xvar,yvar),2)
#         mape = round(he.mape(yvar,xvar,remove_neg=True),2) # mean abs % error

#         ax[ax_ro,ax_col].plot(xvar, yvar,markersize=3, marker='o', linestyle='none',
#                               markerfacecolor=mfcl, markeredgecolor=mecl, markeredgewidth=0.2)    

#         # Ensure the same limits for x and y axes
#         # ax[ax_ro, ax_col].autoscale(False)  # Disable autoscaling
#         # ax[ax_ro, ax_col].set_aspect('equal', 'box')  # Set equal scaling by changing aspect ratio
#         lims = [
#             np.min([ax[ax_ro, ax_col].get_xlim()[0], ax[ax_ro, ax_col].get_ylim()[0]]),  # min of both axes
#             np.max([ax[ax_ro, ax_col].get_xlim()[1], ax[ax_ro, ax_col].get_ylim()[1]]),  # max of both axes
#         ]
#         ax[ax_ro, ax_col].set_xlim(lims)
#         ax[ax_ro, ax_col].set_ylim(lims)

#         # Now plot the 1:1 line
#         ax[ax_ro, ax_col].plot(lims, lims, 'r--', linewidth=2.5, alpha=0.75, zorder=10)  # 1:1 line

#         # add the metrics to the plot as text
#         txt = "RMSE = {0:.0f}\nMAPE = {1:.2f}".format(*[rmse,mape])  

#         ax[ax_ro,ax_col].text(0.11, 0.85, txt, horizontalalignment='left', 
#             verticalalignment='center', transform = ax[ax_ro,ax_col].transAxes, 
#             fontsize=12, bbox=dict(boxstyle='round', facecolor='silver', alpha=0.5))

#      # Hide x-tick labels for all but the bottom row of the subplot in each column
#         if ax_ro < 2:
#             for label in ax[ax_ro, ax_col].get_xticklabels():
#                 label.set_visible(False)
        
#         if [ax_ro,ax_col] in [[0,3],[1,3],[2,3]]:
#             ax[ax_ro,ax_col].text(1.02, 0.15,prdct,fontsize=12,
#             rotation='vertical', transform=ax[ax_ro,ax_col].transAxes) # ,fontweight='bold'

#         if [ax_ro,ax_col] in [[0,0],[0,1],[0,2],[0,3]]:
#             ax[ax_ro,ax_col].set_title(ttle,fontsize=15)     

# for a in ax.flatten():
#     a.tick_params(axis='x', rotation=45)

#     a.minorticks_on()
#     a.tick_params(which='both', direction='in', top=True, right=True, bottom=True, left=True,labelsize=13)
#     a.grid(which='major', linestyle='--', linewidth='0.5', color='grey')   
        # # Hide y-tick labels for all but the leftmost subplot of each row
        # if ax_col > 0:
        #     for label in ax[ax_ro, ax_col].get_yticklabels():
        #         label.set_visible(False)



# # global
# glob_data_lst = [original_autosnow_bskt, rf_estimated_autosnow_bskt]

# eval_df_global = cat_evaluate(glob_data_lst,x_shp)
# gc.collect()
#------------------------------------    
# # RF_with_clim_categorical_stats_global
# svnem_csv = '_'.join(['RF_with_ERA5_only_data_categorical_stats_global',cde_run_dte]) + '.csv'
# eval_df_global.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_global = eval_df_global[clms2plt]

# svnem_plt = '_'.join(['RF_with_ERA5_only_data_categorical_stats_global',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_global,15,13,'Global')
# plt.tight_layout()

# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))
# plt.close()

# #-----------------------------------------------------------------------------------------
# # nh
# nh_data_lst = [orig_atsnw_nh, rf_est_atsnw_nh]

# eval_df_nh = cat_evaluate(nh_data_lst,x_shp)
# gc.collect()
# # RF_with_clim_categorical_stats_nh
# svnem_csv = '_'.join(['RF_with_ERA5_only_data_categorical_stats_nh',cde_run_dte]) + '.csv'
# eval_df_nh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_nh = eval_df_nh[clms2plt]

# svnem_plt = '_'.join(['RF_with_ERA5_only_data_categorical_stats_nh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_nh,15,13,'Northern Hemisphere')
# plt.tight_layout()
# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))
# plt.close()
# gc.collect()

# #-----------------------------------------------------

# # sh
# sh_data_lst = [orig_atsnw_sh, rf_est_atsnw_sh]

# eval_df_sh = cat_evaluate(sh_data_lst,x_shp)
# gc.collect()
# # RF_with_clim_categorical_stats_sh
# svnem_csv = '_'.join(['RF_with_ERA5_only_data_categorical_stats_sh',cde_run_dte]) + '.csv'
# eval_df_sh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_sh = eval_df_sh[clms2plt]

# svnem_plt = '_'.join(['RF_with_ERA5_only_data_categorical_stats_sh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_sh,15,13, 'Southern Hemisphere')
# plt.tight_layout()

# plt.close()
# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))

# #-----------------------------------------------------------------------------------------

# # seasons: NH
# # winter
# nh_winter_data_lst = [nh_winter_orig_atsnw, nh_winter_rf_atsnw]

# winter_eval_df_nh = cat_evaluate(nh_winter_data_lst,x_shp)
# gc.collect()
# # RF_with_clim_categorical_stats_winter_nh
# svnem_csv = '_'.join(['RF_with_ERA5_only_data_categorical_stats_winter_nh',cde_run_dte]) + '.csv'
# winter_eval_df_nh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_winter_nh = winter_eval_df_nh[clms2plt]

# svnem_plt = '_'.join(['RF_with_ERA5_only_data_categorical_stats_winter_nh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_winter_nh,15,13,'NH - DJF')
# plt.tight_layout()

# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))
# plt.close()
# gc.collect()
# #--------------------
# # spring
# nh_spring_data_lst = [nh_spring_orig_atsnw, nh_spring_rf_atsnw]

# spring_eval_df_nh = cat_evaluate(nh_spring_data_lst,x_shp)
# gc.collect()
# # RF_with_clim_categorical_stats_spring_nh
# svnem_csv = '_'.join(['RF_with_ERA5_data_only_categorical_stats_spring_nh',cde_run_dte]) + '.csv'
# spring_eval_df_nh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_spring_nh = spring_eval_df_nh[clms2plt]

# svnem_plt = '_'.join(['RF_with_ERA5_data_only_categorical_stats_spring_nh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_spring_nh,15,13,'NH - MAM')
# plt.tight_layout()

# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))
# plt.close()

# #--------------------
# # summer
# nh_summer_data_lst = [nh_summer_orig_atsnw, nh_summer_rf_atsnw]
# summer_eval_df_nh = cat_evaluate(nh_summer_data_lst,x_shp)
# gc.collect()
# # RF_with_clim_categorical_stats_summer_nh
# svnem_csv = '_'.join(['RF_with_ERA5_data_only_categorical_stats_summer_nh',cde_run_dte]) + '.csv'
# summer_eval_df_nh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_summer_nh = summer_eval_df_nh[clms2plt]
# svnem_plt = '_'.join(['RF_with_ERA5_data_only_categorical_stats_summer_nh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_summer_nh,15,13,'NH - JJA')
# plt.tight_layout()

# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))
# plt.close()
# gc.collect()

# #--------------------
# # autumn
# nh_autumn_data_lst = [nh_autumn_orig_atsnw, nh_autumn_rf_atsnw]
# autumn_eval_df_nh = cat_evaluate(nh_autumn_data_lst,x_shp)
# gc.collect()
# # RF_with_clim_categorical_stats_autumn_nh
# svnem_csv = '_'.join(['RF_with_ERA5_data_only_categorical_stats_autumn_nh',cde_run_dte]) + '.csv'
# autumn_eval_df_nh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_autumn_nh = autumn_eval_df_nh[clms2plt]
# svnem_plt = '_'.join(['RF_with_ERA5_data_only_categorical_stats_autumn_nh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_autumn_nh,15,13,'NH - SON')
# plt.tight_layout()

# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))
# plt.close()
# #-----------------------------------------------------------------------------------------

# # seasons: SH
# # winter
# sh_winter_data_lst = [sh_winter_orig_atsnw, sh_winter_rf_atsnw]

# winter_eval_df_sh = cat_evaluate(sh_winter_data_lst,x_shp)
# gc.collect()
# # RF_with_clim_categorical_stats_winter_sh
# svnem_csv = '_'.join(['RF_with_ERA5_data_only_categorical_stats_winter_sh',cde_run_dte]) + '.csv'
# winter_eval_df_sh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_winter_sh = winter_eval_df_sh[clms2plt]

# svnem_plt = '_'.join(['RF_with_ERA5_data_only_categorical_stats_winter_sh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_winter_sh,15,13,'SH - JJA')
# plt.tight_layout()

# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))
# plt.close()

# gc.collect()
# #--------------------
# # spring
# sh_spring_data_lst = [sh_spring_orig_atsnw, sh_spring_rf_atsnw]

# spring_eval_df_sh = cat_evaluate(sh_spring_data_lst,x_shp)
# gc.collect()
# # RF_with_clim_categorical_stats_spring_sh
# svnem_csv = '_'.join(['RF_with_ERA5_data_only_categorical_stats_spring_sh',cde_run_dte]) + '.csv'
# spring_eval_df_sh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_spring_sh = spring_eval_df_sh[clms2plt]

# svnem_plt = '_'.join(['RF_with_ERA5_data_only_categorical_stats_spring_sh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_spring_sh,15,13,'SH - SON')
# plt.tight_layout()

# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))
# plt.close()
# #--------------------
# # summer
# sh_summer_data_lst = [sh_summer_orig_atsnw, sh_summer_rf_atsnw]
# summer_eval_df_sh = cat_evaluate(sh_summer_data_lst,x_shp)
# gc.collect()
# # RF_with_clim_categorical_stats_summer_sh
# svnem_csv = '_'.join(['RF_with_ERA5_only_data_categorical_stats_summer_sh',cde_run_dte]) + '.csv'
# summer_eval_df_sh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_summer_sh = summer_eval_df_sh[clms2plt]
# svnem_plt = '_'.join(['RF_with_ERA5_only_data_categorical_stats_summer_sh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_summer_sh,15,13,'SH - DJF')
# plt.tight_layout()

# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))
# plt.close()
# gc.collect()

# #--------------------
# # autumn
# sh_autumn_data_lst = [sh_autumn_orig_atsnw, sh_autumn_rf_atsnw]
# autumn_eval_df_sh = cat_evaluate(sh_autumn_data_lst,x_shp)
# gc.collect()
# # RF_with_clim_categorical_stats_autumn_sh
# svnem_csv = '_'.join(['RF_with_ERA5_only_data_categorical_stats_autumn_sh',cde_run_dte]) + '.csv'
# autumn_eval_df_sh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_autumn_sh = autumn_eval_df_sh[clms2plt]
# svnem_plt = '_'.join(['RF_with_ERA5_only_data_categorical_stats_autumn_sh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_autumn_sh,15,13,'SH - MAM')
# plt.tight_layout()

# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))
# plt.close()
# gc.collect()

# print('done!')
#-----------------------------------------------------------------------------------------------
# def plot_2d_hit_miss_map(cat_arr,area,clrmp,nbs):
#     # from mpl_toolkits.axes_grid1 import make_axes_locatable

#     projc = ccrs.PlateCarree()

#     if area == 'global':
#         img_extent = (-180, 180, -90, 90)
#         lblsze = 12
#         yclse = -0.2
#         # nro,ncol = 1,2
#         wspce = 0.000005
#         hspce = 0.08
#     elif area == 'nh':
#         img_extent = [-180,180,45,90]
#         lblsze = 10
#         yclse = -0.13
#         # nro,ncol = 2,1
#         wspce = 0.08
#         hspce = 0.001
#     elif area == 'sh':
#         img_extent = [-180,180,-90,-45]
#         lblsze = 10
#         yclse = -0.13
#         # nro,ncol = 2,1
#         wspce = 0.08
#         hspce = 0.000005

#     minss,maxs = 0,365
#     # min(np.nanmin(hit_arr),np.nanmin(miss_arr)),max(np.nanmax(hit_arr),np.nanmax(miss_arr))
#     levels = MaxNLocator(nbins=nbs).tick_values(minss,maxs)
#     cmap = matplotlib.cm.get_cmap(clrmp,nbs)

#     norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

#     fig, axes = plt.subplots(2,2, figsize=(10, 4),sharey=True,sharex=True, 
#                              subplot_kw={'projection': projc}, dpi=1000,
#                             ) #  gridspec_kw={'hspace':hspce,'wspace':wspce}
#     plt.subplots_adjust(hspace=hspce,wspace=wspce)

#     # Hits plot
#     # axes[0,0].coastlines()

#     axes[0,0].imshow(cat_arr[0],cmap =cmap, extent=img_extent,origin="upper",transform=projc)
#     axes[0,0].set_title('Hits', fontsize =15)

#     # axes[1,0].coastlines()
#     axes[1,0].imshow(cat_arr[1],cmap =cmap, extent=img_extent,origin="upper",transform=projc)

#     axes[0,0].text(yclse, 0.55, 'ERA5 only', va='bottom', ha='center',
#                 rotation='vertical', rotation_mode='anchor',
#                 fontsize=lblsze,transform=axes[0,0].transAxes)# 1.03, 0.5
#     axes[1,0].text(yclse, 0.55, 'Clim + ERA5', va='bottom', ha='center',
#                 rotation='vertical', rotation_mode='anchor',
#                 fontsize=lblsze,transform=axes[1,0].transAxes)# 1.03, 0.5
    
#     gls_ax = axes[0,0].gridlines(crs=projc,color='grey', linestyle='--', 
#                                lw = 0.35, draw_labels={ "left": "y"}) #"bottom": "x",
#     gls_ax.xlabel_style={'size':lblsze}   
#     gls_ax.ylabel_style={'size':lblsze} # ,'rotation':45

#     gls_ax = axes[1,0].gridlines(crs=projc,color='grey', linestyle='--', 
#                                lw = 0.35, draw_labels={ "bottom": "x","left": "y"}) #
#     gls_ax.xlabel_style={'size':lblsze}   
#     gls_ax.ylabel_style={'size':lblsze} # ,'rotation':45

#     #-------------------------------------------------------

#     # Miss plot
#     # axes[0,1].coastlines()
#     ax_plt = axes[0,1].imshow(cat_arr[2],cmap =cmap, extent=img_extent,origin="upper",transform=projc)
#     axes[0,1].set_title('Miss', fontsize = 15)

#     # axes[0,1].text(1.03, 0.5, 'Miss', va='bottom', ha='center',
#     #         rotation='vertical', rotation_mode='anchor',
#     #         fontsize=15,transform=axes[1].transAxes)
    
#     gls_ax = axes[0,1].gridlines(crs=projc,color='grey', linestyle='--', 
#                                lw = 0.35) # , draw_labels={"bottom": "x", "left": "y"}
#     gls_ax.xlabel_style={'size':12}   
#     gls_ax.ylabel_style={'size':12} # ,'rotation':45

#     # axes[1,1].coastlines()
#     ax_plt = axes[1,1].imshow(cat_arr[3],cmap =cmap, extent=img_extent,origin="upper",transform=projc)

#     # axes[1,1].text(1.03, 0.5, 'Miss', va='bottom', ha='center',
#             # rotation='vertical', rotation_mode='anchor',
#             # fontsize=15,transform=axes[1,1].transAxes)
    
#     gls_ax = axes[1,1].gridlines(crs=projc,color='grey', linestyle='--', 
#                                lw = 0.35, draw_labels={"bottom": "x"}) # , "left": "y"
#     gls_ax.xlabel_style={'size':lblsze}   
#     gls_ax.ylabel_style={'size':lblsze} # ,'rotation':45

#     # Create a ScalarMappable to set the colorbar
#     # Create a custom normalization instance for the color bar
#     norm3 = Normalize(vmin=minss, vmax=maxs)
#     sm = ScalarMappable(cmap=cmap) # , norm=norm3
#     sm.set_array([]) 

#     tck_intvl = (maxs-minss)/nbs

#     # divider = make_axes_locatable(axes[1,1])
#     # cax = divider.append_axes("bottom", size="5%", pad=0.1)

#     cbar = plt.colorbar(mappable = sm,ax = axes,location='bottom',pad=0.08,
#                         ticks=np.arange(minss, maxs+1,tck_intvl), norm=norm3,
#                         format=FuncFormatter(custom_formatter),fraction=0.1,extend = 'max')
#     cbar.norm.vmin = minss
#     cbar.norm.vmax = maxs 

#     # Fine-tune layout
#     # plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9, wspace=0.01, hspace=0.01)

#     # plt.tight_layout()

#-----------------------------------------------
# do pixel-wise hit/miss calculate
# glb_hits_2d_rf = make_2d_hit_miss_map(rf_estimated_autosnow_bskt,original_autosnow_bskt,'hit')
# glb_miss_2d_rf = make_2d_hit_miss_map(rf_estimated_autosnow_bskt,original_autosnow_bskt,'miss')

# glb_hits_2d_climrf = make_2d_hit_miss_map(clim_rf_estimated_autosnow_bskt,original_autosnow_bskt,'hit')
# glb_miss_2d_clim_rf = make_2d_hit_miss_map(clim_rf_estimated_autosnow_bskt,original_autosnow_bskt,'miss')

# glb_arr2plt = [glb_hits_2d_rf,glb_hits_2d_climrf, glb_miss_2d_rf,glb_miss_2d_clim_rf]
# plot_2d_hit_miss_map(glb_arr2plt,'global','RdYlGn_r',12)
# nh_hit_svenme = '_'.join(['hits_miss_global',cde_run_dte]) +'.png'
# plt.savefig(os.path.join(path_to_put_plots,nh_hit_svenme))
# plt.close() 

print('done with global')


#----------------------------------------------------------------
#%%
# for l in sorted(all_autosnow_files[:155]):   

#     yr_DOY = os.path.basename(l).split('_')[4]

#     yr  = int(os.path.basename(l).split('_')[4][:4])#int(l[:4])

#     doy_ = os.path.basename(l).split('_')[4][-3:]

#     doY = int(os.path.basename(l).split('_')[4][-3:]) #int(l[-3:])

#     doy = os.path.basename(l).split('_')[4][-3:]

#     dt = datetime(yr, 1, 1) + timedelta(doY - 1) # datetime.

#     dt_sve = dt.strftime('%Y%m%d') 

#     # Create a datetime object
#     date_time = pd.to_datetime(f'{yr}-{doY}', format='%Y-%j')

#     mnth = pd.to_datetime(dt_sve).month
#     #------------------------------------------------------
#     # define and check for existence of files to read
#     # RF estimated using only ERA5
#     ml_e_read_nme = os.path.join(path_to_estimated_autosnw,'_'.join(['RF_estimated_autosnow_using_only_ERA5_data',
#                                                                      yr_DOY,'0.1deg_wgs']) + '.tif')
#     # RF estimated using Climatology
#     ml_ec_read_nme = os.path.join(path_to_estimated_autosnw,'_'.join(['RF_estimated_autosnow_using_alldataclim',
#                                                                       yr_DOY,'0.1deg_wgs']) + '.tif')
#     # RF estimated using temp dependent Climatology
#     climatology_read_nme = os.path.join(path_to_clim_only_autosnw_estimated,'_'.join(['alldata_1992_2022_clim_subsetted_by_airTemp',
#                                                                                       yr_DOY,'0.1deg_wgs']) + '.tif')

#     e_read_nme = os.path.join(path_to_era5_based_snowice,'_'.join(['ERA5_seaice-snowcover_based_autosnow_labeled_class_data',
#                                                                    yr_DOY]) + '.tif')
#     # ERA5_seaice-snowcover_based_autosnow_labeled_class_data_1989020.tif

#     #------------------------------------------------------

#     if ((os.path.isfile(ml_e_read_nme)) and (os.path.isfile(ml_ec_read_nme)) and \
#         (os.path.isfile(climatology_read_nme)) and (os.path.isfile(e_read_nme))):

#         date_time_ = date_time

#         # read  data    
#         ml_e_estimated = xr.open_dataarray(ml_e_read_nme)
#         ml_e_estimated_arr = ml_e_estimated.data[0,:,:]
#         ml_e_estimated_arr = np.where(ml_e_estimated_arr > 3,np.nan,ml_e_estimated_arr)
        
#         ml_ec_estimated = xr.open_dataarray(ml_ec_read_nme)
#         ml_ec_estimated_arr = ml_ec_estimated.data[0,:,:]
#         ml_ec_estimated_arr = np.where(ml_ec_estimated_arr > 3,np.nan,ml_ec_estimated_arr)

#         climatology_estimated = xr.open_dataarray(climatology_read_nme)
#         climatology_estimated_arr = climatology_estimated.data[0,:,:]
#         climatology_estimated_arr = np.where(climatology_estimated_arr > 3,np.nan,climatology_estimated_arr)   

#         e_estimated = xr.open_dataarray(e_read_nme)
#         e_estimated_arr = e_estimated.data[0,:,:]

#         # the original autosnow data
#         gmais_dat = xr.open_dataarray(l) 
#         gmasi_dat_array = gmais_dat.data[0,:,:]
#         gmasi_dat_array = np.where(gmasi_dat_array > 3,np.nan,gmasi_dat_array)
#         y_shp,x_shp = gmasi_dat_array.shape[0],gmasi_dat_array.shape[1] 

#         ml_ecc_read_nme = os.path.join(path_to_estimated_autosnw,
#                                                    '_'.join(['corrected_RF_estimated_autosnow_using_alldataclim',
#                                                              yr_DOY,'0.1deg_wgs']) + '.tif')
#         if os.path.isfile( ml_ecc_read_nme):

#             ml_ecc_estimated = xr.open_dataarray(ml_ecc_read_nme)
#             ml_ecc_estimated_arr = ml_ecc_estimated.data[0,:,:]
#             ml_ecc_estimated_arr = np.where(ml_ecc_estimated_arr > 3,np.nan,ml_ecc_estimated_arr)

#         else:
#             print('corrected file doesnt exist, so I am doing the correction')
#             # do corrected the RF using alldata climatology data
#             #------------------------------------------------------

#             # Post process the RF-based autosnow using ERA5 and autosnow climatology data
#             # this implementation uses Rutgers data, in which the week is on Monday
#             # and it includes data from Tuesday through to Monday
#             input_date_str = date_time.strftime('%Y-%m-%d')
#             relevant_monday = find_relevant_monday(input_date_str) # which is the week in which our current date falls in

#             # Rutgers snow cover for the week having our date
#             snc_relev_mond = ds_rutgers.snow_cover_extent.sel(time=pd.to_datetime(relevant_monday))
#             snc_relev_mond = snc_relev_mond.isel(lat=slice(None,None,-1)).values
#             snc_relev_mond[901:,:] = np.nan

#             # finding adjacent Mondays in previous years
#             input_date_str_m = relevant_monday.strftime('%Y-%m-%d')
#             past_years = sorted(list(range(1980, relevant_monday.year)),reverse=True) 
#             monday_dates = find_monday_of_same_week_past_years(input_date_str_m, past_years)
#             # these represent adjacent weeks in previous years
#             monday_dates_ = pd.to_datetime(monday_dates,format='%Y-%m-%dT%H:%M:%S') # pd.DatetimeIndex(
#             #[pd.to_datetime(x).date(format='%Y-%m-%d') for x in monday_dates] # .strftime('%Y-%m-%d')

#             time_series = pd.Series(ds_rutgers.time.to_index())

#             # Initialize an empty list to hold indices of matching dates
#             matching_indices = []

#             # Iterate over monday_dates to find matches in the dataset's time index
#             for dte in monday_dates_:
#                 matching = time_series[time_series == dte]
#                 if not matching.empty:
#                     # Collect indices of matching dates
#                     matching_indices.extend(matching.index.tolist())

#             # Now, use matching_indices to select or filter data in ds_rutgers
#             data_var = ds_rutgers.sel(time=ds_rutgers.time[matching_indices]) #if time is a coordinate   

#             # flip data along the vertical axis, i.e up down
#             data_var = data_var.isel(lat=slice(None, None, -1))

#             # get the snow cover data representing adjacent weeks in previous years 
#             # 0 = no snow, 1 = snow covered pixel over land only
#             snc_var = data_var.snow_cover_extent.values
#             snc_var[:,901:,:] = np.nan # the data is only valid in the Northern hemisphere
#             # get the most common classification of each pixel based on adjacent weeks in previous years data
#             sce_mode_ = np.apply_along_axis(lambda a: stats.mode(a,nan_policy='omit')[0][0], 0, snc_var)
#             sce_mode = sce_mode_.copy()
#             # if most coomon is 1, relable to 2 (snow covered land in autosnow label)
#             sce_mode = np.where((sce_mode == 1) & (lsm == 1), 2 ,sce_mode) # 
#             # if most common is 0 over land, relable to 1 (snow free land in autosnow label)
#             sce_mode = np.where(((sce_mode == 0) & (lsm == 1)), 1 ,sce_mode)
#             sce_mode = sce_mode.astype(float)
#             sce_mode[901:, :] = np.nan  # restrict the data to NH

#             # Here, we calculate the prbability of a pixel having no snow or snow based on the past years data        
#             # Create a boolean mask where the snow or no snow condition is met 
#             mask_snw = (snc_var == 1)
#             mask_nosnw= (snc_var == 0)
#             # Sum the occurrences of value_to_check across the first dimension (num_arrays), ignoring np.nan
#             count_snw = np.nansum(mask_snw, axis=0)
#             count_snw = count_snw.astype(float)  # Convert to float
#             count_snw[901:, :] = np.nan  # restrict the data to NH

#             count_nosnw = np.nansum(mask_nosnw, axis=0)
#             count_nosnw = count_nosnw.astype(float)  # Convert to float
#             count_nosnw[901:, :] = np.nan  # Now you can assign np.nan

#             # Calculate the probability for each pixel, considering only the non-nan values
#             snw_probab = count_snw / snc_var.shape[0]
#             nosnw_probab = count_nosnw / snc_var.shape[0]

#             past_years_week_ave_temp_snw_land_lst = []
#             past_years_week_ave_temp_nosnw_land_lst = []
#             # the tempearture analysis
#             for t in data_var.time.values:
#                 input_t = pd.to_datetime(t).strftime('%Y-%m-%d')

#                 week_dates = get_aggregated_dates(input_t)

#                 week_ave_temp_lst = []
#                 # find and read all surface temperature on these dates  _1983_
#                 for tt in week_dates:
#                     year_tt = pd.to_datetime(tt).year

#                     day_in_week = pd.to_datetime(tt)

#                     flenme_temp = '_'.join(['skin_temperature',str(year_tt),'daily_mean.nc'])

#                     surface_temps = xr.open_dataset(os.path.join(path_to_era5_vars,flenme_temp))

#                     surface_temp = surface_temps.skt.sel(time=day_in_week).values

#                     week_ave_temp_lst.append(surface_temp)

#                 week_stck = np.dstack(week_ave_temp_lst)
#                 week_ave_temp = np.nanmean(week_stck,axis=2)

#                 week_snw_cv = data_var.snow_cover_extent.sel(time = t).values

#                 week_snw_cv_ave_temp_land = np.where((week_snw_cv == 1) & (lsm == 1),week_ave_temp,np.nan)

#                 week_nosnw_cv_ave_temp_land = np.where((week_snw_cv == 0) & (lsm == 1),week_ave_temp,np.nan)

#                 past_years_week_ave_temp_snw_land_lst.append(week_snw_cv_ave_temp_land)
#                 past_years_week_ave_temp_nosnw_land_lst.append(week_nosnw_cv_ave_temp_land)
            
#             past_years_week_ave_temp_snw_land = np.nanmean(past_years_week_ave_temp_snw_land_lst)

#             past_years_week_ave_temp_nosnw_land = np.nanmean(past_years_week_ave_temp_nosnw_land_lst)

#             # find the abs of the days temp with the diff temp conditions
#             day_flenme_temp = '_'.join(['skin_temperature',str(date_time.year),'daily_mean.nc'])

#             day_surface_temps = xr.open_dataset(os.path.join(path_to_era5_vars,day_flenme_temp))
#             day_surface_temp = day_surface_temps.skt.sel(time=pd.to_datetime(date_time)).values

#             abs_diff_snw = np.abs(day_surface_temp - past_years_week_ave_temp_snw_land)

#             abs_diff_nosnw = np.abs(day_surface_temp - past_years_week_ave_temp_nosnw_land)
            
#             # the post processing
#             RF_clim_est = ml_ec_estimated_arr.copy()

#             est_snw = ml_ec_estimated_arr.copy()

#             est_snw[901:, :] = np.nan  

#             est_snw_bol = (est_snw == 2)

#             est_snw_bol = est_snw_bol.astype(float)

#             est_snw_bol[901:, :] = np.nan         

#             # # save some intermdiate files for checking
#             # meta_autosnow_cpy = meta_autosnow.copy()
#             # meta_autosnow_cpy.update({'dtype':np.float32})
#             # rasterio_based_save_array_to_disk(path_to_put_intermediate_files,'nh_snow_climatology.tif',
#             #                                           meta_autosnow_cpy,snw_probab)
#             # rasterio_based_save_array_to_disk(path_to_put_intermediate_files,'nh_nosnow_climatology.tif',
#             #                                           meta_autosnow_cpy,nosnw_probab)
#             # rasterio_based_save_array_to_disk(path_to_put_intermediate_files,'nh_abs_diff_snw.tif',
#             #                                           meta_autosnow_cpy,abs_diff_snw)
#             # rasterio_based_save_array_to_disk(path_to_put_intermediate_files,'nh_abs_diff_nosnw.tif',
#             #                                           meta_autosnow_cpy,abs_diff_nosnw)
            

#             snw_pp__ = np.full(est_snw.shape, np.nan)        

#             snw_pp__ = np.where(est_snw_bol,4,snw_pp__)
#             snw_pp__[901:,:] = np.nan

#             # Apply the conditions
#             condition1 = (est_snw == 2) & (snw_probab >= 0.2) & (abs_diff_snw <= np.nanmedian(abs_diff_snw))
#             condition2 = (est_snw == 2) & (snw_probab < 0.2) & (abs_diff_snw > np.nanmedian(abs_diff_snw))        

#             snw_pp__[condition1] = 2
#             snw_pp__[condition2] = 1

#             snw_pp__ = snw_pp__.astype(float)
#             snw_pp__[901:,:]  = np.nan

#             RF_clim_est_pp = np.where(snw_pp__ <= 2,snw_pp__,RF_clim_est) 

#             # for all the 4 areas representing undeff areas, we fill with previous days observations
#             # if previous day doesnt exist we maintian current observation

#             current_day = datetime.strptime(input_date_str, '%Y-%m-%d').date()
        
#             prev_day = current_day - timedelta(days=1)

#             next_day = current_day + timedelta(days=1)

#             # RF estimated using Climatology
#             if len(str(day_of_year(prev_day.strftime('%Y-%m-%d')))) == 1:
#                 prev_doy = str(prev_day.year) + '00' +str(day_of_year(prev_day.strftime('%Y-%m-%d')))

#             elif len(str(day_of_year(prev_day.strftime('%Y-%m-%d')))) == 2:
#                 prev_doy = str(prev_day.year) + '0' +str(day_of_year(prev_day.strftime('%Y-%m-%d')))
#             else:
#                 prev_doy = str(prev_day.year) + str(day_of_year(prev_day.strftime('%Y-%m-%d')))

#             #-------------------------------------------------

#             if len(str(day_of_year(next_day.strftime('%Y-%m-%d')))) == 1:
#                 nxt_doy = str(next_day.year) + '00' +str(day_of_year(next_day.strftime('%Y-%m-%d')))

#             elif len(str(day_of_year(next_day.strftime('%Y-%m-%d')))) == 2:
#                 nxt_doy = str(next_day.year) + '0' +str(day_of_year(next_day.strftime('%Y-%m-%d')))
#             else:
#                 nxt_doy = str(next_day.year) + str(day_of_year(next_day.strftime('%Y-%m-%d')))

#             #-------------------------------------------------

#             prev_day_climrfmp_read_nme = os.path.join(path_to_estimated_autosnw,
#                                                     '_'.join(['RF_estimated_autosnow_using_alldataclim',
#                                                         prev_doy,'0.1deg_wgs']) + '.tif')
            
#             nxt_day_climrfmp_read_nme = os.path.join(path_to_estimated_autosnw,
#                                                     '_'.join(['RF_estimated_autosnow_using_alldataclim',
#                                                         nxt_doy,'0.1deg_wgs']) + '.tif')
            
#             #-------------------------------------------------

#             if os.path.isfile(prev_day_climrfmp_read_nme):

#                 prev_climrf_autsnw_estimated = xr.open_dataarray(prev_day_climrfmp_read_nme)
#                 prev_climrf_autsnw_estimated_arr = prev_climrf_autsnw_estimated.data[0,:,:]
#                 prev_climrf_autsnw_estimated_arr = np.where(prev_climrf_autsnw_estimated_arr > 3,np.nan,
#                                                             prev_climrf_autsnw_estimated_arr)
                
#                 snow_corrected_rf_clim = np.where(snw_pp__ > 2, prev_climrf_autsnw_estimated_arr, RF_clim_est_pp)
#             else:
#                 nxt_climrf_autsnw_estimated = xr.open_dataarray(nxt_day_climrfmp_read_nme)
#                 nxt_climrf_autsnw_estimated_arr = nxt_climrf_autsnw_estimated.data[0,:,:]
#                 nxt_climrf_autsnw_estimated_arr = np.where(nxt_climrf_autsnw_estimated_arr > 3,np.nan,
#                                                             nxt_climrf_autsnw_estimated_arr)
                
#                 snow_corrected_rf_clim = np.where(snw_pp__ > 2, nxt_climrf_autsnw_estimated_arr, RF_clim_est_pp)        
            
#             #---------------------------------------------

#             #  # Step 1: Apply distance transform to identify proximity to coastline
#             # # Invert land_sea_mask for distance calculation (distance_transform_edt considers non-zero as features)
#             # distances = distance_transform_edt(np.logical_not(lsm))

#             # # Define classification based on distance to coast (specific thresholds would be derived from the article)
#             # # For example purposes, distances <= 2 are considered close to the coast, >2 and <=5 are intermediate, and >5 are far
#             # # These thresholds should be adjusted according to the article's specifications or empirical analysis
#             # coastal_proximity = np.digitize(distances, bins=[2, 5])

#             # # Step 2: Correct surface data based on proximity and land-sea mask
#             # # Pixels close to the coast or on land may need special handling to correct for land spillover or misclassification
#             corrected_surface_data = np.copy(snow_corrected_rf_clim)        

#             corrected_surface_data = generic_filter(corrected_surface_data, correct_pixel, size=(3, 3), mode='nearest')

#             #-------------------------- regional specfic corrections -------------------------------
#             corrected_surface_data_ = np.where(antartica_msk,2,corrected_surface_data)

#             ml_ecc_estimated_arr = np.where(tropical_mask,1,corrected_surface_data_)

#             # save the finally correcetd map
#             # meta_autosnow_cpy_ = meta_autosnow.copy()
#             # meta_autosnow_cpy_.update({'dtype':np.int16})
#             rasterio_based_save_array_to_disk(path_to_estimated_autosnw,ml_ecc_read_nme,
#                                                     meta_autosnow,ml_ecc_estimated_arr)

#         #------------------------------------------------------

#         ml_e_gmasi_flat = np.column_stack([gmasi_dat_array.flatten(),ml_e_estimated_arr.flatten()])
#         ml_e_gmasi_hits = calculate_hits_miss(ml_e_gmasi_flat[:,1],ml_e_gmasi_flat[:,0],integer_list,'hit') 
#         ml_e_gmasi_miss = calculate_hits_miss(ml_e_gmasi_flat[:,1],ml_e_gmasi_flat[:,0],integer_list,'miss') 
#         ml_e_gmasi_count = ml_e_gmasi_hits + ml_e_gmasi_miss

#         ml_ec_gmasi_flat = np.column_stack([gmasi_dat_array.flatten(),ml_ec_estimated_arr.flatten()])
#         ml_ec_gmasi_hits = calculate_hits_miss(ml_ec_gmasi_flat[:,1],ml_ec_gmasi_flat[:,0],integer_list,'hit') 
#         ml_ec_gmasi_miss = calculate_hits_miss(ml_ec_gmasi_flat[:,1],ml_ec_gmasi_flat[:,0],integer_list,'miss') 
#         ml_ec_gmasi_count = ml_ec_gmasi_hits + ml_ec_gmasi_miss

#         climatology_gmasi_flat = np.column_stack([gmasi_dat_array.flatten(),climatology_estimated_arr.flatten()])
#         climatology_gmasi_hits = calculate_hits_miss(climatology_gmasi_flat[:,1],climatology_gmasi_flat[:,0],integer_list,'hit') 
#         climatology_gmasi_miss = calculate_hits_miss(climatology_gmasi_flat[:,1],climatology_gmasi_flat[:,0],integer_list,'miss') 
#         climatology_gmasi_count = climatology_gmasi_hits + climatology_gmasi_miss

#         ml_ecc_gmasi_flat = np.column_stack([gmasi_dat_array.flatten(),ml_ecc_estimated_arr.flatten()])
#         ml_ecc_gmasi_hits = calculate_hits_miss(ml_ecc_gmasi_flat[:,1],ml_ecc_gmasi_flat[:,0],integer_list,'hit') 
#         ml_ecc_gmasi_miss = calculate_hits_miss(ml_ecc_gmasi_flat[:,1],ml_ecc_gmasi_flat[:,0],integer_list,'miss') 
#         ml_ecc_gmasi_count = ml_ecc_gmasi_hits + ml_ecc_gmasi_miss

#         e_gmasi_flat = np.column_stack([gmasi_dat_array.flatten(),e_estimated_arr.flatten()])
#         e_gmasi_hits = calculate_hits_miss(e_gmasi_flat[:,1],e_gmasi_flat[:,0],integer_list,'hit') 
#         e_gmasi_miss = calculate_hits_miss(e_gmasi_flat[:,1],e_gmasi_flat[:,0],integer_list,'miss') 
#         e_gmasi_count = e_gmasi_hits + e_gmasi_miss

#         # calculate hit/miss in %
#         hit_miss_df.loc[date_time,'ML-E-GMASI-hit'] = round((ml_e_gmasi_hits/ml_e_gmasi_count)*100,2)
#         hit_miss_df.loc[date_time,'ML_E-GMASI-miss'] = round((ml_e_gmasi_miss/ml_e_gmasi_count)*100,2)

#         hit_miss_df.loc[date_time,'ML-EC-GMASI-hit'] = round((ml_ec_gmasi_hits/ml_ec_gmasi_count)*100,2)
#         hit_miss_df.loc[date_time,'ML-EC_GMASI-miss'] = round((ml_ec_gmasi_miss/ml_ec_gmasi_count)*100,2)

#         hit_miss_df.loc[date_time,'Climatology_GMASI-hit'] = round((climatology_gmasi_hits/climatology_gmasi_count)*100,2)
#         hit_miss_df.loc[date_time,'Climatology_GMASI-miss'] = round((climatology_gmasi_miss/climatology_gmasi_count)*100,2)

#         hit_miss_df.loc[date_time,'ML-ECC-GMASI-hit'] = round((ml_ecc_gmasi_hits/ml_ecc_gmasi_count)*100,2)
#         hit_miss_df.loc[date_time,'ML-ECC-GMASI-miss'] = round((ml_ecc_gmasi_miss/ml_ecc_gmasi_count)*100,2)

#         hit_miss_df.loc[date_time,'E-GMASI-hit'] = round((e_gmasi_hits/e_gmasi_count)*100,2)
#         hit_miss_df.loc[date_time,'E-GMASI-miss'] = round((e_gmasi_miss/e_gmasi_count)*100,2)
#         #------------------------------------------------------

#         # do hit/miss in % per class
#         ml_e_wtr_hit = np.sum((ml_e_gmasi_flat[:,0] == 0) & (ml_e_gmasi_flat[:,1] == 0))
#         ml_e_wtr_miss = np.sum((ml_e_gmasi_flat[:,0] == 0) & (ml_e_gmasi_flat[:,1] != 0))
#         ml_e_wtr_cnt = ml_e_wtr_hit + ml_e_wtr_miss
#         hit_miss_df.loc[date_time,'ML-E-wter-hit']  = round((ml_e_wtr_hit/ml_e_wtr_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'ML-E-wter-miss']  = round((ml_e_wtr_miss/ml_e_wtr_cnt)*100,2) 

#         ml_e_snfr_hit = np.sum((ml_e_gmasi_flat[:,0] == 1) & (ml_e_gmasi_flat[:,1] == 1))
#         ml_e_snfr_miss = np.sum((ml_e_gmasi_flat[:,0] == 1) & (ml_e_gmasi_flat[:,1] != 1))
#         ml_e_snfr_cnt = ml_e_snfr_hit + ml_e_snfr_miss
#         hit_miss_df.loc[date_time,'ML-E-snflnd-hit']  = round((ml_e_snfr_hit/ml_e_snfr_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'ML-E-snflnd-miss']  = round((ml_e_snfr_miss/ml_e_snfr_cnt)*100,2) 

#         ml_e_snc_hit = np.sum((ml_e_gmasi_flat[:,0] == 2) & (ml_e_gmasi_flat[:,1] == 2))
#         ml_e_snc_miss = np.sum((ml_e_gmasi_flat[:,0] == 2) & (ml_e_gmasi_flat[:,1] != 2))
#         ml_e_snc_cnt = ml_e_snc_hit + ml_e_snc_miss
#         hit_miss_df.loc[date_time,'ML-E-snclnd-hit']  = round((ml_e_snc_hit/ml_e_snc_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'ML-E-snclnd-miss']  = round((ml_e_snc_miss/ml_e_snc_cnt)*100,2) 

#         ml_e_ice_hit = np.sum((ml_e_gmasi_flat[:,0] == 3) & (ml_e_gmasi_flat[:,1] == 3))
#         ml_e_ice_miss = np.sum((ml_e_gmasi_flat[:,0] == 3) & (ml_e_gmasi_flat[:,1] != 3))
#         ml_e_ice_cnt = ml_e_ice_hit + ml_e_ice_miss
#         hit_miss_df.loc[date_time,'ML-E-ice-hit']  = round((ml_e_ice_hit/ml_e_ice_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'ML-E-ice-miss']  = round((ml_e_ice_miss/ml_e_ice_cnt)*100,2) 
    
#         #------------------------------------------------------

#         ml_ec_wtr_hit = np.sum((ml_ec_gmasi_flat[:,0] == 0) & (ml_ec_gmasi_flat[:,1] == 0))
#         ml_ec_wtr_miss = np.sum((ml_ec_gmasi_flat[:,0] == 0) & (ml_ec_gmasi_flat[:,1] != 0))
#         ml_ec_wtr_cnt = ml_ec_wtr_hit + ml_ec_wtr_miss
#         hit_miss_df.loc[date_time,'ML-EC-wter-hit']  = round((ml_ec_wtr_hit/ml_ec_wtr_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'ML-EC-wter-miss']  = round((ml_ec_wtr_miss/ml_ec_wtr_cnt)*100,2) 

#         ml_ec_snfr_hit = np.sum((ml_ec_gmasi_flat[:,0] == 1) & (ml_ec_gmasi_flat[:,1] == 1))
#         ml_ec_snfr_miss = np.sum((ml_ec_gmasi_flat[:,0] == 1) & (ml_ec_gmasi_flat[:,1] != 1))
#         ml_ec_snfr_cnt = ml_ec_snfr_hit + ml_ec_snfr_miss
#         hit_miss_df.loc[date_time,'ML-EC-snflnd-hit']  = round((ml_ec_snfr_hit/ml_ec_snfr_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'ML-EC-snflnd-miss']  = round((ml_ec_snfr_miss/ml_ec_snfr_cnt)*100,2) 

#         ml_ec_snc_hit = np.sum((ml_ec_gmasi_flat[:,0] == 2) & (ml_ec_gmasi_flat[:,1] == 2))
#         ml_ec_snc_miss = np.sum((ml_ec_gmasi_flat[:,0] == 2) & (ml_ec_gmasi_flat[:,1] != 2))
#         ml_ec_snc_cnt = ml_ec_snc_hit + ml_ec_snc_miss
#         hit_miss_df.loc[date_time,'ML-EC-snclnd-hit']  = round((ml_ec_snc_hit/ml_ec_snc_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'ML-EC-snclnd-miss']  = round((ml_ec_snc_miss/ml_ec_snc_cnt)*100,2) 

#         ml_ec_ice_hit = np.sum((ml_ec_gmasi_flat[:,0] == 3) & (ml_ec_gmasi_flat[:,1] == 3))
#         ml_ec_ice_miss = np.sum((ml_ec_gmasi_flat[:,0] == 3) & (ml_ec_gmasi_flat[:,1] != 3))
#         ml_ec_ice_cnt = ml_ec_ice_hit + ml_ec_ice_miss
#         hit_miss_df.loc[date_time,'ML-EC-ice-hit']  = round((ml_ec_ice_hit/ml_ec_ice_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'ML-EC-ice-miss']  = round((ml_ec_ice_miss/ml_ec_ice_cnt)*100,2) 

#         #------------------------------------------------------

#         climatology_wtr_hit = np.sum((climatology_gmasi_flat[:,0] == 0) & (climatology_gmasi_flat[:,1] == 0))
#         climatology_wtr_miss = np.sum((climatology_gmasi_flat[:,0] == 0) & (climatology_gmasi_flat[:,1] != 0))
#         climatology_wtr_cnt = climatology_wtr_hit + climatology_wtr_miss
#         hit_miss_df.loc[date_time,'Climatology-wter-hit']  = round((climatology_wtr_hit/climatology_wtr_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'Climatology-wter-miss']  = round((climatology_wtr_miss/climatology_wtr_cnt)*100,2) 

#         climatology_snfr_hit = np.sum((climatology_gmasi_flat[:,0] == 1) & (climatology_gmasi_flat[:,1] == 1))
#         climatology_snfr_miss = np.sum((climatology_gmasi_flat[:,0] == 1) & (climatology_gmasi_flat[:,1] != 1))
#         climatology_snfr_cnt = climatology_snfr_hit + climatology_snfr_miss
#         hit_miss_df.loc[date_time,'Climatology-snflnd-hit']  = round((climatology_snfr_hit/climatology_snfr_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'Climatology-snflnd-miss']  = round((climatology_snfr_miss/climatology_snfr_cnt)*100,2) 

#         climatology_snc_hit = np.sum((climatology_gmasi_flat[:,0] == 2) & (climatology_gmasi_flat[:,1] == 2))
#         climatology_snc_miss = np.sum((climatology_gmasi_flat[:,0] == 2) & (climatology_gmasi_flat[:,1] != 2))
#         climatology_snc_cnt = climatology_snc_hit + climatology_snc_miss
#         hit_miss_df.loc[date_time,'Climatology-snclnd-hit']  = round((climatology_snc_hit/climatology_snc_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'Climatology-snclnd-miss']  = round((climatology_snc_miss/climatology_snc_cnt)*100,2) 

#         climatology_ice_hit = np.sum((climatology_gmasi_flat[:,0] == 3) & (climatology_gmasi_flat[:,1] == 3))
#         climatology_ice_miss = np.sum((climatology_gmasi_flat[:,0] == 3) & (climatology_gmasi_flat[:,1] != 3))
#         climatology_ice_cnt = climatology_ice_hit + climatology_ice_miss
#         hit_miss_df.loc[date_time,'Climatology-ice-hit']  = round((climatology_ice_hit/climatology_ice_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'Climatology-ice-miss']  = round((climatology_ice_miss/climatology_ice_cnt)*100,2) 

#         #------------------------------------------------------

#         ml_ecc_wtr_hit = np.sum((ml_ecc_gmasi_flat[:,0] == 0) & (ml_ecc_gmasi_flat[:,1] == 0))
#         ml_ecc_wtr_miss = np.sum((ml_ecc_gmasi_flat[:,0] == 0) & (ml_ecc_gmasi_flat[:,1] != 0))
#         ml_ecc_wtr_cnt = ml_ecc_wtr_hit + ml_ecc_wtr_miss
#         hit_miss_df.loc[date_time,'ML-ECC-wter-hit']  = round((ml_ecc_wtr_hit/ml_ecc_wtr_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'ML-ECC-wter-miss']  = round((ml_ecc_wtr_miss/ml_ecc_wtr_cnt)*100,2) 

#         ml_ecc_snfr_hit = np.sum((ml_ecc_gmasi_flat[:,0] == 1) & (ml_ecc_gmasi_flat[:,1] == 1))
#         ml_ecc_snfr_miss = np.sum((ml_ecc_gmasi_flat[:,0] == 1) & (ml_ecc_gmasi_flat[:,1] != 1))
#         ml_ecc_snfr_cnt = ml_ecc_snfr_hit + ml_ecc_snfr_miss
#         hit_miss_df.loc[date_time,'ML-ECC-snflnd-hit']  = round((ml_ecc_snfr_hit/ml_ecc_snfr_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'ML-ECC-snflnd-miss']  = round((ml_ecc_snfr_miss/ml_ecc_snfr_cnt)*100,2) 

#         ml_ecc_snc_hit = np.sum((ml_ecc_gmasi_flat[:,0] == 2) & (ml_ecc_gmasi_flat[:,1] == 2))
#         ml_ecc_snc_miss = np.sum((ml_ecc_gmasi_flat[:,0] == 2) & (ml_ecc_gmasi_flat[:,1] != 2))
#         ml_ecc_snc_cnt = ml_ecc_snc_hit + ml_ecc_snc_miss
#         hit_miss_df.loc[date_time,'ML-ECC-snclnd-hit']  = round((ml_ecc_snc_hit/ml_ecc_snc_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'ML-ECC-snclnd-miss']  = round((ml_ecc_snc_miss/ml_ecc_snc_cnt)*100,2) 

#         ml_ecc_ice_hit = np.sum((ml_ecc_gmasi_flat[:,0] == 3) & (ml_ecc_gmasi_flat[:,1] == 3))
#         ml_ecc_ice_miss = np.sum((ml_ecc_gmasi_flat[:,0] == 3) & (ml_ecc_gmasi_flat[:,1] != 3))
#         ml_ecc_ice_cnt = ml_ecc_ice_hit + ml_ecc_ice_miss
#         hit_miss_df.loc[date_time,'ML-ECC-ice-hit']  = round((ml_ecc_ice_hit/ml_ecc_ice_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'ML-ECC-ice-miss']  = round((ml_ecc_ice_miss/ml_ecc_ice_cnt)*100,2) 

#         #------------------------------------------------------

#         e_wtr_hit = np.sum((e_gmasi_flat[:,0] == 0) & (e_gmasi_flat[:,1] == 0))
#         e_wtr_miss = np.sum((e_gmasi_flat[:,0] == 0) & (e_gmasi_flat[:,1] != 0))
#         e_wtr_cnt = e_wtr_hit + e_wtr_miss
#         hit_miss_df.loc[date_time,'E-wter-hit']  = round((e_wtr_hit/e_wtr_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'E-wter-miss']  = round((e_wtr_miss/e_wtr_cnt)*100,2) 

#         e_snfr_hit = np.sum((e_gmasi_flat[:,0] == 1) & (e_gmasi_flat[:,1] == 1))
#         e_snfr_miss = np.sum((e_gmasi_flat[:,0] == 1) & (e_gmasi_flat[:,1] != 1))
#         e_snfr_cnt = e_snfr_hit + e_snfr_miss
#         hit_miss_df.loc[date_time,'E-snflnd-hit']  = round((e_snfr_hit/e_snfr_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'E-snflnd-miss']  = round((e_snfr_miss/e_snfr_cnt)*100,2) 

#         e_snc_hit = np.sum((e_gmasi_flat[:,0] == 2) & (e_gmasi_flat[:,1] == 2))
#         e_snc_miss = np.sum((e_gmasi_flat[:,0] == 2) & (e_gmasi_flat[:,1] != 2))
#         e_snc_cnt = e_snc_hit + e_snc_miss
#         hit_miss_df.loc[date_time,'E-snclnd-hit']  = round((e_snc_hit/e_snc_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'E-snclnd-miss']  = round((e_snc_miss/e_snc_cnt)*100,2) 

#         e_ice_hit = np.sum((e_gmasi_flat[:,0] == 3) & (e_gmasi_flat[:,1] == 3))
#         e_ice_miss = np.sum((e_gmasi_flat[:,0] == 3) & (e_gmasi_flat[:,1] != 3))
#         e_ice_cnt = e_ice_hit + e_ice_miss
#         hit_miss_df.loc[date_time,'E-ice-hit']  = round((e_ice_hit/e_ice_cnt)*100,2) 
#         hit_miss_df.loc[date_time,'E-ice-miss']  = round((e_ice_miss/e_ice_cnt)*100,2) 

#         #------------------------------------------------------

#         # append estimates and original to baskets
#         ml_e_glb_bskt.append(ml_e_estimated_arr)

#         ml_ec_glb_bskt.append(ml_ec_estimated_arr)

#         climatology_glb_bskt.append(climatology_estimated_arr)

#         ml_ecc_glb_bskt.append(ml_ecc_estimated_arr)

#         e_glb_bskt.append(e_estimated_arr)

#         gmasi_glb_bskt.append(gmasi_dat_array) 
#         gc.collect()
#         #------------------------------------------------------

#         # segregate the data into different lists according NH, SH and seasons
#         ml_e_arr_nh = ml_e_estimated_arr[0:nh_row[0],:].astype(np.int16)
#         ml_e_nh.append(ml_e_arr_nh)

#         ml_e_arr_sh = ml_e_estimated_arr[sh_row[0]:y_shp,:].astype(np.int16)
#         ml_e_sh.append(ml_e_arr_sh)
#         #-------------------

#         gmasi_arr_nh = gmasi_dat_array[0:nh_row[0],:].astype(np.int16)
#         gmasi_nh.append(gmasi_arr_nh)

#         gmasi_arr_sh = gmasi_dat_array[sh_row[0]:y_shp,:].astype(np.int16)
#         gmasi_sh.append(gmasi_arr_sh)
#         #-------------------

#         ml_ec_arr_nh = ml_ec_estimated_arr[0:nh_row[0],:].astype(np.int16)
#         ml_ec_nh.append(ml_ec_arr_nh)

#         ml_ec_arr_sh = ml_ec_estimated_arr[sh_row[0]:y_shp,:].astype(np.int16)
#         ml_ec_sh.append(ml_ec_arr_sh)
#         #-------------------

#         climatology_arr_nh = climatology_estimated_arr[0:nh_row[0],:].astype(np.int16)
#         climatology_nh.append(climatology_arr_nh)

#         climatology_arr_sh = climatology_estimated_arr[sh_row[0]:y_shp,:].astype(np.int16)
#         climatology_sh.append(climatology_arr_sh)

#         #-------------------

#         ml_ecc_arr_nh = ml_ecc_estimated_arr[0:nh_row[0],:].astype(np.int16)
#         ml_ecc_nh.append(ml_ecc_arr_nh)

#         ml_ecc_arr_sh = ml_ecc_estimated_arr[sh_row[0]:y_shp,:].astype(np.int16)
#         ml_ecc_sh.append(ml_ecc_arr_sh)
#         #-------------------

#         e_arr_nh = e_estimated_arr[0:nh_row[0],:].astype(np.int16)
#         e_nh.append(e_arr_nh)

#         e_arr_sh = e_estimated_arr[sh_row[0]:y_shp,:].astype(np.int16)
#         e_sh.append(e_arr_sh)
        
#     #------------------------------------------------------
#         # do the extent comparison at the seasonal level
#         # NH
#         # rf_nh_gdf = get_gdf(rf_atsnw_arr_nh,rfmp_read_nme,'_nh')
#         # rfclim_nh_gdf = get_gdf(climrf_atsnw_arr_nh,climrfmp_read_nme,'_nh')
#         # tempclim_nh_gdf = get_gdf(temp_climrf_atsnw_arr_nh,temp_climrfmp_read_nme,'_nh')
#         # orig_nh_gdf = get_gdf(orig_atsnw_arr_nh,l,'_nh')
#         # cor_rfclim_nh_gdf = get_gdf(cor_climrf_atsnw_arr_nh,corrected_climrfmp_read_nme,'_nh')
#         ml_e_cnt_nh = count_class_pixels(ml_e_arr_nh)
#         ml_ec_cnt_nh = count_class_pixels(ml_ec_arr_nh)
#         ml_ecc_cnt_nh = count_class_pixels(ml_ecc_arr_nh)
#         climatology_cnt_nh = count_class_pixels(climatology_arr_nh)
#         gmasi_cnt_nh = count_class_pixels(gmasi_arr_nh)
#         e_cnt_nh = count_class_pixels(e_arr_nh)


#         nh_px_cnt.loc[date_time,'ml_e_wtr_px_cnt'] = ml_e_cnt_nh[0]
#         nh_px_cnt.loc[date_time,'ml_e_snfr_px_cnt'] = ml_e_cnt_nh[1]
#         nh_px_cnt.loc[date_time,'ml_e_snc_px_cnt'] = ml_e_cnt_nh[2]
#         nh_px_cnt.loc[date_time,'ml_e_ice_px_cnt'] = ml_e_cnt_nh[3]

#         nh_px_cnt.loc[date_time,'ml_ec_wtr_px_cnt'] = ml_ec_cnt_nh[0]
#         nh_px_cnt.loc[date_time,'ml_ec_snfr_px_cnt'] = ml_ec_cnt_nh[1]
#         nh_px_cnt.loc[date_time,'ml_ec_snc_px_cnt'] = ml_ec_cnt_nh[2]
#         nh_px_cnt.loc[date_time,'ml_ec_ice_px_cnt'] = ml_ec_cnt_nh[3]

#         nh_px_cnt.loc[date_time,'climatology_wtr_px_cnt'] = climatology_cnt_nh[0]
#         nh_px_cnt.loc[date_time,'climatology_snfr_px_cnt'] = climatology_cnt_nh[1]
#         nh_px_cnt.loc[date_time,'climatology_snc_px_cnt'] = climatology_cnt_nh[2]
#         nh_px_cnt.loc[date_time,'climatology_ice_px_cnt'] = climatology_cnt_nh[3]

#         nh_px_cnt.loc[date_time,'gmasi_wtr_px_cnt'] = gmasi_cnt_nh[0]
#         nh_px_cnt.loc[date_time,'gmasi_snfr_px_cnt'] = gmasi_cnt_nh[1]
#         nh_px_cnt.loc[date_time,'gmasi_snc_px_cnt'] = gmasi_cnt_nh[2]
#         nh_px_cnt.loc[date_time,'gmasi_ice_px_cnt'] = gmasi_cnt_nh[3]

#         nh_px_cnt.loc[date_time,'ml_ecc_wtr_px_cnt'] = ml_ecc_cnt_nh[0]
#         nh_px_cnt.loc[date_time,'ml_ecc_snfr_px_cnt'] = ml_ecc_cnt_nh[1]
#         nh_px_cnt.loc[date_time,'ml_ecc_snc_px_cnt'] = ml_ecc_cnt_nh[2]
#         nh_px_cnt.loc[date_time,'ml_ecc_ice_px_cnt'] = ml_ecc_cnt_nh[3]

#         nh_px_cnt.loc[date_time,'e_wtr_px_cnt'] = e_cnt_nh[0]
#         nh_px_cnt.loc[date_time,'e_snfr_px_cnt'] = e_cnt_nh[1]
#         nh_px_cnt.loc[date_time,'e_snc_px_cnt'] = e_cnt_nh[2]
#         nh_px_cnt.loc[date_time,'e_ice_px_cnt'] = e_cnt_nh[3]

#         #---------------------------------------------------
#         # SH
#         # rf_sh_gdf = get_gdf(rf_atsnw_arr_sh,rfmp_read_nme,'_sh')
#         # rfclim_sh_gdf = get_gdf(climrf_atsnw_arr_sh,climrfmp_read_nme,'_sh')
#         # tempclim_sh_gdf = get_gdf(temp_climrf_atsnw_arr_sh,temp_climrfmp_read_nme,'_sh')
#         # orig_sh_gdf = get_gdf(orig_atsnw_arr_sh,l,'_sh')
#         # cor_rfclim_sh_gdf = get_gdf(cor_climrf_atsnw_arr_sh,corrected_climrfmp_read_nme,'_sh')

#         ml_e_cnt_sh = count_class_pixels(ml_e_arr_sh)
#         ml_ec_cnt_sh = count_class_pixels(ml_ec_arr_sh)
#         ml_ecc_cnt_sh = count_class_pixels(ml_ecc_arr_sh)
#         climatology_cnt_sh = count_class_pixels(climatology_arr_sh)
#         gmasi_cnt_sh = count_class_pixels(gmasi_arr_sh)
#         e_cnt_sh = count_class_pixels(e_arr_sh)

#         sh_px_cnt.loc[date_time,'ml_e_wtr_px_cnt'] = ml_e_cnt_sh[0]
#         sh_px_cnt.loc[date_time,'ml_e_snfr_px_cnt'] = ml_e_cnt_sh[1]
#         sh_px_cnt.loc[date_time,'ml_e_snc_px_cnt'] = ml_e_cnt_sh[2]
#         sh_px_cnt.loc[date_time,'ml_e_ice_px_cnt'] = ml_e_cnt_sh[3]

#         sh_px_cnt.loc[date_time,'ml_ec_wtr_px_cnt'] = ml_ec_cnt_sh[0]
#         sh_px_cnt.loc[date_time,'ml_ec_snfr_px_cnt'] = ml_ec_cnt_sh[1]
#         sh_px_cnt.loc[date_time,'ml_ec_snc_px_cnt'] = ml_ec_cnt_sh[2]
#         sh_px_cnt.loc[date_time,'ml_ec_ice_px_cnt'] = ml_ec_cnt_sh[3]

#         sh_px_cnt.loc[date_time,'climatology_wtr_px_cnt'] = climatology_cnt_sh[0]
#         sh_px_cnt.loc[date_time,'climatology_snfr_px_cnt'] = climatology_cnt_sh[1]
#         sh_px_cnt.loc[date_time,'climatology_snc_px_cnt'] = climatology_cnt_sh[2]
#         sh_px_cnt.loc[date_time,'climatology_ice_px_cnt'] = climatology_cnt_sh[3]

#         sh_px_cnt.loc[date_time,'gmasi_wtr_px_cnt'] = gmasi_cnt_sh[0]
#         sh_px_cnt.loc[date_time,'gmasi_snfr_px_cnt'] = gmasi_cnt_sh[1]
#         sh_px_cnt.loc[date_time,'gmasi_snc_px_cnt'] = gmasi_cnt_sh[2]
#         sh_px_cnt.loc[date_time,'gmasi_ice_px_cnt'] = gmasi_cnt_sh[3]

#         sh_px_cnt.loc[date_time,'ml_ecc_wtr_px_cnt'] = ml_ecc_cnt_sh[0]
#         sh_px_cnt.loc[date_time,'ml_ecc_snfr_px_cnt'] = ml_ecc_cnt_sh[1]
#         sh_px_cnt.loc[date_time,'ml_ecc_snc_px_cnt'] = ml_ecc_cnt_sh[2]
#         sh_px_cnt.loc[date_time,'ml_ecc_ice_px_cnt'] = ml_ecc_cnt_sh[3]

#         sh_px_cnt.loc[date_time,'e_wtr_px_cnt'] = e_cnt_sh[0]
#         sh_px_cnt.loc[date_time,'e_snfr_px_cnt'] = e_cnt_sh[1]
#         sh_px_cnt.loc[date_time,'e_snc_px_cnt'] = e_cnt_sh[2]
#         sh_px_cnt.loc[date_time,'e_ice_px_cnt'] = e_cnt_sh[3]
#         #----------------------------------------------------

#         # append based on season and hemisphere
#         season_nh = find_season(mnth,'Northern')
#         if season_nh is 'Winter':        
#             nh_winter_ml_e.append(ml_e_arr_nh) 
#             nh_winter_gmasi.append(gmasi_arr_nh)      
#             nh_winter_ml_ec.append(ml_ec_arr_nh)  
#             nh_winter_climatology.append(climatology_arr_nh) 
#             nh_winter_ml_ecc.append(ml_ecc_arr_nh)  
#             nh_winter_e.append(e_arr_nh)     

#             ml_e_winter_cnt_nh = count_class_pixels(ml_e_arr_nh)
#             ml_ec_winter_cnt_nh = count_class_pixels(ml_ec_arr_nh)
#             ml_ecc_winter_cnt_nh = count_class_pixels(ml_ecc_arr_nh)
#             climatology_winter_cnt_nh = count_class_pixels(climatology_arr_nh)
#             gmasi_winter_cnt_nh = count_class_pixels(gmasi_arr_nh) 
#             e_winter_cnt_nh = count_class_pixels(e_arr_nh)       

#             nh_winter_px_cnt.loc[date_time,'ml_e_wtr_px_cnt'] = ml_e_winter_cnt_nh[0]
#             nh_winter_px_cnt.loc[date_time,'ml_e_snfr_px_cnt'] = ml_e_winter_cnt_nh[1]
#             nh_winter_px_cnt.loc[date_time,'ml_e_snc_px_cnt'] = ml_e_winter_cnt_nh[2]
#             nh_winter_px_cnt.loc[date_time,'ml_e_ice_px_cnt'] = ml_e_winter_cnt_nh[3]

#             nh_winter_px_cnt.loc[date_time,'ml_ec_wtr_px_cnt'] = ml_ec_winter_cnt_nh[0]
#             nh_winter_px_cnt.loc[date_time,'ml_ec_snfr_px_cnt'] = ml_ec_winter_cnt_nh[1]
#             nh_winter_px_cnt.loc[date_time,'ml_ec_snc_px_cnt'] = ml_ec_winter_cnt_nh[2]
#             nh_winter_px_cnt.loc[date_time,'ml_ec_ice_px_cnt'] = ml_ec_winter_cnt_nh[3]

#             nh_winter_px_cnt.loc[date_time,'climatology_wtr_px_cnt'] = climatology_winter_cnt_nh[0]
#             nh_winter_px_cnt.loc[date_time,'climatology_snfr_px_cnt'] = climatology_winter_cnt_nh[1]
#             nh_winter_px_cnt.loc[date_time,'climatology_snc_px_cnt'] = climatology_winter_cnt_nh[2]
#             nh_winter_px_cnt.loc[date_time,'climatology_ice_px_cnt'] = climatology_winter_cnt_nh[3]

#             nh_winter_px_cnt.loc[date_time,'gmasi_wtr_px_cnt'] = gmasi_winter_cnt_nh[0]
#             nh_winter_px_cnt.loc[date_time,'gmasi_snfr_px_cnt'] = gmasi_winter_cnt_nh[1]
#             nh_winter_px_cnt.loc[date_time,'gmasi_snc_px_cnt'] = gmasi_winter_cnt_nh[2]
#             nh_winter_px_cnt.loc[date_time,'gmasi_ice_px_cnt'] = gmasi_winter_cnt_nh[3]

#             nh_winter_px_cnt.loc[date_time,'ml_ecc_wtr_px_cnt'] = ml_ecc_winter_cnt_nh[0]
#             nh_winter_px_cnt.loc[date_time,'ml_ecc_snfr_px_cnt'] = ml_ecc_winter_cnt_nh[1]
#             nh_winter_px_cnt.loc[date_time,'ml_ecc_snc_px_cnt'] = ml_ecc_winter_cnt_nh[2]
#             nh_winter_px_cnt.loc[date_time,'ml_ecc_ice_px_cnt'] = ml_ecc_winter_cnt_nh[3]

#             nh_winter_px_cnt.loc[date_time,'e_wtr_px_cnt'] = e_winter_cnt_nh[0]
#             nh_winter_px_cnt.loc[date_time,'e_snfr_px_cnt'] = e_winter_cnt_nh[1]
#             nh_winter_px_cnt.loc[date_time,'e_snc_px_cnt'] = e_winter_cnt_nh[2]
#             nh_winter_px_cnt.loc[date_time,'e_ice_px_cnt'] = e_winter_cnt_nh[3]

#             #----------------------------------------------------

#         elif season_nh is 'Spring':        
#             nh_spring_ml_e.append(ml_e_arr_nh)
#             nh_spring_gmasi.append(gmasi_arr_nh)   
#             nh_spring_ml_ec.append(ml_ec_arr_nh) 
#             nh_spring_climatology.append(climatology_arr_nh) 
#             nh_spring_ml_ecc.append(ml_ecc_arr_nh) 
#             nh_spring_e.append(e_arr_nh)    

#             ml_e_spring_cnt_nh = count_class_pixels(ml_e_arr_nh)
#             ml_ec_spring_cnt_nh = count_class_pixels(ml_ec_arr_nh)
#             ml_ecc_spring_cnt_nh = count_class_pixels(ml_ecc_arr_nh)
#             climatology_spring_cnt_nh = count_class_pixels(climatology_arr_nh)
#             gmasi_spring_cnt_nh = count_class_pixels(gmasi_arr_nh)
#             e_spring_cnt_nh = count_class_pixels(e_arr_nh)              

#             nh_spring_px_cnt.loc[date_time,'ml_e_wtr_px_cnt'] = ml_e_spring_cnt_nh[0]
#             nh_spring_px_cnt.loc[date_time,'ml_e_snfr_px_cnt'] = ml_e_spring_cnt_nh[1]
#             nh_spring_px_cnt.loc[date_time,'ml_e_snc_px_cnt'] = ml_e_spring_cnt_nh[2]
#             nh_spring_px_cnt.loc[date_time,'ml_e_ice_px_cnt'] = ml_e_spring_cnt_nh[3]

#             nh_spring_px_cnt.loc[date_time,'ml_ec_wtr_px_cnt'] = ml_ec_spring_cnt_nh[0]
#             nh_spring_px_cnt.loc[date_time,'ml_ec_snfr_px_cnt'] = ml_ec_spring_cnt_nh[1]
#             nh_spring_px_cnt.loc[date_time,'ml_ec_snc_px_cnt'] = ml_ec_spring_cnt_nh[2]
#             nh_spring_px_cnt.loc[date_time,'ml_ec_ice_px_cnt'] = ml_ec_spring_cnt_nh[3]

#             nh_spring_px_cnt.loc[date_time,'climatology_wtr_px_cnt'] = climatology_spring_cnt_nh[0]
#             nh_spring_px_cnt.loc[date_time,'climatology_snfr_px_cnt'] = climatology_spring_cnt_nh[1]
#             nh_spring_px_cnt.loc[date_time,'climatology_snc_px_cnt'] = climatology_spring_cnt_nh[2]
#             nh_spring_px_cnt.loc[date_time,'climatology_ice_px_cnt'] = climatology_spring_cnt_nh[3]

#             nh_spring_px_cnt.loc[date_time,'gmasi_wtr_px_cnt'] = gmasi_spring_cnt_nh[0]
#             nh_spring_px_cnt.loc[date_time,'gmasi_snfr_px_cnt'] = gmasi_spring_cnt_nh[1]
#             nh_spring_px_cnt.loc[date_time,'gmasi_snc_px_cnt'] = gmasi_spring_cnt_nh[2]
#             nh_spring_px_cnt.loc[date_time,'gmasi_ice_px_cnt'] = gmasi_spring_cnt_nh[3]
            
#             nh_spring_px_cnt.loc[date_time,'ml_ecc_wtr_px_cnt'] = ml_ecc_spring_cnt_nh[0]
#             nh_spring_px_cnt.loc[date_time,'ml_ecc_snfr_px_cnt'] = ml_ecc_spring_cnt_nh[1]
#             nh_spring_px_cnt.loc[date_time,'ml_ecc_snc_px_cnt'] = ml_ecc_spring_cnt_nh[2]
#             nh_spring_px_cnt.loc[date_time,'ml_ecc_ice_px_cnt'] = ml_ecc_spring_cnt_nh[3]

#             nh_spring_px_cnt.loc[date_time,'e_wtr_px_cnt'] = e_spring_cnt_nh[0]
#             nh_spring_px_cnt.loc[date_time,'e_snfr_px_cnt'] = e_spring_cnt_nh[1]
#             nh_spring_px_cnt.loc[date_time,'e_snc_px_cnt'] = e_spring_cnt_nh[2]
#             nh_spring_px_cnt.loc[date_time,'e_ice_px_cnt'] = e_spring_cnt_nh[3]

#             #----------------------------------------------------

#         elif season_nh is 'Summer':
#             nh_summer_ml_e.append(ml_e_arr_nh)
#             nh_summer_gmasi.append(gmasi_arr_nh)
#             nh_summer_ml_ec.append(ml_ec_arr_nh)
#             nh_summer_climatology.append(climatology_arr_nh)
#             nh_summer_ml_ecc.append(ml_ecc_arr_nh)
#             nh_summer_e.append(e_arr_nh)

#             ml_e_summer_cnt_nh = count_class_pixels(ml_e_arr_nh)
#             ml_ec_summer_cnt_nh = count_class_pixels(ml_ec_arr_nh)
#             ml_ecc_summer_cnt_nh = count_class_pixels(ml_ecc_arr_nh)
#             climatology_summer_cnt_nh = count_class_pixels(climatology_arr_nh)
#             gmasi_summer_cnt_nh = count_class_pixels(gmasi_arr_nh)
#             e_summer_cnt_nh = count_class_pixels(e_arr_nh)   

#             nh_summer_px_cnt.loc[date_time,'ml_e_wtr_px_cnt'] = ml_e_summer_cnt_nh[0]
#             nh_summer_px_cnt.loc[date_time,'ml_e_snfr_px_cnt'] = ml_e_summer_cnt_nh[1]
#             nh_summer_px_cnt.loc[date_time,'ml_e_snc_px_cnt'] = ml_e_summer_cnt_nh[2]
#             nh_summer_px_cnt.loc[date_time,'ml_e_ice_px_cnt'] = ml_e_summer_cnt_nh[3]

#             nh_summer_px_cnt.loc[date_time,'ml_ec_wtr_px_cnt'] = ml_ec_summer_cnt_nh[0]
#             nh_summer_px_cnt.loc[date_time,'ml_ec_snfr_px_cnt'] = ml_ec_summer_cnt_nh[1]
#             nh_summer_px_cnt.loc[date_time,'ml_ec_snc_px_cnt'] = ml_ec_summer_cnt_nh[2]
#             nh_summer_px_cnt.loc[date_time,'ml_ec_ice_px_cnt'] = ml_ec_summer_cnt_nh[3]

#             nh_summer_px_cnt.loc[date_time,'climatology_wtr_px_cnt'] = climatology_summer_cnt_nh[0]
#             nh_summer_px_cnt.loc[date_time,'climatology_snfr_px_cnt'] = climatology_summer_cnt_nh[1]
#             nh_summer_px_cnt.loc[date_time,'climatology_snc_px_cnt'] = climatology_summer_cnt_nh[2]
#             nh_summer_px_cnt.loc[date_time,'tclimatology_ice_px_cnt'] = climatology_summer_cnt_nh[3]

#             nh_summer_px_cnt.loc[date_time,'gmasi_wtr_px_cnt'] = gmasi_summer_cnt_nh[0]
#             nh_summer_px_cnt.loc[date_time,'gmasi_snfr_px_cnt'] = gmasi_summer_cnt_nh[1]
#             nh_summer_px_cnt.loc[date_time,'gmasi_snc_px_cnt'] = gmasi_summer_cnt_nh[2]
#             nh_summer_px_cnt.loc[date_time,'gmasi_ice_px_cnt'] = gmasi_summer_cnt_nh[3]

#             nh_summer_px_cnt.loc[date_time,'ml_ecc_wtr_px_cnt'] = ml_ecc_summer_cnt_nh[0]
#             nh_summer_px_cnt.loc[date_time,'ml_ecc_snfr_px_cnt'] = ml_ecc_summer_cnt_nh[1]
#             nh_summer_px_cnt.loc[date_time,'ml_ecc_snc_px_cnt'] = ml_ecc_summer_cnt_nh[2]
#             nh_summer_px_cnt.loc[date_time,'ml_ecc_ice_px_cnt'] = ml_ecc_summer_cnt_nh[3]

#             nh_summer_px_cnt.loc[date_time,'e_wtr_px_cnt'] = e_summer_cnt_nh[0]
#             nh_summer_px_cnt.loc[date_time,'e_snfr_px_cnt'] = e_summer_cnt_nh[1]
#             nh_summer_px_cnt.loc[date_time,'e_snc_px_cnt'] = e_summer_cnt_nh[2]
#             nh_summer_px_cnt.loc[date_time,'e_ice_px_cnt'] = e_summer_cnt_nh[3]

#             #----------------------------------------------------

#         elif season_nh is 'Autumn':
#             nh_autumn_ml_e.append(ml_e_arr_nh)
#             nh_autumn_gmasi.append(gmasi_arr_nh)
#             nh_autumn_ml_ec.append(ml_ec_arr_nh)
#             nh_autumn_climatology.append(climatology_arr_nh)
#             nh_autumn_ml_ecc.append(ml_ecc_arr_nh)
#             nh_autumn_e.append(e_arr_nh)

#             ml_e_autumn_cnt_nh = count_class_pixels(ml_e_arr_nh)
#             ml_ec_autumn_cnt_nh = count_class_pixels(ml_ec_arr_nh)
#             ml_ecc_autumn_cnt_nh = count_class_pixels(ml_ecc_arr_nh)
#             climatology_autumn_cnt_nh = count_class_pixels(climatology_arr_nh)
#             gmasi_autumn_cnt_nh = count_class_pixels(gmasi_arr_nh) 
#             e_autumn_cnt_nh = count_class_pixels(e_arr_nh) 

#             nh_autumn_px_cnt.loc[date_time,'ml_e_wtr_px_cnt'] = ml_e_autumn_cnt_nh[0]
#             nh_autumn_px_cnt.loc[date_time,'ml_e_snfr_px_cnt'] = ml_e_autumn_cnt_nh[1]
#             nh_autumn_px_cnt.loc[date_time,'ml_e_snc_px_cnt'] = ml_e_autumn_cnt_nh[2]
#             nh_autumn_px_cnt.loc[date_time,'ml_e_ice_px_cnt'] = ml_e_autumn_cnt_nh[3]

#             nh_autumn_px_cnt.loc[date_time,'ml_ec_wtr_px_cnt'] = ml_ec_autumn_cnt_nh[0]
#             nh_autumn_px_cnt.loc[date_time,'ml_ec_snfr_px_cnt'] = ml_ec_autumn_cnt_nh[1]
#             nh_autumn_px_cnt.loc[date_time,'ml_ec_snc_px_cnt'] = ml_ec_autumn_cnt_nh[2]
#             nh_autumn_px_cnt.loc[date_time,'ml_ec_ice_px_cnt'] = ml_ec_autumn_cnt_nh[3]

#             nh_autumn_px_cnt.loc[date_time,'climatology_wtr_px_cnt'] = climatology_autumn_cnt_nh[0]
#             nh_autumn_px_cnt.loc[date_time,'climatology_snfr_px_cnt'] = climatology_autumn_cnt_nh[1]
#             nh_autumn_px_cnt.loc[date_time,'climatology_snc_px_cnt'] = climatology_autumn_cnt_nh[2]
#             nh_autumn_px_cnt.loc[date_time,'climatology_ice_px_cnt'] = climatology_autumn_cnt_nh[3]

#             nh_autumn_px_cnt.loc[date_time,'gmasi_wtr_px_cnt'] = gmasi_autumn_cnt_nh[0]
#             nh_autumn_px_cnt.loc[date_time,'gmasi_snfr_px_cnt'] = gmasi_autumn_cnt_nh[1]
#             nh_autumn_px_cnt.loc[date_time,'gmasi_snc_px_cnt'] = gmasi_autumn_cnt_nh[2]
#             nh_autumn_px_cnt.loc[date_time,'gmasi_ice_px_cnt'] = gmasi_autumn_cnt_nh[3]

#             nh_autumn_px_cnt.loc[date_time,'ml_ecc_wtr_px_cnt'] = ml_ecc_autumn_cnt_nh[0]
#             nh_autumn_px_cnt.loc[date_time,'ml_ecc_snfr_px_cnt'] = ml_ecc_autumn_cnt_nh[1]
#             nh_autumn_px_cnt.loc[date_time,'ml_ecc_snc_px_cnt'] = ml_ecc_autumn_cnt_nh[2]
#             nh_autumn_px_cnt.loc[date_time,'ml_ecc_ice_px_cnt'] = ml_ecc_autumn_cnt_nh[3]

#             nh_autumn_px_cnt.loc[date_time,'e_wtr_px_cnt'] = e_autumn_cnt_nh[0]
#             nh_autumn_px_cnt.loc[date_time,'e_snfr_px_cnt'] = e_autumn_cnt_nh[1]
#             nh_autumn_px_cnt.loc[date_time,'e_snc_px_cnt'] = e_autumn_cnt_nh[2]
#             nh_autumn_px_cnt.loc[date_time,'e_ice_px_cnt'] = e_autumn_cnt_nh[3]

#             #----------------------------------------------------

#         season_sh = find_season(mnth,'Southern')
#         if season_sh is 'Winter':
#             sh_winter_ml_e.append(ml_e_arr_sh)
#             sh_winter_gmasi.append(gmasi_arr_sh)
#             sh_winter_ml_ec.append(climatology_arr_sh)
#             sh_winter_climatology.append(climatology_arr_sh)
#             sh_winter_ml_ecc.append(ml_ecc_arr_sh)
#             sh_winter_e.append(e_arr_sh)

#             ml_e_winter_cnt_sh = count_class_pixels(ml_e_arr_sh)
#             ml_ec_winter_cnt_sh = count_class_pixels(ml_ec_arr_sh)
#             ml_ecc_winter_cnt_sh = count_class_pixels(ml_ecc_arr_sh)
#             climatology_winter_cnt_sh = count_class_pixels(climatology_arr_sh)
#             gmasi_winter_cnt_sh = count_class_pixels(gmasi_arr_sh)
#             e_winter_cnt_sh = count_class_pixels(e_arr_sh)            

#             sh_winter_px_cnt.loc[date_time,'ml_wtr_px_cnt'] = ml_e_winter_cnt_sh[0]
#             sh_winter_px_cnt.loc[date_time,'ml_snfr_px_cnt'] = ml_e_winter_cnt_sh[1]
#             sh_winter_px_cnt.loc[date_time,'ml_snc_px_cnt'] = ml_e_winter_cnt_sh[2]
#             sh_winter_px_cnt.loc[date_time,'ml_ice_px_cnt'] = ml_e_winter_cnt_sh[3]

#             sh_winter_px_cnt.loc[date_time,'ml_ec_wtr_px_cnt'] = ml_ec_winter_cnt_sh[0]
#             sh_winter_px_cnt.loc[date_time,'ml_ec_snfr_px_cnt'] = ml_ec_winter_cnt_sh[1]
#             sh_winter_px_cnt.loc[date_time,'ml_ec_snc_px_cnt'] = ml_ec_winter_cnt_sh[2]
#             sh_winter_px_cnt.loc[date_time,'ml_ec_ice_px_cnt'] = ml_ec_winter_cnt_sh[3]

#             sh_winter_px_cnt.loc[date_time,'climatology_wtr_px_cnt'] = climatology_winter_cnt_sh[0]
#             sh_winter_px_cnt.loc[date_time,'climatology_snfr_px_cnt'] = climatology_winter_cnt_sh[1]
#             sh_winter_px_cnt.loc[date_time,'climatology_snc_px_cnt'] = climatology_winter_cnt_sh[2]
#             sh_winter_px_cnt.loc[date_time,'climatology_ice_px_cnt'] = climatology_winter_cnt_sh[3]

#             sh_winter_px_cnt.loc[date_time,'gmasi_wtr_px_cnt'] = gmasi_winter_cnt_sh[0]
#             sh_winter_px_cnt.loc[date_time,'gmasi_snfr_px_cnt'] = gmasi_winter_cnt_sh[1]
#             sh_winter_px_cnt.loc[date_time,'gmasi_snc_px_cnt'] = gmasi_winter_cnt_sh[2]
#             sh_winter_px_cnt.loc[date_time,'gmasi_ice_px_cnt'] = gmasi_winter_cnt_sh[3]

#             sh_winter_px_cnt.loc[date_time,'ml_ecc_wtr_px_cnt'] = ml_ecc_winter_cnt_sh[0]
#             sh_winter_px_cnt.loc[date_time,'ml_ecc_snfr_px_cnt'] = ml_ecc_winter_cnt_sh[1]
#             sh_winter_px_cnt.loc[date_time,'ml_ecc_snc_px_cnt'] = ml_ecc_winter_cnt_sh[2]
#             sh_winter_px_cnt.loc[date_time,'ml_ecc_ice_px_cnt'] = ml_ecc_winter_cnt_sh[3]

#             sh_winter_px_cnt.loc[date_time,'e_wtr_px_cnt'] = e_winter_cnt_sh[0]
#             sh_winter_px_cnt.loc[date_time,'e_snfr_px_cnt'] = e_winter_cnt_sh[1]
#             sh_winter_px_cnt.loc[date_time,'e_snc_px_cnt'] = e_winter_cnt_sh[2]
#             sh_winter_px_cnt.loc[date_time,'e_ice_px_cnt'] = e_winter_cnt_sh[3]

#             #----------------------------------------------------

#         elif season_sh is 'Spring':
#             sh_spring_ml_e.append(ml_e_arr_sh)
#             sh_spring_gmasi.append(gmasi_arr_sh)
#             sh_spring_ml_ec.append(ml_ec_arr_sh)
#             sh_spring_climatology.append(climatology_arr_sh)
#             sh_spring_ml_ecc.append(ml_ecc_arr_sh)
#             sh_spring_e.append(e_arr_sh)
            
#             ml_e_spring_cnt_sh = count_class_pixels(ml_e_arr_sh)
#             ml_ec_spring_cnt_sh = count_class_pixels(ml_ec_arr_sh)
#             ml_ecc_spring_cnt_sh = count_class_pixels(ml_ecc_arr_sh)
#             climatology_spring_cnt_sh = count_class_pixels(climatology_arr_sh)
#             gmasi_spring_cnt_sh = count_class_pixels(gmasi_arr_sh)
#             e_spring_cnt_sh = count_class_pixels(e_arr_sh)

#             sh_spring_px_cnt.loc[date_time,'ml_e_wtr_px_cnt'] = ml_e_spring_cnt_sh[0]
#             sh_spring_px_cnt.loc[date_time,'ml_e_snfr_px_cnt'] = ml_e_spring_cnt_sh[1]
#             sh_spring_px_cnt.loc[date_time,'ml_e_snc_px_cnt'] = ml_e_spring_cnt_sh[2]
#             sh_spring_px_cnt.loc[date_time,'ml_e_ice_px_cnt'] = ml_e_spring_cnt_sh[3]

#             sh_spring_px_cnt.loc[date_time,'ml_ec_wtr_px_cnt'] = ml_ec_spring_cnt_sh[0]
#             sh_spring_px_cnt.loc[date_time,'ml_ec_snfr_px_cnt'] = ml_ec_spring_cnt_sh[1]
#             sh_spring_px_cnt.loc[date_time,'ml_ec_snc_px_cnt'] = ml_ec_spring_cnt_sh[2]
#             sh_spring_px_cnt.loc[date_time,'ml_ec_ice_px_cnt'] = ml_ec_spring_cnt_sh[3]

#             sh_spring_px_cnt.loc[date_time,'climatology_wtr_px_cnt'] = climatology_spring_cnt_sh[0]
#             sh_spring_px_cnt.loc[date_time,'climaytology_snfr_px_cnt'] = climatology_spring_cnt_sh[1]
#             sh_spring_px_cnt.loc[date_time,'climaytology_snc_px_cnt'] = climatology_spring_cnt_sh[2]
#             sh_spring_px_cnt.loc[date_time,'climaytology_ice_px_cnt'] = climatology_spring_cnt_sh[3]

#             sh_spring_px_cnt.loc[date_time,'gmasi_wtr_px_cnt'] = gmasi_spring_cnt_sh[0]
#             sh_spring_px_cnt.loc[date_time,'gmasi_snfr_px_cnt'] = gmasi_spring_cnt_sh[1]
#             sh_spring_px_cnt.loc[date_time,'gmasi_snc_px_cnt'] = gmasi_spring_cnt_sh[2]
#             sh_spring_px_cnt.loc[date_time,'gmasi_ice_px_cnt'] = gmasi_spring_cnt_sh[3]

#             sh_spring_px_cnt.loc[date_time,'ml_ecc_wtr_px_cnt'] = ml_ecc_spring_cnt_sh[0]
#             sh_spring_px_cnt.loc[date_time,'ml_ecc_snfr_px_cnt'] = ml_ecc_spring_cnt_sh[1]
#             sh_spring_px_cnt.loc[date_time,'ml_ecc_snc_px_cnt'] = ml_ecc_spring_cnt_sh[2]
#             sh_spring_px_cnt.loc[date_time,'ml_ecc_ice_px_cnt'] = ml_ecc_spring_cnt_sh[3]

#             sh_spring_px_cnt.loc[date_time,'e_wtr_px_cnt'] = e_spring_cnt_sh[0]
#             sh_spring_px_cnt.loc[date_time,'e_snfr_px_cnt'] = e_spring_cnt_sh[1]
#             sh_spring_px_cnt.loc[date_time,'e_snc_px_cnt'] = e_spring_cnt_sh[2]
#             sh_spring_px_cnt.loc[date_time,'e_ice_px_cnt'] = e_spring_cnt_sh[3]


#             #----------------------------------------------------

#         elif season_sh is 'Summer':
#             sh_summer_ml_e.append(ml_e_arr_sh)
#             sh_summer_gmasi.append(gmasi_arr_sh)  
#             sh_summer_climatology.append(climatology_arr_sh)    
#             sh_summer_ml_ec.append(ml_ec_arr_sh)  
#             sh_summer_ml_ecc.append(ml_ecc_arr_sh) 
#             sh_summer_e.append(e_arr_sh)    

#             ml_e_summer_cnt_sh = count_class_pixels(ml_e_arr_sh)
#             ml_ec_summer_cnt_sh = count_class_pixels(ml_ec_arr_sh)
#             ml_ecc_summer_cnt_sh = count_class_pixels(ml_ecc_arr_sh)
#             climatology_summer_cnt_sh = count_class_pixels(climatology_arr_sh)
#             gmasi_summer_cnt_sh = count_class_pixels(gmasi_arr_sh)
#             e_summer_cnt_sh = count_class_pixels(e_arr_sh)

#             sh_summer_px_cnt.loc[date_time,'ml_e_wtr_px_cnt'] = ml_e_summer_cnt_sh[0]
#             sh_summer_px_cnt.loc[date_time,'ml_e_snfr_px_cnt'] = ml_e_summer_cnt_sh[1]
#             sh_summer_px_cnt.loc[date_time,'ml_e_snc_px_cnt'] = ml_e_summer_cnt_sh[2]
#             sh_summer_px_cnt.loc[date_time,'ml_e_ice_px_cnt'] = ml_e_summer_cnt_sh[3]

#             sh_summer_px_cnt.loc[date_time,'ml_ec_wtr_px_cnt'] = ml_ec_summer_cnt_sh[0]
#             sh_summer_px_cnt.loc[date_time,'ml_ec_snfr_px_cnt'] = ml_ec_summer_cnt_sh[1]
#             sh_summer_px_cnt.loc[date_time,'ml_ec_snc_px_cnt'] = ml_ec_summer_cnt_sh[2]
#             sh_summer_px_cnt.loc[date_time,'ml_ec_ice_px_cnt'] = ml_ec_summer_cnt_sh[3]

#             sh_summer_px_cnt.loc[date_time,'climatology_wtr_px_cnt'] = climatology_summer_cnt_sh[0]
#             sh_summer_px_cnt.loc[date_time,'climatology_snfr_px_cnt'] = climatology_summer_cnt_sh[1]
#             sh_summer_px_cnt.loc[date_time,'climatology_snc_px_cnt'] = climatology_summer_cnt_sh[2]
#             sh_summer_px_cnt.loc[date_time,'climatology_ice_px_cnt'] = climatology_summer_cnt_sh[3]

#             sh_summer_px_cnt.loc[date_time,'gmasi_wtr_px_cnt'] = gmasi_summer_cnt_sh[0]
#             sh_summer_px_cnt.loc[date_time,'gmasi_snfr_px_cnt'] = gmasi_summer_cnt_sh[1]
#             sh_summer_px_cnt.loc[date_time,'gmasi_snc_px_cnt'] = gmasi_summer_cnt_sh[2]
#             sh_summer_px_cnt.loc[date_time,'gmasi_ice_px_cnt'] = gmasi_summer_cnt_sh[3]

#             sh_summer_px_cnt.loc[date_time,'ml_ecc_wtr_px_cnt'] = ml_ecc_summer_cnt_sh[0]
#             sh_summer_px_cnt.loc[date_time,'ml_ecc_snfr_px_cnt'] = ml_ecc_summer_cnt_sh[1]
#             sh_summer_px_cnt.loc[date_time,'ml_ecc_snc_px_cnt'] = ml_ecc_summer_cnt_sh[2]
#             sh_summer_px_cnt.loc[date_time,'ml_ecc_ice_px_cnt'] = ml_ecc_summer_cnt_sh[3]

#             sh_summer_px_cnt.loc[date_time,'e_wtr_px_cnt'] = e_summer_cnt_sh[0]
#             sh_summer_px_cnt.loc[date_time,'e_snfr_px_cnt'] = e_summer_cnt_sh[1]
#             sh_summer_px_cnt.loc[date_time,'e_snc_px_cnt'] = e_summer_cnt_sh[2]
#             sh_summer_px_cnt.loc[date_time,'e_ice_px_cnt'] = e_summer_cnt_sh[3]

#             #----------------------------------------------------
#         elif season_sh is 'Autumn':
#             sh_autumn_ml_e.append(ml_e_arr_sh)
#             sh_autumn_gmasi.append(gmasi_arr_sh)
#             sh_autumn_ml_ec.append(ml_ec_arr_sh) 
#             sh_autumn_climatology.append(climatology_arr_sh) 
#             sh_autumn_ml_ecc.append(ml_ecc_arr_sh)
#             sh_autumn_e.append(e_arr_sh) 

#             ml_e_autumn_cnt_sh = count_class_pixels(ml_e_arr_sh)
#             ml_ec_autumn_cnt_sh = count_class_pixels(ml_ec_arr_sh)
#             ml_ecc_autumn_cnt_sh = count_class_pixels(ml_ecc_arr_sh)
#             climatology_autumn_cnt_sh = count_class_pixels(climatology_arr_sh)
#             gmasi_autumn_cnt_sh = count_class_pixels(gmasi_arr_sh)
#             e_autumn_cnt_sh = count_class_pixels(e_arr_sh)

#             sh_autumn_px_cnt.loc[date_time,'ml_e_wtr_px_cnt'] = ml_e_autumn_cnt_sh[0]
#             sh_autumn_px_cnt.loc[date_time,'ml_e_snfr_px_cnt'] = ml_e_autumn_cnt_sh[1]
#             sh_autumn_px_cnt.loc[date_time,'ml_e_snc_px_cnt'] = ml_e_autumn_cnt_sh[2]
#             sh_autumn_px_cnt.loc[date_time,'ml_e_ice_px_cnt'] = ml_e_autumn_cnt_sh[3]

#             sh_autumn_px_cnt.loc[date_time,'ml_ec_wtr_px_cnt'] = ml_ec_autumn_cnt_sh[0]
#             sh_autumn_px_cnt.loc[date_time,'ml_ec_snfr_px_cnt'] = ml_ec_autumn_cnt_sh[1]
#             sh_autumn_px_cnt.loc[date_time,'ml_ec_snc_px_cnt'] = ml_ec_autumn_cnt_sh[2]
#             sh_autumn_px_cnt.loc[date_time,'ml_ec_ice_px_cnt'] = ml_ec_autumn_cnt_sh[3]

#             sh_autumn_px_cnt.loc[date_time,'climatology_wtr_px_cnt'] = climatology_autumn_cnt_sh[0]
#             sh_autumn_px_cnt.loc[date_time,'climatology_snfr_px_cnt'] = climatology_autumn_cnt_sh[1]
#             sh_autumn_px_cnt.loc[date_time,'climatology_snc_px_cnt'] = climatology_autumn_cnt_sh[2]
#             sh_autumn_px_cnt.loc[date_time,'climatology_ice_px_cnt'] = climatology_autumn_cnt_sh[3]

#             sh_autumn_px_cnt.loc[date_time,'gmasi_wtr_px_cnt'] = gmasi_autumn_cnt_sh[0]
#             sh_autumn_px_cnt.loc[date_time,'gmasi_snfr_px_cnt'] = gmasi_autumn_cnt_sh[1]
#             sh_autumn_px_cnt.loc[date_time,'gmasi_snc_px_cnt'] = gmasi_autumn_cnt_sh[2]
#             sh_autumn_px_cnt.loc[date_time,'gmasi_ice_px_cnt'] = gmasi_autumn_cnt_sh[3]  

#             sh_autumn_px_cnt.loc[date_time,'ml_ecc_wtr_px_cnt'] = ml_ecc_autumn_cnt_sh[0]
#             sh_autumn_px_cnt.loc[date_time,'ml_ecc_snfr_px_cnt'] = ml_ecc_autumn_cnt_sh[1]
#             sh_autumn_px_cnt.loc[date_time,'ml_ecc_snc_px_cnt'] = ml_ecc_autumn_cnt_sh[2]
#             sh_autumn_px_cnt.loc[date_time,'ml_ecc_ice_px_cnt'] = ml_ecc_autumn_cnt_sh[3]

#             sh_autumn_px_cnt.loc[date_time,'e_wtr_px_cnt'] = e_autumn_cnt_sh[0]
#             sh_autumn_px_cnt.loc[date_time,'e_snfr_px_cnt'] = e_autumn_cnt_sh[1]
#             sh_autumn_px_cnt.loc[date_time,'e_snc_px_cnt'] = e_autumn_cnt_sh[2]
#             sh_autumn_px_cnt.loc[date_time,'e_ice_px_cnt'] = e_autumn_cnt_sh[3]

#         #------------------------------------------------------
        
#         # compare the extent (km^2) per autosnow class estimated by the model and original
#             # RF ERA5 based estimates
#         ml_e_vec_flnme = os.path.join(path_to_put_intermediate_files,os.path.basename(ml_e_read_nme).replace('.tif','.shp'))
#         ml_e_arr = ml_e_estimated_arr.astype(np.int16).copy()    
#         ml_e_gdf = array_to_vector(ml_e_arr, ml_e_vec_flnme, integer_list, 
#                                            meta_crs.to_string(),meta_trns,)  
        
#         area_extent.loc[date_time,'ml_e_wtr_total_area'] = get_total_area(ml_e_gdf,0)
#         area_extent.loc[date_time,'ml_e_snfr_total_area'] = get_total_area(ml_e_gdf,1)
#         area_extent.loc[date_time,'ml_e_snc_total_area'] = get_total_area(ml_e_gdf,2)
#         area_extent.loc[date_time,'ml_e_ice_total_area'] = get_total_area(ml_e_gdf,3)

#         ml_e_cnt = count_class_pixels(ml_e_arr.astype(np.int16))           
        
#         px_cnt.loc[date_time, 'ml_e_wtr_px_cnt'] = ml_e_cnt[0]
#         px_cnt.loc[date_time, 'ml_e_snfr_px_cnt'] = ml_e_cnt[1]
#         px_cnt.loc[date_time, 'ml_e_snc_px_cnt'] = ml_e_cnt[2]
#         px_cnt.loc[date_time, 'ml_e_ice_px_cnt'] = ml_e_cnt[3]

#         #------------------------------------------------------
        
#         # RF ERA5 + clim autosnow based estimates
#         ml_ec_vec_flnme = os.path.join(path_to_put_intermediate_files,os.path.basename(ml_ec_read_nme).replace('.tif','.shp'))
#         ml_ec_arr = ml_ec_estimated_arr.astype(np.int16).copy()    
#         ml_ec_gdf = array_to_vector(ml_ec_arr, ml_ec_vec_flnme, integer_list, 
#                                             meta_crs.to_string(),meta_trns,) 

#         ml_ec_cnt = count_class_pixels(ml_ec_arr.astype(np.int16)) 
        
#         area_extent.loc[date_time,'ml_ec_wtr_total_area'] = get_total_area(ml_ec_gdf,0)
#         area_extent.loc[date_time,'ml_ec_snfr_total_area'] = get_total_area(ml_ec_gdf,1)
#         area_extent.loc[date_time,'ml_ec_snc_total_area'] = get_total_area(ml_ec_gdf,2)
#         area_extent.loc[date_time,'ml_ec_ice_total_area'] = get_total_area(ml_ec_gdf,3)
        
#         px_cnt.loc[date_time, 'ml_ec_wtr_px_cnt'] = ml_ec_cnt[0]
#         px_cnt.loc[date_time, 'ml_ec_snfr_px_cnt'] = ml_ec_cnt[1]
#         px_cnt.loc[date_time, 'ml_ec_snc_px_cnt'] = ml_ec_cnt[2]
#         px_cnt.loc[date_time, 'ml_ec_ice_px_cnt'] = ml_ec_cnt[3]
#         #------------------------------------------------------

#         # original autosnow based estimates
#         gmasi_vec_flnme = os.path.join(path_to_put_intermediate_files,os.path.basename(l).replace('.tif','.shp'))
#         gmasi_array = gmasi_dat_array.astype(np.int16).copy()    
#         gmasi_gdf = array_to_vector(gmasi_array, gmasi_vec_flnme, integer_list, 
#                                             meta_crs.to_string(),meta_trns,) 
       
#         gmasi_cnt_sh = count_class_pixels(gmasi_array.astype(np.int16))
        
#         area_extent.loc[date_time,'gmasi_wtr_total_area'] = get_total_area(gmasi_gdf,0)
#         area_extent.loc[date_time,'gmasi_snfr_total_area'] = get_total_area(gmasi_gdf,1)
#         area_extent.loc[date_time,'gmasi_snc_total_area'] = get_total_area(gmasi_gdf,2)
#         area_extent.loc[date_time,'gmasi_ice_total_area'] = get_total_area(gmasi_gdf,3)
        
#         px_cnt.loc[date_time, 'gmasi_wtr_px_cnt'] = gmasi_cnt_sh[0]
#         px_cnt.loc[date_time, 'gmasi_snfr_px_cnt'] = gmasi_cnt_sh[1]
#         px_cnt.loc[date_time, 'gmasi_snc_px_cnt'] = gmasi_cnt_sh[2]
#         px_cnt.loc[date_time, 'gmasi_ice_px_cnt'] = gmasi_cnt_sh[3]  

#         #------------------------------------------------------
        
#         # corrected RF ERA5 + clim autosnow based estimates
#         ml_ecc_vec_flnme = os.path.join(path_to_put_intermediate_files,os.path.basename(ml_ecc_read_nme).replace('.tif','.shp'))
#         ml_ecc_arr = ml_ecc_estimated_arr.astype(np.int16).copy()    
#         ml_ecc_gdf = array_to_vector(ml_ecc_arr, ml_ecc_vec_flnme,
#                                                 integer_list, meta_crs.to_string(),meta_trns,)  
        
#         ml_ecc_cnt = count_class_pixels(ml_ecc_arr.astype(np.int16))
        
#         area_extent.loc[date_time,'ml_ecc_wtr_total_area'] = get_total_area(ml_ecc_gdf,0)
#         area_extent.loc[date_time,'ml_ecc_snfr_total_area'] = get_total_area(ml_ecc_gdf,1)
#         area_extent.loc[date_time,'ml_ecc_snc_total_area'] = get_total_area(ml_ecc_gdf,2)
#         area_extent.loc[date_time,'ml_ecc_ice_total_area'] = get_total_area(ml_ecc_gdf,3)
        
#         px_cnt.loc[date_time, 'ml_ecc_wtr_px_cnt'] = ml_ecc_cnt[0]
#         px_cnt.loc[date_time, 'ml_ecc_snfr_px_cnt'] = ml_ecc_cnt[1]
#         px_cnt.loc[date_time, 'ml_ecc_snc_px_cnt'] = ml_ecc_cnt[2]
#         px_cnt.loc[date_time, 'ml_ecc_ice_px_cnt'] = ml_ecc_cnt[3]

#         #------------------------------------------------------
        
#         # temp clim autosnow based estimates
#         climatology_vec_flnme = os.path.join(path_to_put_intermediate_files,
#                                              os.path.basename(climatology_read_nme).replace('.tif','.shp'))
#         climatology_arr = climatology_estimated_arr.astype(np.int16).copy()    
#         climatology_gdf = array_to_vector(climatology_estimated_arr, climatology_vec_flnme, integer_list, 
#                                                   meta_crs.to_string(),meta_trns,)  

#         climatology_cnt_sh = count_class_pixels(climatology_arr.astype(np.int16))
        
#         area_extent.loc[date_time,'climatology_wtr_total_area'] = get_total_area(climatology_gdf,0)
#         area_extent.loc[date_time,'climatology_snfr_total_area'] = get_total_area(climatology_gdf,1)
#         area_extent.loc[date_time,'climatology_snc_total_area'] = get_total_area(climatology_gdf,2)
#         area_extent.loc[date_time,'climatology_ice_total_area'] = get_total_area(climatology_gdf,3)
        
#         px_cnt.loc[date_time, 'climatology_wtr_px_cnt'] = climatology_cnt_sh[0]
#         px_cnt.loc[date_time, 'climatology_snfr_px_cnt'] = climatology_cnt_sh[1]
#         px_cnt.loc[date_time, 'climatology_snc_px_cnt'] = climatology_cnt_sh[2]
#         px_cnt.loc[date_time, 'climatology_ice_px_cnt'] = climatology_cnt_sh[3]      

#         #------------------------------------------------------
        
#         # e estimates
#         e_vec_flnme = os.path.join(path_to_put_intermediate_files,
#                                              os.path.basename(e_read_nme).replace('.tif','.shp'))
#         e_arr = e_estimated_arr.astype(np.int16).copy()    
#         e_gdf = array_to_vector(e_estimated_arr, e_vec_flnme, integer_list, meta_crs.to_string(),meta_trns,)  

#         e_cnt_sh = count_class_pixels(e_arr.astype(np.int16))
        
#         area_extent.loc[date_time,'e_wtr_total_area'] = get_total_area(e_gdf,0)
#         area_extent.loc[date_time,'e_snfr_total_area'] = get_total_area(e_gdf,1)
#         area_extent.loc[date_time,'e_snc_total_area'] = get_total_area(e_gdf,2)
#         area_extent.loc[date_time,'e_ice_total_area'] = get_total_area(e_gdf,3)
        
#         px_cnt.loc[date_time, 'e_wtr_px_cnt'] = e_cnt_sh[0]
#         px_cnt.loc[date_time, 'e_snfr_px_cnt'] = e_cnt_sh[1]
#         px_cnt.loc[date_time, 'e_snc_px_cnt'] = e_cnt_sh[2]
#         px_cnt.loc[date_time, 'e_ice_px_cnt'] = e_cnt_sh[3]    

#         #------------------------------------------------------

#         count += 1

#         if count % 25 == 0:
#             print(str(count) + ' files are read so far')

#------------------------------------------------
# #   ('E',nh_summer_hits_2d_e, nh_summer_miss_2d_e),
# plot_percent_hit_mis(nh_summer_arr2_plt,plt_extent_nh, range(45, 90, 15))




# print('done with nh')
#--------------------------------------------------------------------------------------------

# sh_arr2_plt = [('ML-E',sh_hits_2d_ml_e, sh_miss_2d_ml_e), 
#                ('ML-EC',sh_hits_2d_ml_ec, sh_miss_2d_ml_ec),
#                ('ML-ECC',sh_hits_2d_ml_ecc, sh_miss_2d_ml_ecc),               
#                ('CLIM',sh_hits_2d_climatology, sh_miss_2d_climatology)]
# # ('E',sh_hits_2d_e, sh_miss_2d_e),
# plot_percent_hit_mis(sh_arr2_plt,plt_extent_sh, range(-90, -45, 15))

# svenme = '_'.join(['percent_hit_miss_sh',cde_run_dte]) + '.png'
# plt.savefig(os.path.join(path_to_put_plots,svenme),bbox_inches='tight')
# plt.close()

# #--------------------------------------------------------------------------------------------


# sh_winter_arr2_plt = [('ML-E',sh_winter_hits_2d_ml_e, sh_winter_miss_2d_ml_e), 
#                       ('ML-EC',sh_winter_hits_2d_ml_ec, sh_winter_miss_2d_ml_ec),
#                       ('ML-ECC',sh_winter_hits_2d_ml_ecc, sh_winter_miss_2d_ml_ecc),                      
#                       ('CLIM',sh_winter_hits_2d_climatology, sh_winter_miss_2d_climatology)]
# # ('E',sh_winter_hits_2d_e, sh_winter_miss_2d_e),
# plot_percent_hit_mis(sh_winter_arr2_plt,plt_extent_sh, range(-90, -45, 15))

# svenme = '_'.join(['winter_percent_hit_miss_sh',cde_run_dte]) + '.png'
# plt.savefig(os.path.join(path_to_put_plots,svenme),bbox_inches='tight')
# plt.close()

# #--------------------------------------------------------------------------------------------

# sh_summer_arr2_plt = [('ML-E',sh_summer_hits_2d_ml_e, sh_summer_miss_2d_ml_e), 
#                       ('ML-EC',sh_summer_hits_2d_ml_ec, sh_summer_miss_2d_ml_ec),
#                       ('ML-ECC',sh_summer_hits_2d_ml_ecc, sh_summer_miss_2d_ml_ecc),                      
#                       ('CLIM',sh_summer_hits_2d_climatology, sh_summer_miss_2d_climatology)]
# # ('E',sh_summer_hits_2d_e, sh_summer_miss_2d_e),
# plot_percent_hit_mis(sh_summer_arr2_plt,plt_extent_sh, range(-90, -45, 15))

# svenme = '_'.join(['summer_percent_hit_miss_sh',cde_run_dte]) + '.png'
# plt.savefig(os.path.join(path_to_put_plots,svenme),bbox_inches='tight')
# plt.close()

#---------------------------------------------------
# axes[0,0].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-E-GMASI-hit'],ls='-',c=colors[0], lw=lw,label='ML-E')
# axes[0,0].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-EC-GMASI-hit'],ls='-',c=colors[1], lw=lw, label='ML-EC')
# axes[0,0].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-ECC-GMASI-hit'],ls='-',c=colors[6], lw=lw, label='ML-ECC')
# # axes[0,0].plot(hit_miss_df.index,hit_miss_df.loc[:,'E-GMASI-hit'],ls='-',c=colors[7], lw=lw, label='E')
# axes[0,0].plot(hit_miss_df.index,hit_miss_df.loc[:,'Climatology-GMASI-hit'],ls='-',c=colors[4], lw=lw, label='CLIM')


# # axes[0,0].xaxis.tick_top()
# axes[0,0].set_ylim(95, 100)
# axes[0,0].set_title('Combined')
# axes[0,0].set_ylabel('Fraction of\n match [%]',fontsize=13)
# # axes[0,0].legend(loc='best',frameon=False,fontsize=7)

# y_ticks = axes[0,0].get_yticks()
# # Round the tick values to desired precision, here rounding to nearest whole number
# # Convert the tick values to integers
# int_ticks = [int(val) for val in y_ticks]
# # Set new, rounded y-axis tick labels
# axes[0,0].set_yticklabels(int_ticks)

# axes[0,1].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-E-wter-hit'],ls='-',c=colors[0], lw=lw,)
# axes[0,1].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-EC-wter-hit'],ls='-',c=colors[1], lw=lw, )
# axes[0,1].plot(hit_miss_df.index,hit_miss_df.loc[:,'Climatology-wter-hit'],ls='-',c=colors[4], lw=lw,)
# axes[0,1].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-ECC-wter-hit'],ls='-',c=colors[6], lw=lw,)
# # axes[0,1].plot(hit_miss_df.index,hit_miss_df.loc[:,'E-wter-hit'],ls='-',c=colors[7], lw=lw,)


# # axes[0,1].xaxis.tick_top()
# axes[0,1].set_ylim(97, 100) # 85
# # axes[0,1].set_yticklabels([])
# axes[0,1].set_title('Water', fontsize=15)

# axes[0,2].plot(hit_miss_df.index, hit_miss_df.loc[:,'ML-E-snflnd-hit'],ls='-',c=colors[0], lw=lw,)
# axes[0,2].plot(hit_miss_df.index, hit_miss_df.loc[:,'ML-EC-snflnd-hit'],ls='-',c=colors[1], lw=lw,)
# axes[0,2].plot(hit_miss_df.index, hit_miss_df.loc[:,'Climatology-snflnd-hit'],ls='-',c=colors[4], lw=lw,)
# axes[0,2].plot(hit_miss_df.index, hit_miss_df.loc[:,'ML-ECC-snflnd-hit'],ls='-',c=colors[6], lw=lw,)
# # axes[0,2].plot(hit_miss_df.index, hit_miss_df.loc[:,'E-snflnd-hit'],ls='-',c=colors[7], lw=lw,)

# # axes[0,2].xaxis.tick_top()
# axes[0,2].set_ylim(90, 100)
# # axes[0,2].set_yticklabels([])
# axes[0,2].set_title('Snow free', fontsize=15)

# axes[0,3].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-E-snclnd-hit'],ls='-',c=colors[0], lw=lw,)
# axes[0,3].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-EC-snclnd-hit'],ls='-',c=colors[1], lw=lw,)
# axes[0,3].plot(hit_miss_df.index,hit_miss_df.loc[:,'Climatology-snclnd-hit'],ls='-',c=colors[4], lw=lw,)
# axes[0,3].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-ECC-snclnd-hit'],ls='-',c=colors[6], lw=lw,)
# # axes[0,3].plot(hit_miss_df.index,hit_miss_df.loc[:,'E-snclnd-hit'],ls='-',c=colors[7], lw=lw,)

# # axes[0,3].xaxis.tick_top()
# axes[0,3].set_ylim(90, 100)
# # axes[0,3].set_yticklabels([])
# axes[0,3].set_title('Snow cover', fontsize=15)

# axes[0,4].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-E-ice-hit'],ls='-',c=colors[0], lw=lw,)
# axes[0,4].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-EC-ice-hit'],ls='-',c=colors[1], lw=lw)
# axes[0,4].plot(hit_miss_df.index,hit_miss_df.loc[:,'Climatology-ice-hit'],ls='-',c=colors[4], lw=lw,)
# axes[0,4].plot(hit_miss_df.index,hit_miss_df.loc[:,'ML-ECC-ice-hit'],ls='-',c=colors[6], lw=lw)
# # axes[0,4].plot(hit_miss_df.index,hit_miss_df.loc[:,'E-ice-hit'],ls='-',c=colors[7], lw=lw)

# axes[0,4].set_ylim(85, 100) # 
# # axes[0,4].set_yticklabels([])
# axes[0,4].set_title('Ice', fontsize=15)
'''
fig,axes = plt.subplots(2, 2,figsize=(10,5), sharex=True,dpi=1000,                         
                         gridspec_kw={'width_ratios': [0.98]*2},) # gridspec_kw={'width_ratios': [0.98]*2},
plt.subplots_adjust(bottom=0.2)
lw= 1
axes[0,0].plot(area_extent.index, area_extent.loc[:,'ml_e_gmasi_wtr_diff'],
               ls='-', c=colors[0], lw=lw,)
axes[0,0].plot(area_extent.index, area_extent.loc[:,'ml_ec_gmasi_wtr_diff'],
               ls='-', c=colors[1], lw=lw, )
axes[0,0].plot(area_extent.index, area_extent.loc[:,'climatology_gmasi_wtr_diff'],
               ls='-', c=colors[4], lw=lw, )
axes[0,0].plot(area_extent.index, area_extent.loc[:,'ml_ecc_gmasi_wtr_diff'],
               ls='-', c=colors[6], lw=lw,)

# Apply the custom y-axis label formatting
# axes[0,0].yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))

# Add text to the upper-left corner of the entire figure
axes[0,0].text(0.02, 0.98, 'Water', ha='left', va='top', fontsize=15,transform=axes[0, 0].transAxes)

axes[0,0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[12]))
axes[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#--------------------

axes[0,1].plot(area_extent.index, area_extent.loc[:,'ml_e_gmasi_snfr_diff'],
               ls='-', c=colors[0], lw=lw, )
axes[0,1].plot(area_extent.index, area_extent.loc[:,'ml_ec_gmasi_snfr_diff'],
               ls='-', c=colors[1], lw=lw, )
axes[0,1].plot(area_extent.index, area_extent.loc[:,'climatology_gmasi_snfr_diff'],
               ls='-', c=colors[4], lw=lw, )
axes[0,1].plot(area_extent.index, area_extent.loc[:,'ml_ecc_gmasi_snfr_diff'],
               ls='-', c=colors[6], lw=lw, )
# axes[0,1].plot(area_extent.index, area_extent.loc[:,'e_gmasi_snfr_diff'],
#                ls='-', c=colors[7], lw=lw, )

# axes[0,1].yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))

axes[0,1].text(0.02, 0.98, 'Snow free', ha='left', va='top', fontsize=15, transform=axes[0, 1].transAxes)

axes[0,1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[12]))
axes[0,1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#--------------------

axes[1,0].plot(area_extent.index, area_extent.loc[:,'ml_e_gmasi_snc_diff'],
               ls='-', c=colors[0], lw=lw, )
axes[1,0].plot(area_extent.index,area_extent.loc[:,'ml_ec_gmasi_snc_diff'],
               ls='-', c=colors[1], lw=lw,)
axes[1,0].plot(area_extent.index,area_extent.loc[:,'climatology_gmasi_snc_diff'],
               ls='-', c=colors[4], lw=lw, )
axes[1,0].plot(area_extent.index,area_extent.loc[:,'ml_ecc_gmasi_snc_diff'],
               ls='-', c=colors[6], lw=lw, )
# axes[1,0].plot(area_extent.index,area_extent.loc[:,'e_gmasi_snc_diff'],
#                ls='-', c=colors[7], lw=lw, )

# axes[1,0].yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))

axes[1,0].text(0.02, 0.98, 'Snow cover', ha='left', va='top', fontsize=15, transform=axes[1, 0].transAxes)

axes[1,0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[12]))
axes[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#--------------------

axes[1,1].plot(area_extent.index, area_extent.loc[:,'ml_e_gmasi_ice_diff'],
               ls='-', c=colors[0], lw=lw, label='ML-E')
axes[1,1].plot(area_extent.index,area_extent.loc[:,'ml_ec_gmasi_ice_diff'],
               ls='-', c=colors[1], lw=lw, label='ML-EC')
axes[1,1].plot(area_extent.index,area_extent.loc[:,'climatology_gmasi_ice_diff'],
               ls='-', c=colors[4], lw=lw, label='CLIM')
axes[1,1].plot(area_extent.index,area_extent.loc[:,'ml_ecc_gmasi_ice_diff'],
               ls='-', c=colors[6], lw=lw, label='ML-ECC')
# axes[1,1].plot(area_extent.index,area_extent.loc[:,'e_gmasi_ice_diff'],
#                ls='-', c=colors[7], lw=lw, label='E')

# axes[1,1].yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))

axes[1,1].text(0.02, 0.98, 'Ice', ha='left', va='top', fontsize=15, transform=axes[1,1].transAxes)

axes[1,1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[12]))
axes[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

for ax in axes.flatten():    

    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, 
                   right=True, bottom=True, left=True)
    ax.grid(which='major', linestyle='--', linewidth='0.5', color='grey')

fig.text(0.08, 0.5, 'Land surface cover type extent anomaly [%]', ha='center', va='center', 
         rotation='vertical', fontsize=12)

fig.text(0.5,0.12, 'Year', ha='center', va='center', 
         rotation='horizontal', fontsize=15)

# Common legend below the plot
lines, labels = axes[1, 1].get_legend_handles_labels()
fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.035), 
           fontsize=15, ncol=5,columnspacing=1,frameon=False) 

svenme = '_'.join(['surface_cover_type_extent_anomaly',cde_run_dte]) + '.png'
plt.savefig(os.path.join(path_to_put_plots,svenme),bbox_inches='tight')
plt.close()
'''