'''
Author: Dr K. K. Kumah

estimating autosnow data by using past all data (from all years) climatology. 
Basically, we are classifying a pixel, in the past) as water, snow free, snow covered or ice 
based on what that pixel has mostly been in the future. 
However, we do not use all data from the future, but we only use future data with air tempearture
close to the past air temperature in question.
Example: 
    suppose we are doing classification for 1988 using future 20 years (1989-2009), each day in 1989 will
    have 20 samples based on which we do our classification by using mode of the samples. However, we
    do not use all 20 samples but subset of the samples which has air temperature closer (i.e. small difference)
    to the day (in question) in 1989
'''

# %%
# import packages
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import map_manipulation_functions as my_map_functions
import evaluation_fucntions_algorithms as my_eval_func

import scipy.stats as stats

import xarray as xr
import rasterio
from rasterio.warp import Resampling
from pyproj import CRS
#%%
# define paths
path_to_autosnow_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_in_geotif'
path_to_save_df = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/save_dfs_Oct'
path_to_put_intermediate_files = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/alldata_climatology_based_autosnow_estimate'
path_to_era5_daily_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/era5_daily_data_for_extending_autosnow'
path_to_era5_2mairTeemp_mnth_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/era5_nc_files_monthly'
#%%
# decalre global variables
autosnow_nme_part1 = 'gmasi_snowice_reproc_v003'
autosnow_nme_part2 = '0.1deg_wgs.tif'

# get the shape of the autosnow array
file_sample = os.path.join(path_to_autosnow_data,'gmasi_snowice_reproc_v003_1987184_0.1deg_wgs.tif')
with xr.open_dataarray(file_sample)as dt:

    dat = dt.data[0]

    y_shp,x_shp = dat.shape
#--------------------------------------------------
autosnow_meta = rasterio.open(file_sample).meta

autosnow_meta_cpy = autosnow_meta.copy()
autosnow_meta_cpy.update({'dtype': np.float32})
#--------------------------------------------------

# make list of row, col indexes
lst_row_col_idx = []

for row in range(0,y_shp):

    for col in range(0,x_shp):

        id_tuple = tuple((row,col))

        lst_row_col_idx.append(id_tuple)

classes = ['Water','Snow free land', 'Snow covered land','Ice']

# read the aitemperature for this day
obs_day_airTemp_file = os.path.join(path_to_era5_daily_data,'2m_temperature_1988_daily_mean.nc')

obs_day_airTemp = xr.open_dataarray(os.path.join(path_to_autosnow_data,obs_day_airTemp_file))

cc = CRS.from_authority(code=4326,auth_name='EPSG')

#%%
print('************ begin estimation ********************')

all_autosn_obs_arr_in_yr = []

all_autosn_est_arr_in_yr = []
count = 0
for d in range(1,367):

    estimated_autosnow_arr = np.empty(dat.shape)

    lst_autosnow_ars = []

    if len(str(d)) == 1:
        dofy = '00' + str(d)

    elif len(str(d)) == 2:

        dofy = '0' + str(d)

    else:
        dofy = str(d)  


    # read the original autosnow file
    obs_yr_doy = ''.join(['1988',dofy])    

    obs_autosnow_file = '_'.join([autosnow_nme_part1,obs_yr_doy,autosnow_nme_part2])

    obs_autosn = xr.open_dataarray(os.path.join(path_to_autosnow_data,obs_autosnow_file))       

    obs_autosn = obs_autosn.data[0,:,:]

    obs_autosn = np.where(obs_autosn > 3, np.nan,obs_autosn)

    all_autosn_obs_arr_in_yr.append(obs_autosn)  

    datetime_of_obs_day = datetime(1988, 1, 1) + timedelta(days=d - 1)

    obs_2m_airTemp_control = obs_day_airTemp.sel(time = datetime_of_obs_day).values

    #-------------------------------------------------  
    air_temp_diff_list = []
    for y in range(1989,2009):

        yr = str(y)

        est_yr_doy = ''.join([yr,dofy])

        autosnow_file = '_'.join([autosnow_nme_part1,est_yr_doy,autosnow_nme_part2])

        autosnow_file_to_read = os.path.join(path_to_autosnow_data,autosnow_file)

        if os.path.isfile(autosnow_file_to_read):

            airTemp_file_for_est = os.path.join(path_to_era5_2mairTeemp_mnth_data,'2m_temperature_monthly_mean_' 
                                                + str(y) + '_download.nc')

            obs_day_airTemp_for_est = xr.open_dataarray(os.path.join(path_to_autosnow_data,airTemp_file_for_est))

            obs_day_airTemp_for_est = obs_day_airTemp_for_est.where(obs_day_airTemp_for_est < 400,np.nan)

            # spatially resample the monthly data to match the daily data
            # write crs to data
            obs_day_airTemp_for_est.rio.write_crs(cc.to_string(), inplace=True)

            # resample data spatially to autosnow spatial resolution (0.1 deg for our study case)
            obs_day_airTemp_for_est_res = obs_day_airTemp_for_est.rio.reproject(
            obs_day_airTemp_for_est.rio.crs,
            shape=obs_2m_airTemp_control.shape, # set the shape as the autosnow data shape
            resampling=Resampling.bilinear,
            )

            datetime_of_test_day = datetime(y, 1, 1) + timedelta(days=d - 1)

            test_tme = pd.to_datetime(datetime_of_test_day.strftime('%Y-%m'))

            obs_2m_airTemp_test = obs_day_airTemp_for_est_res.sel(time = test_tme).values

            # find the difference between the 2m air temp on the obs day and the test day
            air_temp_diff = np.abs(obs_2m_airTemp_control - obs_2m_airTemp_test)

            air_temp_diff_list.append(air_temp_diff)

            #----------------------------------------
            # read the autosnow file
            dat_auto = xr.open_dataarray(autosnow_file_to_read)        

            dat_auto = dat_auto.data[0,:,:]

            dat_auto = np.where(dat_auto > 3, np.nan,dat_auto)

            lst_autosnow_ars.append(dat_auto)

            
    airtemp_diff_3d = my_map_functions.make_3d_array(x_shp,y_shp,air_temp_diff_list)

    # out_arr = np.apply_along_axis(lambda a: stats.rankdata(a, method='min',nan_policy='omit'), 2, airtemp_diff_3d)
    # rank the temp differences from least to largest
    out_arr_rnk = stats.mstats.rankdata(airtemp_diff_3d, axis=2, use_missing=False)

    autosnow_3d = my_map_functions.make_3d_array(x_shp,y_shp,lst_autosnow_ars)

    autosnow_3d_sampled_by_rank = np.where(out_arr_rnk > 11,np.nan,autosnow_3d)

    estimated_autosnow_arr = np.apply_along_axis(lambda a: stats.mode(a,nan_policy='omit')[0][0], 2, autosnow_3d_sampled_by_rank)

    # clm_std_arr = np.apply_along_axis(lambda a: round(np.std(a),3), 2, autosnow_3d)

    # save the estimated autosnow and std for use with machine learning techn
    # autsnow_estimates_svenme = '_'.join(['alldata_clim_bsed',obs_yr_doy,autosnow_nme_part2])
    # my_map_functions.rasterio_based_save_array_to_disk(path_to_put_intermediate_files, autsnow_estimates_svenme,
    #                                                    autosnow_meta, estimated_autosnow_arr)    
    
    # autsnow_std_svenme = '_'.join(['std_alldata_clim_bsed',obs_yr_doy,autosnow_nme_part2])
    # my_map_functions.rasterio_based_save_array_to_disk(path_to_put_intermediate_files, autsnow_std_svenme,
    #                                                    autosnow_meta_cpy, clm_std_arr)

    all_autosn_est_arr_in_yr.append(estimated_autosnow_arr)

    count = count + 1

    if count % 50 == 0:
        print(str(count) + ' Autosnow files processed so far') 

print('******************* done with estimation ****************************')
print('*********************************************************************')
print('*********************************************************************')
print('*********************************************************************')

# autosnow_rf_diff = obs_autosn - estimated_autosnow_arr
# autosnow_rf_diff[autosnow_rf_diff==0.]= np.nan   
#%%
print('******************* begin evaluation ******************* ')
# compute metrics
autosn_est_1d = my_map_functions.make_3d_array(x_shp,y_shp,all_autosn_est_arr_in_yr)
autosn_est_1d = autosn_est_1d.reshape((y_shp*x_shp*autosn_est_1d.shape[2]))

autosn_obs_1d = my_map_functions.make_3d_array(x_shp,y_shp,all_autosn_obs_arr_in_yr)
autosn_obs_1d = autosn_obs_1d.reshape((y_shp*x_shp*autosn_obs_1d.shape[2]))

#-------------------------------------------------------

# single day analysis for present
# here we compared what we have with all data climatology results
clim_file_to_comp = os.path.join(r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/alldata_climatology_based_autosnow_estimate',
                                 'alldata_clim_bsed_1988320_0.1deg_wgs.tif') # d = 20, 320
clim_aut_lst = []
clim_dat_auto = xr.open_dataarray(clim_file_to_comp)    
clim_dat_auto = clim_dat_auto.data[0,:,:]
clim_aut_lst.append(clim_dat_auto)

clim_auto_1d = my_map_functions.make_3d_array(x_shp,y_shp,clim_aut_lst)
clim_auto_1d = autosn_est_1d.reshape((y_shp*x_shp*clim_auto_1d.shape[2]))
#-------------------------------------------------------

df_obs_est = pd.DataFrame(np.column_stack((autosn_obs_1d,autosn_est_1d)),columns=['obs','est'])

n_per_class_in_y = pd.DataFrame(df_obs_est['obs'].value_counts())

n_per_class_in_y['Class']  = classes

lst_cat_stats = []
#-------------------------------------------------------

for i in enumerate(classes):

    tval = float(i[0])

    class_val = i[1]

    lst_cat_stats.append(my_eval_func.binary_cat_metrics(df_obs_est,'obs','est',tval,class_val))

dfs_cat_stats = pd.concat(lst_cat_stats,axis=0)

dfs_cat_stats = dfs_cat_stats.merge(n_per_class_in_y[['count', 'Class']],right_on='Class',
                                    left_on=dfs_cat_stats.index)

dfs_cat_stats['label'] = dfs_cat_stats.index

dfs_cat_stats.index = dfs_cat_stats['Class']

dfs_cat_stats.drop(columns='Class',inplace=True)

dfs_cat_stats = pd.DataFrame(dfs_cat_stats.filter(items=['count','Hits', 'Miss', 'False alarms',
                                                         'label','POD', 'FAR', 'POFD', 'ACC', 'CSI',
                                                         'ETS']))

dfs_cat_stats.to_csv(os.path.join(path_to_save_df,'categorical_stats_alldata_climatologiy_subsetted_by_temp_scenario_results_16_Oct.csv'))

print('*********************************************************************')
print('*********************************************************************')
print('*********************************************************************')
print('done!')

#%%%


#%%
# std_try = rasterio.open(os.path.join(path_to_put_intermediate_files, autsnow_estimates_svenme))
    
# std_try_arr = std_try.read()[0]

# for i in enumerate(classes):

#%%
'''
airtemp_diff_3d_0 = airtemp_diff_3d[:,:,0]
airtemp_diff_3d_1 = airtemp_diff_3d[:,:,1]
airtemp_diff_3d_2 = airtemp_diff_3d[:,:,2]

out_arr_0 = out_arr_[:,:,0]
out_arr_1 = out_arr_[:,:,1]
out_arr_2 = out_arr_[:,:,2]

import numpy as np

# Create or load your 3D array (replace this with your data)
# For the example, we'll create a random 3D array.
array_3d = np.random.rand(3, 4, 5)

# Calculate ranked indices for the elements in the 3D array
ranked_indices = np.argsort(array_3d, axis=None)

# If you want to access elements based on their rank, you can use the ranked indices
# For example, to get the element with the highest rank:
highest_ranked_element = array_3d.flat[ranked_indices[-1]]

# Or, to get the element with the lowest rank:
lowest_ranked_element = array_3d.flat[ranked_indices[0]]

# You can also reshape the ranked indices to match the shape of the original array
ranked_indices_3d = ranked_indices.reshape(array_3d.shape)

'''