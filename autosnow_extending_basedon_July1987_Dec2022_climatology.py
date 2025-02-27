'''
Author: Dr K. K. Kumah, Dec 20, 2023

This code estimates autosnow data based on July 1987 - Dec 2022 autsnow climatology.
Basically, we are classifying a pixel as water, snow free, snow covered or ice based 
on the 35 years of available data

Also, the probability of the pixel been a certain class is calculated as the number
of times the pixel has been that class divided by the number of observations, i.e., 35

This data is stored and used togther with ERA5 land suarface variables for generating
autosnow for periods between Jan 1980 - mid 1987
'''

# %%
# import packages
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import map_manipulation_functions as my_map_functions
import evaluation_fucntions_algorithms as my_eval_func

import scipy.stats as stats

import xarray as xr
import rasterio
#%%
# define paths
path_to_autosnow_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_in_geotif'
path_to_save_df = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/save_dfs_Oct'
path_to_put_intermediate_files = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/alldata_climatology_based_autosnow_estimate'

#%%
# decalre global variables
# sections of the autosnow file name
autosnow_nme_part1 = 'gmasi_snowice_reproc_v003'
autosnow_nme_part2 = '0.1deg_wgs.tif'

# get the shape of the autosnow array
file_sample = os.path.join(path_to_autosnow_data,'gmasi_snowice_reproc_v003_1987184_0.1deg_wgs.tif')
with xr.open_dataarray(file_sample)as dt:

    dat = dt.data[0]

    y_shp,x_shp = dat.shape
#--------------------------------------------------
# collect the meta data of the autosnow file which will be used to save the newly comoputed files
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

# 0 = Water; 1 = Snow free land; 2 = Snow covered land; 3 = Ice

#%%
# define fucntions
# 
def count_occurrences(lst, target):
    '''
    this function is used to count the number of times a pixel belong to a particular autosnow class
    requires:
    lst = list of numbers
    target = target value to find its occurrences
    '''
    count = 0
    for num in lst:
        if num == target:
            count += 1
    return count

#%%
print('************ begin estimation ********************')
print('*********************************************************************')
print('*********************************************************************')
print('*********************************************************************')

all_autosn_obs_arr_in_yr = []

all_autosn_est_arr_in_yr = []

count = 0

for d in range(1,367):

    lst_autosnow_ars = []

    if len(str(d)) == 1:
        dofy = '00' + str(d)

    elif len(str(d)) == 2:

        dofy = '0' + str(d)

    else:
        dofy = str(d)          

    for y in range(1987,2023):

        yr = str(y)

        est_yr_doy = ''.join([yr,dofy])

        autosnow_file = '_'.join([autosnow_nme_part1,est_yr_doy,autosnow_nme_part2])

        autosnow_file_to_read = os.path.join(path_to_autosnow_data,autosnow_file)

        if os.path.isfile(autosnow_file_to_read):

            dat_auto = xr.open_dataarray(autosnow_file_to_read)        

            dat_auto = dat_auto.data[0,:,:]

            dat_auto = np.where(dat_auto > 3, np.nan,dat_auto)

            lst_autosnow_ars.append(dat_auto)

    autosnow_3d = my_map_functions.make_3d_array(x_shp,y_shp,lst_autosnow_ars)

    estimated_autosnow_arr = np.apply_along_axis(lambda a: stats.mode(a)[0][0], 2, autosnow_3d)

    # find the number of times a pixel belong to a particular autosnow class
    #  and use it to compute the probability of its class
    water_class_count = np.apply_along_axis(lambda a: count_occurrences(a,0),2,autosnow_3d)
    water_class_prob = np.round(water_class_count/autosnow_3d.shape[2],3) 

    snow_free_land_class_count = np.apply_along_axis(lambda a: count_occurrences(a,1),2,autosnow_3d)
    snow_free_land_class_prob = np.round(snow_free_land_class_count/autosnow_3d.shape[2],3)

    snow_covered_land_class_count = np.apply_along_axis(lambda a: count_occurrences(a,2),2,autosnow_3d)
    snow_covered_land_class_prob = np.round(snow_covered_land_class_count/autosnow_3d.shape[2],3)

    ice_class_count = np.apply_along_axis(lambda a: count_occurrences(a,3),2,autosnow_3d)
    ice_class_prob = np.round(ice_class_count/autosnow_3d.shape[2],3)

    clm_std_arr = np.apply_along_axis(lambda a: round(np.std(a),3), 2, autosnow_3d)

    # save the estimated autosnow and std for use with machine learning techn
    yr_doy_sve = '1987_2022_DOY_' + dofy
    autsnow_estimates_svenme = '_'.join(['autosnow_based_on_alldata_clim',yr_doy_sve,autosnow_nme_part2])
    my_map_functions.rasterio_based_save_array_to_disk(path_to_put_intermediate_files, autsnow_estimates_svenme,
                                                    autosnow_meta, estimated_autosnow_arr)    
    
    autsnow_std_svenme = '_'.join(['std_of_autosnow_based_on_alldata_clim',yr_doy_sve,autosnow_nme_part2])
    my_map_functions.rasterio_based_save_array_to_disk(path_to_put_intermediate_files, autsnow_std_svenme,
                                                    autosnow_meta_cpy, clm_std_arr)

    water_prob_svenme = '_'.join(['water_class_prob_based_on_all_data_clim',yr_doy_sve,autosnow_nme_part2])
    my_map_functions.rasterio_based_save_array_to_disk(path_to_put_intermediate_files, water_prob_svenme,
                                                    autosnow_meta, water_class_prob)    
    
    snow_free_land_svenme = '_'.join(['snow_free_land_class_prob_based_on_all_data_clim',yr_doy_sve,autosnow_nme_part2])
    my_map_functions.rasterio_based_save_array_to_disk(path_to_put_intermediate_files, snow_free_land_svenme,
                                                    autosnow_meta_cpy, snow_free_land_class_prob)
    
    snow_covered_land_svenme = '_'.join(['snow_covered_land_class_prob_based_on_all_data_clim',yr_doy_sve,autosnow_nme_part2])
    my_map_functions.rasterio_based_save_array_to_disk(path_to_put_intermediate_files, snow_covered_land_svenme,
                                                    autosnow_meta_cpy, snow_covered_land_class_prob)
    
    ice_svenme = '_'.join(['ice_class_prob_based_on_all_data_clim',yr_doy_sve,autosnow_nme_part2])
    my_map_functions.rasterio_based_save_array_to_disk(path_to_put_intermediate_files, ice_svenme,
                                                    autosnow_meta_cpy, ice_class_prob)
    
    count = count + 1

    if count % 100 == 0:
        print(str(count) + ' daily files processed so far') 

print('******************* done with estimation ****************************')
print('*********************************************************************')
print('*********************************************************************')
print('*********************************************************************')
