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

from concurrent.futures import ProcessPoolExecutor
#%%
# define paths
path_to_autosnow_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_in_geotif'
path_to_save_df = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/save_dfs_Oct'
path_to_put_intermediate_files = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/extending_autosnow_estimated/all_data_1992_2022_airTemp_subsetted_climatology_estimate'
#r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/alldata_climatology_based_autosnow_estimate'
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
autosnow_meta_cpy = rasterio.open(file_sample).meta.copy()

# autosnow_meta_cpy = autosnow_meta_cpy.copy()
# autosnow_meta_cpy.update({'dtype': np.float32})
#--------------------------------------------------
cc = CRS.from_authority(code=4326,auth_name='EPSG')

#%%
# define functions
# function to handle leap year issue in data
def year_day_to_datetime(year, day_of_year):
    # Check if the year is a leap year
    is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    # Adjust the day of the year for leap years
    day_of_year = day_of_year + 1 if is_leap_year and day_of_year >= 60 else day_of_year

    # Create a datetime object
    date_object = datetime.strptime(f"{year}-{day_of_year}", "%Y-%j")

    return date_object, day_of_year

#----------------------------------------------------------------------------------------------
# def process_year(est_yr):
#     # read the air temperature for this year
#     obs_airTemp_nme = '_'.join(['2m_temperature',str(est_yr),'daily_mean']) +'.nc'
#     obs_day_airTemp_file = os.path.join(path_to_era5_daily_data,obs_airTemp_nme)

#     obs_day_airTemp = xr.open_dataarray(os.path.join(path_to_autosnow_data,obs_day_airTemp_file))

#     for d in range(1,366):

#         lst_autosnow_ars = []

#         doy = year_day_to_datetime(est_yr,d)[1]

#         if len(str(doy)) == 1:
#             dofy = '00' + str(doy)

#         elif len(str(doy)) == 2:

#             dofy = '0' + str(doy)

#         else:
#             dofy = str(doy)     

#         # define year doy string which would be sued in the file name save the estimated autosnow
#         est_yr_doy = str(est_yr) + str(dofy)    

#         autsnow_estimates_svenme = '_'.join(['alldata_1992_2022_clim_subsetted_by_airTemp',est_yr_doy,autosnow_nme_part2])
#         file_checker = os.path.join(path_to_put_intermediate_files,autsnow_estimates_svenme) 

#         if os.path.isfile(file_checker):
#             continue
#         datetime_of_obs_day = year_day_to_datetime(est_yr,d)[0]

#         # get the air temperature for this day
#         obs_2m_airTemp_control = obs_day_airTemp.sel(time = datetime_of_obs_day).values

#         #------------------------------------------------- 
#         # with this list bag, we gather air temperarures differences between the day (we are estimating) 
#         # and mean month air temperatures
#         air_temp_diff_list = [] 
#         for y in range(1992,2023): #  - the climatological period

#             yr = str(y)

#             clim_yr_doy = ''.join([yr,dofy])

#             # make autosnow file name for the data the climatological period 
#             autosnow_file = '_'.join([autosnow_nme_part1,clim_yr_doy,autosnow_nme_part2])

#             autosnow_file_to_read = os.path.join(path_to_autosnow_data,autosnow_file)

#             if os.path.isfile(autosnow_file_to_read):

#                 airTemp_file_for_est = os.path.join(path_to_era5_2mairTeemp_mnth_data,
#                                        '2m_temperature_monthly_mean_' + str(y) + '_download.nc')
                
#                 datetime_of_test_day = year_day_to_datetime(y,d)[0]

#                 test_tme = pd.to_datetime(datetime_of_test_day.strftime('%Y-%m'))

#                 obs_day_airTemp_for_est = xr.open_dataarray(os.path.join(path_to_autosnow_data,airTemp_file_for_est))

#                 obs_2m_airTemp_test = obs_day_airTemp_for_est.sel(time = test_tme)

#                 # obs_day_airTemp_for_est = obs_day_airTemp_for_est.where(obs_day_airTemp_for_est < 400,np.nan)

#                 # spatially resample the monthly data to match the daily data
#                 # write crs to data
#                 obs_2m_airTemp_test.rio.write_crs(cc.to_string(), inplace=True)

#                 # resample data spatially to autosnow spatial resolution (0.1 deg for our study case)
#                 obs_day_airTemp_for_est_res = obs_2m_airTemp_test.rio.reproject(
#                 obs_2m_airTemp_test.rio.crs,
#                 shape=obs_2m_airTemp_control.shape, # set the shape as the autosnow data shape
#                 resampling=Resampling.bilinear,)                       

#                 # find the difference between the 2m air temp on the obs day and the test day
#                 air_temp_diff = np.abs(obs_2m_airTemp_control - obs_day_airTemp_for_est_res)

#                 air_temp_diff_list.append(air_temp_diff.values)

#                 #----------------------------------------
#                 # read the autosnow file
#                 dat_auto = xr.open_dataarray(autosnow_file_to_read)        

#                 dat_auto = dat_auto.data[0,:,:]

#                 dat_auto = np.where(dat_auto > 3, np.nan,dat_auto)

#                 lst_autosnow_ars.append(dat_auto) # collect autosnow for within the climatological period
#         #----------------------------------------
#         # stack up the air temp differences in 3d
#         airtemp_diff_3d = np.dstack(air_temp_diff_list) 
        
#         # rank the temp differences from least to largest
#         out_arr_rnk = stats.mstats.rankdata(airtemp_diff_3d, axis=2, use_missing=False)

#         # stack up the the climatologoical autosnow data in 3d
#         autosnow_3d = np.dstack(lst_autosnow_ars)

#         # below, we select 11 samples out of the (max 20 samples) data, collected between the 1992-2020 period,
#         # which have air temperature close to the observation day (i.e, the day we are trying to estimate)
#         autosnow_3d_sampled_by_rank = np.where(out_arr_rnk > 11,np.nan,autosnow_3d)

#         # then we estimate the autosnow value based on the mode of these 11 samples
#         estimated_autosnow_arr = np.apply_along_axis(lambda a: stats.mode(a,nan_policy='omit')[0][0], 2, 
#                                                      autosnow_3d_sampled_by_rank)
#         #----------------------------------------

#         # save the estimated autosnow and std for use with machine learning techn
#         autsnow_estimates_svenme = '_'.join(['alldata_1992_2022_clim_subsetted_by_airTemp',est_yr_doy,autosnow_nme_part2])
#         my_map_functions.rasterio_based_save_array_to_disk(path_to_put_intermediate_files, autsnow_estimates_svenme,
#                                                         autosnow_meta_cpy, estimated_autosnow_arr)  

#         count = count + 1

#         if count % 100 == 0:
#             print(str(count) + ' Autosnow files processed so far') 
#%%
print('************ begin estimation ********************')

# if __name__ == '__main__':
#     # autosnow_nme_part1 = 'gmasi_snowice_reproc_v003'
#     # autosnow_nme_part2 = '0.1deg_wgs.tif'

#     # path_to_autosnow_data = '/path/to/autosnow/data'
#     # path_to_era5_daily_data = '/path/to/era5/daily/data'
#     # path_to_era5_2mairTeemp_mnth_data = '/path/to/era5/2m/airtemp/monthly/data'
#     # path_to_put_intermediate_files = '/path/to/put/intermediate/files'

#     # Assuming autosnow_meta_cpy is defined somewhere
#     autosnow_meta_cpy = rasterio.open(file_sample).meta.copy()

#     count = 0
#     est_years_range = range(1991, 1992)  # Adjust the range as needed

#     with ProcessPoolExecutor() as executor:
#         executor.map(process_year, est_years_range)

all_autosn_obs_arr_in_yr = []

all_autosn_est_arr_in_yr = []

count = 0

for est_yr in range(1988,1992):   

    # read the air temperature for this year
    obs_airTemp_nme = '_'.join(['2m_temperature',str(est_yr),'daily_mean']) +'.nc'
    obs_day_airTemp_file = os.path.join(path_to_era5_daily_data,obs_airTemp_nme)

    obs_day_airTemp = xr.open_dataarray(os.path.join(path_to_autosnow_data,obs_day_airTemp_file))

    for d in range(1,366):

        lst_autosnow_ars = []

        doy = year_day_to_datetime(est_yr,d)[1]

        if len(str(doy)) == 1:
            dofy = '00' + str(doy)

        elif len(str(doy)) == 2:

            dofy = '0' + str(doy)

        else:
            dofy = str(doy)     

        # define year doy string which would be used in the file name save the estimated autosnow
        est_yr_doy = str(est_yr) + str(dofy)     

        # skip already existing files
        autsnow_estimates_svenme = '_'.join(['alldata_1992_2022_clim_subsetted_by_airTemp',est_yr_doy,autosnow_nme_part2])

        if os.path.isfile(os.path.join(path_to_put_intermediate_files, autsnow_estimates_svenme)):
            continue
        
        datetime_of_obs_day = year_day_to_datetime(est_yr,d)[0]

        # get the air temperature for this day
        obs_2m_airTemp_control = obs_day_airTemp.sel(time = datetime_of_obs_day).values

        #------------------------------------------------- 
        # with this list bag, we gather air temperarures differences between the day (we are estimating) 
        # and mean month air temperatures
        air_temp_diff_list = [] 
        for y in range(1992,2023): #  - the climatological period

            yr = str(y)

            clim_yr_doy = ''.join([yr,dofy])

            # make autosnow file name for the data the climatological period 
            autosnow_file = '_'.join([autosnow_nme_part1,clim_yr_doy,autosnow_nme_part2])

            autosnow_file_to_read = os.path.join(path_to_autosnow_data,autosnow_file)

            if os.path.isfile(autosnow_file_to_read):

                airTemp_file_for_est = os.path.join(path_to_era5_2mairTeemp_mnth_data,
                                       '2m_temperature_monthly_mean_' + str(y) + '_download.nc')
                
                datetime_of_test_day = year_day_to_datetime(y,d)[0]

                test_tme = pd.to_datetime(datetime_of_test_day.strftime('%Y-%m'))

                obs_day_airTemp_for_est = xr.open_dataarray(os.path.join(path_to_autosnow_data,airTemp_file_for_est))

                obs_2m_airTemp_test = obs_day_airTemp_for_est.sel(time = test_tme)

                # obs_day_airTemp_for_est = obs_day_airTemp_for_est.where(obs_day_airTemp_for_est < 400,np.nan)

                # spatially resample the monthly data to match the daily data
                # write crs to data
                obs_2m_airTemp_test.rio.write_crs(cc.to_string(), inplace=True)

                # resample data spatially to autosnow spatial resolution (0.1 deg for our study case)
                obs_day_airTemp_for_est_res = obs_2m_airTemp_test.rio.reproject(
                obs_2m_airTemp_test.rio.crs,
                shape=obs_2m_airTemp_control.shape, # set the shape as the autosnow data shape
                resampling=Resampling.bilinear,)                       

                # find the difference between the 2m air temp on the obs day and the test day
                air_temp_diff = np.abs(obs_2m_airTemp_control - obs_day_airTemp_for_est_res)

                air_temp_diff_list.append(air_temp_diff.values)

                #----------------------------------------
                # read the autosnow file
                dat_auto = xr.open_dataarray(autosnow_file_to_read)        

                dat_auto = dat_auto.data[0,:,:]

                dat_auto = np.where(dat_auto > 3, np.nan,dat_auto)

                lst_autosnow_ars.append(dat_auto) # collect autosnow for within the climatological period
        #----------------------------------------
        # stack up the air temp differences in 3d
        airtemp_diff_3d = np.dstack(air_temp_diff_list) 
        
        # rank the temp differences from least to largest
        out_arr_rnk = stats.mstats.rankdata(airtemp_diff_3d, axis=2, use_missing=False)

        # stack up the the climatologoical autosnow data in 3d
        autosnow_3d = np.dstack(lst_autosnow_ars)

        # below, we select 11 samples out of the (max 20 samples) data, collected between the 1992-2020 period,
        # which have air temperature close to the observation day (i.e, the day we are trying to estimate)
        autosnow_3d_sampled_by_rank = np.where(out_arr_rnk > 11,np.nan,autosnow_3d)

        # then we estimate the autosnow value based on the mode of these 11 samples
        estimated_autosnow_arr = np.apply_along_axis(lambda a: stats.mode(a,nan_policy='omit')[0][0], 2, 
                                                     autosnow_3d_sampled_by_rank)
        #----------------------------------------

        # save the estimated autosnow and std for use with machine learning techn
        
        my_map_functions.rasterio_based_save_array_to_disk(path_to_put_intermediate_files, autsnow_estimates_svenme,
                                                        autosnow_meta_cpy, estimated_autosnow_arr)  

        count = count + 1

        if count % 100 == 0:
            print(str(count) + ' Autosnow files processed so far') 

print('******************* done with estimation ****************************')
print('*********************************************************************')
print('*********************************************************************')
print('*********************************************************************')


#%%
read_try = rasterio.open(os.path.join(path_to_put_intermediate_files,'alldata_1992_2022_clim_subsetted_by_airTemp_1991001_0.1deg_wgs.tif'))

read_try_arr = read_try.read(1)