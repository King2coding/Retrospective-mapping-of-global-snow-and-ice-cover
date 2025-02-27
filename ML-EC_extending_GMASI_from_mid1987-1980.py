'''
Author: Dr K. K. Kumah

Date = Monday, 2024-12-01
This code is what is used now
***************************************************************************************
This code estimates/predict surface type using two method:
    ML-E :- in which we trained random forest using ERA5-based surface variabl;es as predictors/features
    and GMASI-Autosnow as target

    ML-EC :- similar toi ERA5 except we complement the features with Climatological information
    derived from the GMASI-Autosnow

'''
#%%
# import apckages
import warnings
warnings.filterwarnings('ignore')
import datetime
from datetime import datetime, date, timedelta
import os
import pickle
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt

from util_functions import *

import xarray as xr

#%%
#  define path to data

path_to_era5_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/era5_daily_data_for_extending_autosnow'

path_to_models = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/trained_models'

path_to_static_vars = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/extending_autosnow_estimated/static_predictors_for_ML-EC_based_on_1992_2022_data'
# r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/alldata_climatology_based_autosnow_estimate'

path_to_put_extended_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_extended_mid1987_1980'
#%%
# define global variables

# Start and end dates for the extension period
start_date = datetime(1980, 1, 1) # start date of intended reconstruction period
end_date = datetime(1987, 6, 30)  # Stop at June 30, 1987 of intended reconstruction period

# Generate a list of dates between start_date and end_date
reconstructed_date_list = [(start_date + timedelta(days=x)) for x in range((end_date - start_date).days + 1)]


lake_cover = r'/ra1/pubdat/AVHRR_CloudSat_proj/global_variables/lake_cover_2007_0.1deg.nc'

land_sea_mask = r'/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/ancillary_imerg_data/GPM_IMERG_LandSeaMask.2.nc4'
#------------------------------------------------
lc_arr = xr.open_dataset(lake_cover).cl.data[0]

lsm_arr = xr.open_dataset(land_sea_mask).landseamask.data
# transpose the data to get longitude on the x axis, and flip vertically
# so that latitude is displayed south to north as it should be
lsm_arr = np.flip(lsm_arr.transpose(), axis=0)

lsm_arr_ = np.where(lsm_arr < 25, 1, 0).copy()

#------------------------------------------------

lon = np.arange(-180, 180, 0.1)
lat = np.arange(90, -90, -0.1)
mesh_xy = np.meshgrid(lon, lat)

antartica_msk = (mesh_xy[1] <= -66.5) & (lsm_arr_ == 1) # antartica lands mask
# 

# Tropical and equatorial regions: 25°S to 25°N, assumed snow free
tropical_mask = (mesh_xy[1] >= -25) & (mesh_xy[1] <= 25) & (lsm_arr_ == 1)

#------------------------------------------------

# read the model

ml_ec_rf_mdl_fle = os.path.join(path_to_models,'RF_classifier_model_trained_with_ERA5_and_autosnowclim_data_20231201.pkl')
with open(ml_ec_rf_mdl_fle, 'rb') as rf_mdl_file:  
    ml_ec_rf_model = pickle.load(rf_mdl_file)


comn_nme_prt = '0.1deg_wgs'

data_info_summary = (
    "This dataset was generated using a Random Forest machine learning algorithm, "
    "trained on ERA5 surface variables and static climatological features derived "
    "from GMASI-Autosnow data during the 1992–2022 period. "
    "It provides an extension of the GMASI-Autosnow product to the pre-inception period "
    "from July 1987 back to 1980. The data is available at a daily temporal resolution "
    "and a spatial resolution of 0.1 degrees on a lat/lon WGS-84 grid."
)

cde_run_dte = str(date.today().strftime('%Y%m%d'))

gc.collect()

def correction_data(input_array):
    from scipy.ndimage import generic_filter

    # # Step 2: Correct surface data based on proximity and land-sea mask
    # # Pixels close to the coast or on land may need special handling to correct for land spillover or misclassification
    corrected_surface_data1 = generic_filter(input_array, correct_pixel, size=(3, 3), mode='nearest')

    #-------------------------- regional specfic corrections -------------------------------
    corrected_surface_data2 = np.where(antartica_msk,2,corrected_surface_data1)

    corrected_surface_data3 = np.where(tropical_mask,1,corrected_surface_data2)

    del(corrected_surface_data1, corrected_surface_data2)

    return corrected_surface_data3

#%%
print('begin the estimating autosnow using trained RF machine learning algorithm!')
print('****************************')
print('****************************')
print('****************************')

# Function to process a single autosnow file
def process_autosnow_file(dte):
    try:
        # Extract year and DOY
        yr_DOY = f"{dte.year}{dte.timetuple().tm_yday:03d}"
        doy = f"{dte.timetuple().tm_yday:03d}"
        yr = int(dte.year)
        doY = int(dte.timetuple().tm_yday)
        dt = dte #datetime(yr, 1, 1) + timedelta(doY - 1)       

        # Read ERA5 input files
        dwpnt_2m_arr = read_era5_file(path_to_era5_data, '2m_dewpoint_temperature', str(yr), dt)
        temp_2m_arr = read_era5_file(path_to_era5_data, '2m_temperature', str(yr), dt)
        sea_ice_arr = read_era5_file(path_to_era5_data, 'sea_ice_cover', str(yr), dt)
        sst_arr = read_era5_file(path_to_era5_data, 'sea_surface_temperature', str(yr), dt)
        skt_arr = read_era5_file(path_to_era5_data, 'skin_temperature', str(yr), dt)
        albedo_arr = read_era5_file(path_to_era5_data, 'forecast_albedo', str(yr), dt)

        # Read climatology-based data
        clim_autsnw_data_array = read_climatology_file(path_to_static_vars, 
                                                       ['predominant_surface_type_based_on_1992_2022_DOY', 
                                                       doy, comn_nme_prt], '.nc')
        wtr_cls_prob_data_array = read_climatology_file(path_to_static_vars, 
                                                        ['probability_of_grid_being_water_based_on_1992_2022_DOY', 
                                                         doy, comn_nme_prt], '.nc')
        snw_free_lnd_cls_prob_array = read_climatology_file(path_to_static_vars, 
                                                            ['probability_of_grid_being_snowfree_based_on_1992_2022_DOY', 
                                                             doy, comn_nme_prt], '.nc')
        snw_cvrd_lnd_cls_prob_data_array = read_climatology_file(path_to_static_vars, 
                                                                 ['probability_of_grid_being_snowcovered_based_on_1992_2022_DOY', 
                                                                  doy, comn_nme_prt], '.nc')
        ice_cls_prob_data_array = read_climatology_file(path_to_static_vars, 
                                                        ['probability_of_grid_being_ice_cvr_based_on_1992_2022_DOY', 
                                                         doy , comn_nme_prt], '.nc')

        # Prepare auxiliary data
        lc_arr[np.isnan(lc_arr)] = -99999
        lsm_arr[np.isnan(lsm_arr)] = -99999
        lat_2d_arr = mesh_xy[1]
        lon_2d_arr = mesh_xy[0]
        doy_arr = np.full_like(dwpnt_2m_arr, doY)

        # Prepare input lists
        ml_ec_arr_lst = [lon_2d_arr, lat_2d_arr, doy_arr, dwpnt_2m_arr, temp_2m_arr, sst_arr, skt_arr, sea_ice_arr, albedo_arr, clim_autsnw_data_array,
                         wtr_cls_prob_data_array, snw_free_lnd_cls_prob_array, snw_cvrd_lnd_cls_prob_data_array, ice_cls_prob_data_array, lc_arr, lsm_arr]

        # estimate using existing trained RF model
        y_shp, x_shp = clim_autsnw_data_array.shape
        ml_ec_arr_rshp = prepare_rf_input(ml_ec_arr_lst, x_shp, y_shp)
        ml_ec_rf_predicted = ml_ec_rf_model.predict(ml_ec_arr_rshp)
        ml_ec_rf_predicted_arr = ml_ec_rf_predicted.reshape(y_shp, x_shp)
        ml_ec_rf_predicted_arr_ = correction_data(ml_ec_rf_predicted_arr)
        ml_ec_name = '_'.join(['UofA_gmasi_snowice_extended_v001', yr_DOY, '0.1deg']) + '.nc'
        ml_ec_file = os.path.join(path_to_put_extended_data, ml_ec_name)
        spit_nc_file(ml_ec_rf_predicted_arr_, ml_ec_file, lon, lat, 'Snow-Ice', data_info_summary)

        return f"Processed file {yr_DOY}"

    except Exception as e:
        return f"Error processing file {yr_DOY}: {e}"


# Main parallel processing function

def parallel_process_files(file_paths, max_workers=15):
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
        future_to_file = {executor.submit(process_autosnow_file, file_path): file_path for file_path in file_paths}
        
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)  # Collect results for aggregation
            except Exception as exc:
                print(f'{file_path} generated an exception: {exc}')
    
    return results

if __name__ == "__main__":
    print("Starting parallel processing...")

    # Use ProcessPoolExecutor for CPU-bound tasks
    files_to_process = sorted(reconstructed_date_list)  # Adjust the range for testing
    max_workers = 15#os.cpu_count()  # Use all available CPU cores

    results = parallel_process_files(files_to_process, max_workers=max_workers)

    # Log results
    for result in results:
        print(result)

    print("Parallel processing complete!")
#%%
# # The truth test

# ml_ec = xr.open_dataarray(ml_ec_file)
# ml_ec.plot()
