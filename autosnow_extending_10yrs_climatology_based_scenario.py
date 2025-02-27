'''
Author: Dr K. K. Kumah

estimating autosnow data by using past 10 years climatology. basically, we are classifying a pixel as
water, snow free, snow covered or ice based on what that pixel has mostly been in the past 10 years
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

#%%
# define paths
path_to_autosnow_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_in_geotif'
path_to_save_df = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/save_dfs_Oct'

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

# make list of row, col indexes
lst_row_col_idx = []

for row in range(0,y_shp):

    for col in range(0,x_shp):

        id_tuple = tuple((row,col))

        lst_row_col_idx.append(id_tuple)

classes = ['Water','Snow free land', 'Snow covered land','Ice']
"""
0: water
1: snow-free land
2: snow-covered land
3: ice
"""

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

    obs_yr_doy = ''.join(['1988',dofy])    

    obs_autosnow_file = '_'.join([autosnow_nme_part1,obs_yr_doy,autosnow_nme_part2])

    obs_autosn = xr.open_dataarray(os.path.join(path_to_autosnow_data,obs_autosnow_file))       

    obs_autosn = obs_autosn.data[0,:,:]

    obs_autosn = np.where(obs_autosn > 3, np.nan,obs_autosn)

    all_autosn_obs_arr_in_yr.append(obs_autosn)

    for y in range(1989,1998):

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
    #my_map_functions.calculate_array(lst_row_col_idx,autosnow_3d,estimated_autosnow_arr)

    all_autosn_est_arr_in_yr.append(estimated_autosnow_arr)

    count = count + 1

    if count % 50 == 0:
        print(str(count) + ' Autosnow files processed so far') 

print('******************* done with estimation ****************************')
print('*********************************************************************')
print('*********************************************************************')
print('*********************************************************************')

#%%
print('******************* begin evaluation ******************* ')
# compute metrics
autosn_est_1d = my_map_functions.make_3d_array(x_shp,y_shp,all_autosn_est_arr_in_yr)
autosn_est_1d = autosn_est_1d.reshape((y_shp*x_shp*autosn_est_1d.shape[2]))

autosn_obs_1d = my_map_functions.make_3d_array(x_shp,y_shp,all_autosn_obs_arr_in_yr)
autosn_obs_1d = autosn_obs_1d.reshape((y_shp*x_shp*autosn_obs_1d.shape[2]))

df_obs_est = pd.DataFrame(np.column_stack((autosn_obs_1d,autosn_est_1d)),columns=['obs','est'])

n_per_class_in_y = pd.DataFrame(df_obs_est['obs'].value_counts())

n_per_class_in_y['Class']  = classes


lst_cat_stats = []

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

dfs_cat_stats.to_csv(os.path.join(path_to_save_df,'categorical_stats_10yrs_climatologiy_scenario_results_2nd_Oct.csv'))

print('*********************************************************************')
print('*********************************************************************')
print('*********************************************************************')
print('done!')

#%%
    
    

