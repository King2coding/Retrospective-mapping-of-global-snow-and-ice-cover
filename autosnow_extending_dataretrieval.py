'''
Code constructs big data frame containing autosnow and era5-based variables for ML applications in autonsow estimation
'''
#%%
# import packages
import os
import pandas as pd
import numpy as np
import datetime
from datetime import date
import gc

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import map_manipulation_functions as mmf
import seaborn as sns

import xarray as xr

import cartopy.crs as ccrs
import cartopy as cart
import cartopy.feature as cfeature
# import ml_algs as myML

#%%
#  define path to data
path_to_autosnow_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_in_geotif'

path_to_era5_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/era5_daily_data_for_extending_autosnow'

path_to_put_plots = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/plots'

path_to_put_df = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/era5_training_df_that_estimated_autosnow'

path_to_autosno_climatological_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/alldata_climatology_based_autosnow_estimate'
#%%
# declare flobal variables
yrs_str = ['1992','1993','1994','1995']
# all_autosnow_files = [os.path.join(path_to_autosnow_data,a) for a in os.listdir(path_to_autosnow_data) if \
#                       (int(a.split('_')[4][:4]) >= 2007 and int(a.split('_')[4][:4]) <= 2010)]

all_autosnow_files = [os.path.join(path_to_autosnow_data,a) for a in os.listdir(path_to_autosnow_data) if (int(a.split('_')[4][:4]) >=1992) and (int(a.split('_')[4][:4]) <=1995)]

all_era5_data = [os.path.join(path_to_era5_data,e) for e in os.listdir(path_to_era5_data) if (any(x in e for x in yrs_str))]
#---------------------------------------------------------

# lake cover and land sea mask are temporally static variables; it doesn't matter from which time epriod they were used
lake_cover = r'/ra1/pubdat/AVHRR_CloudSat_proj/global_variables/lake_cover_2007_0.1deg.nc'

land_sea_mask = r'/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/ancillary_imerg_data/GPM_IMERG_LandSeaMask.2.nc4'
#r'/ra1/pubdat/AVHRR_CloudSat_proj/global_variables/land_sea_mask_2007_0.1deg.nc'

lc_arr = xr.open_dataset(lake_cover).cl.data[0]

lsm_arr = xr.open_dataset(land_sea_mask).landseamask.data
# transpose the data to get longitude on the x axis, and flip vertically
# so that latitude is displayed south to north as it should be
lsm_arr = np.flip(lsm_arr.transpose(), axis=0)
#---------------------------------------------------------

era5_collectables = {}

for er in all_era5_data:

    ear_var_name = os.path.basename(er).split('.')[0]

    era_data = xr.open_dataset(er)

    era5_collectables[ear_var_name] = era_data
#---------------------------------------------------------

mesh_xy = np.meshgrid(np.arange(-180, 180, 0.1), np.arange(90, -90, -0.1))

img_extent = (-180, 180, -90, 90)

projc = ccrs.PlateCarree()#ccrs.epsg(4326)

cde_run_dte = str(date.today().strftime('%Y%m%d'))

comn_nme_prt1 = 'based_on_alldata_clim_1999_2022_DOY'
comn_nme_prt2 = '0.1deg_wgs'
comn_nme_prt3 = 'prob_based_on_all_data_clim_1999_2022_DOY'
# Define sampling fractions for each autosnow class
# 0 = water, 1 = snow free land, 2 = snow covered land, 3 = ice
sampling_fraction = {0 : 0.2, 1 : 0.2, 2 : 0.3, 3 : 0.3}

gc.collect()

#%%
# begin retrieving and constructing big data frame from arrays
print('************ begin retrieval ********************')
count = 0
all_dfs = []
for a in sorted(all_autosnow_files):  

    dat_auto = xr.open_dataarray(a)

    dat_auto_array = dat_auto.data[0,:,:]

    # there some autosnow pixels with values = 200; therefore set all pixels with values > 3 to nan
    dat_auto_array = np.where(dat_auto_array > 3,np.nan,dat_auto_array)

    y_shp,x_shp = dat_auto_array.shape[0],dat_auto_array.shape[1] 

    # get the date of the autosnow file which is in year and day of year
    yr_DOY = os.path.basename(a).split('_')[4]

    yr = int(yr_DOY[:4])

    dOY = int(yr_DOY[-3:])

    doy = yr_DOY[-3:]   
    #------------------------------------------------------

    dt = datetime.datetime(yr, 1, 1) + datetime.timedelta(dOY - 1) 

    # based on this date, reconstruct keys for getting the data from the surface variables from era5
    # construct file names for reading the autsnow archived data
    dwpnt2m_key = ''.join(['2m_dewpoint_temperature_', str(yr), '_daily_mean'])
    temp2m_key = ''.join(['2m_temperature_', str(yr), '_daily_mean'])

    seaice_key = ''.join(['sea_ice_cover_', str(yr), '_daily_mean'])
    sst_key = ''.join(['sea_surface_temperature_', str(yr), '_daily_mean'])
    skt_key = ''.join(['skin_temperature_', str(yr), '_daily_mean'])
   
    albedo_mean_key = ''.join(['forecast_albedo_', str(yr),'_daily_mean'])
    #------------------------------------------------------
    clim_autsnw_fle = '_'.join(['autosnow',comn_nme_prt1,doy,comn_nme_prt2]) + '.tif'

    clim_autsnw_fle = os.path.join(path_to_autosno_climatological_data,clim_autsnw_fle)

    wtr_cls_prob_fle = '_'.join(['water_class',comn_nme_prt3,doy,comn_nme_prt2]) + '.tif'
    wtr_cls_prob_fle = os.path.join(path_to_autosno_climatological_data,wtr_cls_prob_fle)

    snw_free_lnd_cls_prob_fle = '_'.join(['snow_free_land_class',
                                          comn_nme_prt3,doy,comn_nme_prt2]) + '.tif'
    snw_free_lnd_cls_prob_fle = os.path.join(path_to_autosno_climatological_data,snw_free_lnd_cls_prob_fle)

    snw_cvrd_lnd_cls_prob_fle = '_'.join(['snow_covered_land_class',
                                          comn_nme_prt3,doy,comn_nme_prt2]) + '.tif'
    snw_cvrd_lnd_cls_prob_fle = os.path.join(path_to_autosno_climatological_data,snw_cvrd_lnd_cls_prob_fle)

    ice_cls_prob_fle = '_'.join(['ice_class',comn_nme_prt3,doy,comn_nme_prt2]) + '.tif'
    ice_cls_prob_fle = os.path.join(path_to_autosno_climatological_data,ice_cls_prob_fle)
    #------------------------------------------------------

    dewpt_2m_arr = era5_collectables[dwpnt2m_key].sel(time = dt).d2m.values

    temp_2m_arr = era5_collectables[temp2m_key].sel(time = dt).t2m.values

    sea_ice_arr = era5_collectables[seaice_key].sel(time=dt).siconc.values
    
    sst_arr = era5_collectables[sst_key].sel(time=dt).sst.values

    skt_arr = era5_collectables[skt_key].sel(time=dt).skt.values  

    albedo_arr_mean = era5_collectables[albedo_mean_key].sel(time=dt).fal.values
    
    #------------------------------------------------------
    # the climatology based autonow, water, snow free land, snow covered land and ice class probability data
    clim_autsnw_data_array = xr.open_dataarray(clim_autsnw_fle)
    clim_autsnw_data_array = clim_autsnw_data_array.data[0,:,:]

    wtr_cls_prob_data_array = xr.open_dataarray(wtr_cls_prob_fle).data
    wtr_cls_prob_data_array = wtr_cls_prob_data_array[0,:,:]

    snw_free_lnd_prob_data_array = xr.open_dataarray(snw_free_lnd_cls_prob_fle).data
    snw_free_lnd_prob_data_array = snw_free_lnd_prob_data_array[0,:,:]

    snw_cvrd_lnd_cls_prob_data_array = xr.open_dataarray(snw_cvrd_lnd_cls_prob_fle).data
    snw_cvrd_lnd_cls_prob_data_array = snw_cvrd_lnd_cls_prob_data_array[0,:,:]

    ice_cls_prob_data_array = xr.open_dataarray(ice_cls_prob_fle).data
    ice_cls_prob_data_array = ice_cls_prob_data_array[0,:,:]
    #------------------------------------------------------
    # create some ancillary data
    lat_arr = mesh_xy[1]
    lon_arr = mesh_xy[0] 

    yr_arr = np.empty_like(dewpt_2m_arr)
    yr_arr[:,:] = yr

    doy_arr = np.empty_like(dewpt_2m_arr)
    doy_arr[:,:] = dOY

    #----------------------------------------

    # convert all the arrays to a 1d array

    autsnw_1d = dat_auto_array.flatten() # ie.e the target/label variable

    # all others are features
    dewpnt2m_1d = dewpt_2m_arr.flatten()

    temp2m_1d = temp_2m_arr.flatten()

    sea_ice_1d = sea_ice_arr.flatten()

    sst_1d = sst_arr.flatten()

    skt_1d = skt_arr.flatten()

    albedo_1d = albedo_arr_mean.flatten()

    # the climatology data
    clim_autsnw_1d = clim_autsnw_data_array.flatten()

    wtr_cls_prob_1d = wtr_cls_prob_data_array.flatten()

    snw_free_lnd_prob_1d = snw_free_lnd_prob_data_array.flatten()

    snw_cvrd_lnd_cls_prob_1d = snw_cvrd_lnd_cls_prob_data_array.flatten()

    ice_cls_prob_1d = ice_cls_prob_data_array.flatten()

    # some ancillary data
    lat_1d = lat_arr.flatten()
    lon_1d = lon_arr.flatten()

    lc_1d = lc_arr.flatten()
    lsm_1d = lsm_arr.flatten()

    yr_1d = yr_arr.flatten()

    doy_1d = doy_arr.flatten()

    gc.collect()     
    #------------------------------------------------------#------------------------------------------------------

    # make big data frame
    # albedo_min_arrs_concat, albedo_max_arrs_concat , 
    big_df = pd.DataFrame(list(zip(lon_1d,lat_1d, yr_1d, doy_1d, 
                                   
                                   dewpnt2m_1d, temp2m_1d, sea_ice_1d, sst_1d, skt_1d, albedo_1d, 
                                   
                                   clim_autsnw_1d, wtr_cls_prob_1d, snw_free_lnd_prob_1d, snw_cvrd_lnd_cls_prob_1d, 
                                   
                                   ice_cls_prob_1d, lsm_1d, lc_1d,                                  

                                   autsnw_1d)), 
                                                                      
                                   columns=['longitude','latitude','year','DOY',
                                            
                                            '2m_dewpt','2m_temp','sea_ice','sst',

                                            'skin_temp', 'albedo_mean' , 'clim_auto',

                                            'wtr_cls_prob', 'snw_fre_lnd_cls_prob', 'snw_cvrd_lnd_cls_prob', 

                                            'ice_cls_prob', 'land_sea_mask', 'lake_cover', 'autosnow'])  #'albedo_grad', 
    
    
    #------------------------------------------------------#------------------------------------------------------

    
    small_df = big_df.groupby(['autosnow']).sample(frac=0.0004,replace=True,random_state=42) # ,weights=big_df['weights']

    all_dfs.append(small_df)

    gc.collect()    

    count += 1
    
    if count % 100 == 0:
        print(str(count) + ' files is completed')

df_all = pd.concat(all_dfs,axis=0)

svenme = '_'.join(['big_df_clim_autosn_included_code_run_on',cde_run_dte]) + '.pkl'

df_all.to_pickle(os.path.join(path_to_put_df,svenme))

print('done!')
gc.collect()    
#%%
#small_df.to_pickle(os.path.join(path_to_put_df,'big_df_small_df_1day.pkl'))


# #%%
# cx = xr.open_dataset(r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/era5_nc_files/forecast_albedo_2007_download.nc')
# # do some plot to see how the data looks
# # plot autosnow
# #f = plt.figure()
# # Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
# col_dict={0:"deepskyblue",
#           1:"darkorange",
#           2:"hotpink",
#           3:"seagreen"}

# # We create a colormar from our list of colors
# cm = ListedColormap([col_dict[x] for x in col_dict.keys()])

# labels = np.array(['Water','Snow free \n land','Snow covered \n land','Ice'])
# len_lab = len(labels)

# # prepare normalizer
# ## Prepare bins for the normalizer
# norm_bins = np.sort([*col_dict.keys()]) + 0.5
# norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1) 

# ## Make normalizer and formatter
# norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
# fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

# diff = norm_bins[1:] - norm_bins[:-1]
# tickz = norm_bins[:-1] + diff / 2

# cmp = plt.get_cmap('PiYG', 4)


# ax = plt.axes(projection=projc)

# ax_plt = ax.imshow(dat_auto_array,cmap=cm,extent=img_extent,origin="upper",transform=projc)

# ax.coastlines()

# ax.text(-0.15, 0.55, 'Latitude', va='bottom', ha='center',
#         rotation='vertical', rotation_mode='anchor',fontsize=15,
#         transform=ax.transAxes)
# ax.text(0.5, -0.2, 'Longitude', va='bottom', ha='center',
#         rotation='horizontal', rotation_mode='anchor',fontsize=15,
#         transform=ax.transAxes)

# gls_ax = ax.gridlines(crs=projc,color='grey', linestyle='--', lw = 0.35, draw_labels={"bottom": "x", "left": "y"})
# # add these before plotting
# gls_ax.xlabel_style={'size':15}   
# gls_ax.ylabel_style={'size':15} # ,'rotation':45

# cbar = plt.colorbar(ax_plt,location='bottom',pad=0.15,ticks=tickz, format=fmt)

# tick_locs = (np.arange(len_lab) + 0.5)*(len_lab-1)/len_lab
# cbar.set_ticks(tick_locs)

# plt.savefig(os.path.join(path_to_put_plots,'autosnow_plt.png'))
# plt.close()

# #%%
# mmf.plot_arry_grid(dewpt_2m_arr,img_extent)
# plt.savefig(os.path.join(path_to_put_plots,'dewpnt2m_plt.png'))
# plt.close()

# mmf.plot_arry_grid(temp_2m_arr,img_extent)
# plt.savefig(os.path.join(path_to_put_plots,'temp2m_plt.png'))
# plt.close()

# mmf.plot_arry_grid(sea_ice_arr,img_extent)
# plt.savefig(os.path.join(path_to_put_plots,'seaIce_plt.png'))
# plt.close()

# mmf.plot_arry_grid(sst_arr,img_extent)
# plt.savefig(os.path.join(path_to_put_plots,'sst_plt.png'))
# plt.close()

# mmf.plot_arry_grid(skt_arr,img_extent)
# plt.savefig(os.path.join(path_to_put_plots,'skt_plt.png'))
# plt.close()


# #%%
# # some stats
# df_all = pd.concat(all_dfs,axis=0)

#-------------------------------------------------------------------
# define fucntions
# Define a function to apply sampling to each group
# def sample_group(group):
#     class_value = group['autosnow'].iloc[0]
#     fraction = sampling_fractions.get(class_value, 1.0)  # Default to 1.0 if class not found
#     return group.sample(frac=fraction,replace=True,random_state=42)

# Define a function to apply sampling to the entire DataFrame
'''def sample_dataframe_with_classes(df, class_col, sampling_fractions, total_frac, random_state=None):
    """
    Sample a DataFrame with different fractions for each class in a specific column.

    Parameters:
    - df: DataFrame to be sampled.
    - class_col: Column containing class values.
    - sampling_fractions: Dictionary with class values as keys and corresponding sampling fractions as values.
    - total_frac: Fraction of the entire DataFrame to be sampled randomly.
    - random_state: Random state for reproducibility.

    Returns:
    - Sampled DataFrame.
    """
    random_state = 42
    # Sample total_frac of the entire DataFrame randomly with replacement
    sampled_df = df.sample(frac=total_frac, replace=False, random_state=random_state)

    # List to store sampled subsets for each class
    sampled_subsets = []

    # Apply different sampling fractions for each class
    for class_value, fraction in sampling_fractions.items():
        class_subset = sampled_df[sampled_df[class_col] == class_value]
        sampled_class_subset = class_subset.sample(frac=fraction, replace=False, random_state=random_state)
        
        # Replace the corresponding rows in the random sample with the class-specific sample
        sampled_df = sampled_df.loc[~sampled_df.index.isin(class_subset.index)]
        sampled_df = pd.concat([sampled_df, sampled_class_subset])

    return sampled_df


def subsample_dataframe(df, fractions):
    # Calculate the expected size of the sampled DataFrame
    expected_size = int(0.3 * len(df))

    # Adjust the sampling fractions to achieve the desired size
    total_rows = len(df)
    adjusted_fractions = {group: (fraction * expected_size) / total_rows for group, fraction in fractions.items()}

    # Define the function for sampling within each group
    def sample_group(group):
        fraction = adjusted_fractions[group.name]
        return group.sample(frac=fraction)

    # Apply the sampling function to each group
    sampled_df = df.groupby('autosnow', group_keys=False).apply(sample_group)

    return sampled_df'''