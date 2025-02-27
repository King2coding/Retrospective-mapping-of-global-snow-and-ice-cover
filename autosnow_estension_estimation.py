'''
ML-based autosnow estimation
Code relies on already trained model
'''
#%%
# import apckages
import warnings
warnings.filterwarnings('ignore')
import datetime
from datetime import date
import os
import pickle
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import map_manipulation_functions as map_fnc
import map_manipulation_functions as mmf
import evaluation_fucntions_algorithms as my_eval_fucnts

import rasterio
from rasterio import transform
import xarray as xr

#%%
#  define path to data
path_to_autosnow_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_in_geotif'
#r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_geotif_2007'

path_to_era5_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/era5_daily_data_for_extending_autosnow'
#r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/daily_era5_nc_files'

path_to_put_plots = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/save_plots_Dec'

path_to_put_df = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/save_dfs_Dec'

path_to_models = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/trained_models'

path_to_intermediates = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/intermediates'

path_to_autosno_climatological_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/alldata_climatology_based_autosnow_estimate'

path_to_put_estimated_autosnw = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_estimated_1988_1991'
#%%
# define global variables

all_autosnow_files = [os.path.join(path_to_autosnow_data,a) for a in os.listdir(path_to_autosnow_data) if (int(a.split('_')[4][:4]) >=1988) and (int(a.split('_')[4][:4]) <=1991)]

lake_cover = r'/ra1/pubdat/AVHRR_CloudSat_proj/global_variables/lake_cover_2007_0.1deg.nc'

land_sea_mask = r'/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/ancillary_imerg_data/GPM_IMERG_LandSeaMask.2.nc4'
#------------------------------------------------
lc_arr = xr.open_dataset(lake_cover).cl.data[0]

lsm_arr = xr.open_dataset(land_sea_mask).landseamask.data
# transpose the data to get longitude on the x axis, and flip vertically
# so that latitude is displayed south to north as it should be
lsm_arr = np.flip(lsm_arr.transpose(), axis=0)
#------------------------------------------------

# list_val_dates = ['19880106','19880625','19881215','19880502'] #['2007329','2009006','2009100']

mesh_xy = np.meshgrid(np.arange(-180, 180, 0.1), np.arange(90, -90, -0.1))
#------------------------------------------------
# read one autosnow file and store it metadata
metafile = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_in_geotif/gmasi_snowice_reproc_v003_1987241_0.1deg_wgs.tif'

with rasterio.open(metafile) as dt:

    nh_row = transform.rowcol(dt.transform,-180,45)

    sh_row = transform.rowcol(dt.transform,-180,-45)

    meta_autosnow = dt.meta
#------------------------------------------------

# read the model
# RF_classifier_model_trained_with_ERA5_and_autosnowclim_data_20231201.pkl
rf_mdl_fle = os.path.join(path_to_models,'RF_classifier_model_trained_with_ERA5_20231207.pkl')
with open(rf_mdl_fle, 'rb') as rf_mdl_file:  
    rf_model = pickle.load(rf_mdl_file)

img_extent = (-180, 180, -90, 90)

trns = rasterio.transform.from_origin(-180,90,0.1,0.1)

classes = ['Water','Snow free land', 'Snow covered land','Ice']
"""
0: water
1: snow-free land
2: snow-covered land
3: ice
"""

comn_nme_prt1 = 'based_on_alldata_clim_1999_2022_DOY'
comn_nme_prt2 = '0.1deg_wgs'
comn_nme_prt3 = 'prob_based_on_all_data_clim_1999_2022_DOY'

cde_run_dte = str(date.today().strftime('%Y%m%d'))
#%%
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
    return f'{x / 1e6:.1f}M'
#----------------------------------------------
# fcuntion to make big 1d data from list of arrays
def make_nd_arr(array_lst,nrow,ncol):

    arr3d = np.zeros((nrow,ncol,len(array_lst)))

    for i in enumerate(array_lst):
        arr3d[:,:,i[0]] = array_lst[i[0]]
        del(i)
    # reshape to 1D size for all arrays
    arr1d = arr3d.reshape(nrow*ncol*arr3d.shape[2])

    return arr1d, arr3d

#---------------------------------------------------

def cat_evaluate(lst_of_arrays,xshp):

    orig_arrs = lst_of_arrays[0]
    rf_arrs = lst_of_arrays[1]

    orig_1d,_ = make_nd_arr(orig_arrs,orig_arrs[0].shape[0],xshp)
    rf_1d,_ = make_nd_arr(rf_arrs,rf_arrs[0].shape[0],xshp)

    arr_stck = np.column_stack((orig_1d,rf_1d))

    df_obs_pred = pd.DataFrame(arr_stck,columns=['test','pred'])

    # get the count per class
    n_per_class_in_y = pd.DataFrame(df_obs_pred['test'].value_counts())
    n_per_class_in_y.sort_index(inplace=True)
    n_per_class_in_y['Class']  = classes # per it with the class names
    #---------------------#------------------

    # here we evaluate per class
    lst_cat_stats = []
    for i in enumerate(classes):

        tval = float(i[0])

        class_val = i[1]

        lst_cat_stats.append(my_eval_fucnts.binary_cat_metrics(df_obs_pred,'test','pred',tval,class_val))

    dfs_cat_stats = pd.concat(lst_cat_stats,axis=0)

    dfs_cat_stats = dfs_cat_stats.merge(n_per_class_in_y[['count', 'Class']],right_on='Class',
                                        left_on=dfs_cat_stats.index)

    dfs_cat_stats['label'] = dfs_cat_stats.index

    dfs_cat_stats.index = dfs_cat_stats['Class']

    dfs_cat_stats.drop(columns='Class',inplace=True)

    dfs_cat_stats = pd.DataFrame(dfs_cat_stats.filter(items=['count','Hits', 'Miss', 'False alarms',
                                                            'label','POD', 'FAR', 'POFD', 'ACC', 'CSI',
                                                            'ETS']))
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
    ax.legend(fontsize=xytcklb_sze)
    # Set plot title
    plt.title(plt_tle, fontsize=16)


    #  bbox=dict(boxstyle="round,pad=0.5", edgecolor='k',facecolor='none')
#%%
# The retrieval
# l = '2009006'
print('begin the estimating autosnow using trained RF machine learning algorithm!')
print('****************************')
print('****************************')
print('****************************')

rf_estimated_autosnow_bskt = []
original_autosnow_bskt = []

rf_est_atsnw_nh, rf_est_atsnw_sh = [], []

orig_atsnw_nh, orig_atsnw_sh = [], []

nh_winter_rf_atsnw, nh_spring_rf_atsnw, nh_summer_rf_atsnw, nh_autumn_rf_atsnw = [], [], [], []
sh_winter_rf_atsnw, sh_spring_rf_atsnw, sh_summer_rf_atsnw, sh_autumn_rf_atsnw = [], [], [], []

nh_winter_orig_atsnw, nh_spring_orig_atsnw, nh_summer_orig_atsnw, nh_autumn_orig_atsnw = [], [], [], []
sh_winter_orig_atsnw, sh_spring_orig_atsnw, sh_summer_orig_atsnw, sh_autumn_orig_atsnw = [], [], [], []

count = 0
for l in sorted(all_autosnow_files):

    # define autosnoe file
    # autosnow_file = 'gmasi_snowice_reproc_v003_' + l + '_0.1deg_wgs.tif'
    # autosnow_file = os.path.join(path_to_autosnow_data,autosnow_file)

    yr_DOY = os.path.basename(l).split('_')[4]

    yr  = int(os.path.basename(l).split('_')[4][:4])#int(l[:4])

    doy_ = os.path.basename(l).split('_')[4][-3:]

    doY = int(os.path.basename(l).split('_')[4][-3:]) #int(l[-3:])

    doy = os.path.basename(l).split('_')[4][-3:]

    dt = datetime.datetime(yr, 1, 1) + datetime.timedelta(doY - 1) 

    dt_sve = dt.strftime('%Y%m%d') #str(dt).replace(' ','').replace(':','').replace('-','')

    mnth = pd.to_datetime(dt_sve).month
    #------------------------------------------------------

    # open autosnow files

    dat_auto = xr.open_dataarray(l)   

    # gtr = (dat_auto.spatial_ref.GeoTransform)

    # crs = dat_auto.spatial_ref.crs_wkt

    dat_auto_array = dat_auto.data[0,:,:]

    dat_auto_array = np.where(dat_auto_array > 3,np.nan,dat_auto_array)

    y_shp,x_shp = dat_auto_array.shape[0],dat_auto_array.shape[1] 

    # make nh sh data based on the original data and append to separate list
    orig_atsnw_arr_nh = dat_auto_array[0:nh_row[0],:]
    orig_atsnw_nh.append(orig_atsnw_arr_nh)

    orig_atsnw_arr_sh = dat_auto_array[sh_row[0]:y_shp,:]
    orig_atsnw_sh.append(orig_atsnw_arr_sh)

    # define era5 input files
    dewpnt_2m_file = '2m_dewpoint_temperature_' + str(yr) + '_daily_mean.nc'    
    dewpnt_2m_file = os.path.join(path_to_era5_data,dewpnt_2m_file)

    temp_2m_file = '2m_temperature_' + str(yr) + '_daily_mean.nc'
    temp_2m_file = os.path.join(path_to_era5_data,temp_2m_file)

    sea_ice_file = 'sea_ice_cover_' + str(yr) + '_daily_mean.nc'
    sea_ice_file = os.path.join(path_to_era5_data,sea_ice_file)

    sst_file = 'sea_surface_temperature_' + str(yr) + '_daily_mean.nc'
    sst_file = os.path.join(path_to_era5_data,sst_file)

    skt_file = 'skin_temperature_' + str(yr) + '_daily_mean.nc'
    skt_file = os.path.join(path_to_era5_data,skt_file)

    albedo_file = 'forecast_albedo_' + str(yr) + '_daily_mean.nc'
    albedo_file = os.path.join(path_to_era5_data,albedo_file)

    #------------------------------------------------------
    # clim_autsnw_fle = '_'.join(['autosnow',comn_nme_prt1,doy,comn_nme_prt2]) + '.tif'

    # clim_autsnw_fle = os.path.join(path_to_autosno_climatological_data,clim_autsnw_fle)

    # wtr_cls_prob_fle = '_'.join(['water_class',comn_nme_prt3,doy,comn_nme_prt2]) + '.tif'
    # wtr_cls_prob_fle = os.path.join(path_to_autosno_climatological_data,wtr_cls_prob_fle)

    # snw_free_lnd_cls_prob_fle = '_'.join(['snow_free_land_class',
    #                                       comn_nme_prt3,doy,comn_nme_prt2]) + '.tif'
    # snw_free_lnd_cls_prob_fle = os.path.join(path_to_autosno_climatological_data,snw_free_lnd_cls_prob_fle)

    # snw_cvrd_lnd_cls_prob_fle = '_'.join(['snow_covered_land_class',
    #                                       comn_nme_prt3,doy,comn_nme_prt2]) + '.tif'
    # snw_cvrd_lnd_cls_prob_fle = os.path.join(path_to_autosno_climatological_data,snw_cvrd_lnd_cls_prob_fle)

    # ice_cls_prob_fle = '_'.join(['ice_class',comn_nme_prt3,doy,comn_nme_prt2]) + '.tif'
    # ice_cls_prob_fle = os.path.join(path_to_autosno_climatological_data,ice_cls_prob_fle)
    #------------------------------------------------------

    # read files
    dwpnt_2m_arr = xr.open_dataset(dewpnt_2m_file).sel(time = dt).d2m.values
    dwpnt_2m_arr[np.isnan(dwpnt_2m_arr)] = -99999

    temp_2m_arr  = xr.open_dataset(temp_2m_file).sel(time = dt).t2m.values
    temp_2m_arr[np.isnan(temp_2m_arr)] = -99999

    sea_ice_arr  = xr.open_dataset(sea_ice_file).sel(time=dt).siconc.values
    sea_ice_arr[np.isnan(sea_ice_arr)] = -99999

    sst_arr  = xr.open_dataset(sst_file).sel(time=dt).sst.values
    sst_arr[np.isnan(sst_arr)] = -99999

    skt_arr = xr.open_dataset(skt_file).sel(time=dt).skt.values
    skt_arr[np.isnan(skt_arr)] = -99999

    albedo_arr = xr.open_dataset(albedo_file).sel(time=dt).fal.values
    albedo_arr[np.isnan(albedo_arr)] = -99999

    #------------------------------------------------------
    # the climatology based autonow, water, snow free land, snow covered land and ice class probability data
    # clim_autsnw_data_array = xr.open_dataarray(clim_autsnw_fle)
    # clim_autsnw_data_array = clim_autsnw_data_array.data[0,:,:]
    # clim_autsnw_data_array[np.isnan(clim_autsnw_data_array)] = -99999

    # wtr_cls_prob_data_array = xr.open_dataarray(wtr_cls_prob_fle).data
    # wtr_cls_prob_data_array = wtr_cls_prob_data_array[0,:,:]
    # wtr_cls_prob_data_array[np.isnan(wtr_cls_prob_data_array)] = -99999

    # snw_free_lnd_prob_data_array = xr.open_dataarray(snw_free_lnd_cls_prob_fle).data
    # snw_free_lnd_prob_data_array = snw_free_lnd_prob_data_array[0,:,:]
    # snw_free_lnd_prob_data_array[np.isnan(snw_free_lnd_prob_data_array)] = -99999

    # snw_cvrd_lnd_cls_prob_data_array = xr.open_dataarray(snw_cvrd_lnd_cls_prob_fle).data
    # snw_cvrd_lnd_cls_prob_data_array = snw_cvrd_lnd_cls_prob_data_array[0,:,:]
    # snw_cvrd_lnd_cls_prob_data_array[np.isnan(snw_cvrd_lnd_cls_prob_data_array)] = -99999

    # ice_cls_prob_data_array = xr.open_dataarray(ice_cls_prob_fle).data
    # ice_cls_prob_data_array = ice_cls_prob_data_array[0,:,:]
    # ice_cls_prob_data_array[np.isnan(ice_cls_prob_data_array)] = -99999

    #------------------------------------------------------

    lc_arr[np.isnan(lc_arr)] = -99999

    lsm_arr[np.isnan(lsm_arr)] = -99999

    lat_arr = mesh_xy[1]

    lon_arr = mesh_xy[0]

    doy_arr = np.empty_like(dwpnt_2m_arr)

    doy_arr[:,:] = doY
    #------------------------------------------------------

    # make and append msg data to list
    img_lst = []

    img_lst.append(lon_arr)

    img_lst.append(lat_arr)

    img_lst.append(doy_arr)

    img_lst.append(dwpnt_2m_arr)

    img_lst.append(temp_2m_arr)

    img_lst.append(sst_arr)

    img_lst.append(skt_arr)

    img_lst.append(sea_ice_arr) 

    img_lst.append(albedo_arr)

    # img_lst.append(clim_autsnw_data_array)

    # img_lst.append(wtr_cls_prob_data_array)

    # img_lst.append(snw_free_lnd_prob_data_array)

    # img_lst.append(snw_cvrd_lnd_cls_prob_data_array)

    # img_lst.append(ice_cls_prob_data_array)

    img_lst.append(lc_arr)

    img_lst.append(lsm_arr)        
    #------------------------------------------------------
    # create empty array and stack msg data
    img = np.zeros((y_shp,x_shp,11))

    for i in enumerate(img_lst):
        #print(i[0])
        img[:,:,i[0]] = img_lst[i[0]]
        del(i)
    # reshape to 1D size for all arrays
    img_rshp = img.reshape((y_shp*x_shp),img.shape[2])
    
    # predict autosnow using the trained ML algorithm and reshape back to image size
    rf_predicted = rf_model.predict(img_rshp)
    rf_predicted_autosnow = rf_predicted.reshape(y_shp, x_shp)    

    rf_atsnw_arr_nh = rf_predicted_autosnow[0:nh_row[0],:]
    rf_est_atsnw_nh.append(rf_atsnw_arr_nh)

    rf_atsnw_arr_sh = rf_predicted_autosnow[sh_row[0]:y_shp,:]
    rf_est_atsnw_sh.append(rf_atsnw_arr_sh)
    #------------------------------------------------------

    # # append based on season and hemisphere
    # season_nh = map_fnc.find_season(mnth,'Northern')
    # if season_nh is 'Winter':        
    #     nh_winter_rf_atsnw.append(rf_atsnw_arr_nh) 
    #     nh_winter_orig_atsnw.append(orig_atsnw_arr_nh)       
    #     #----------------------------------------------------

    # elif season_nh is 'Spring':        
    #     nh_spring_rf_atsnw.append(rf_atsnw_arr_nh)
    #     nh_spring_orig_atsnw.append(orig_atsnw_arr_nh)        
    #     #----------------------------------------------------

    # elif season_nh is 'Summer':
    #     # nh_summer_imerg_mw.append(imerg_precip_data_mw_nh)
    #     nh_summer_rf_atsnw.append(rf_atsnw_arr_nh)
    #     nh_summer_orig_atsnw.append(orig_atsnw_arr_nh)
    #     #----------------------------------------------------

    # elif season_nh is 'Autumn':
    #     nh_autumn_rf_atsnw.append(rf_atsnw_arr_nh)
    #     nh_autumn_orig_atsnw.append(orig_atsnw_arr_nh)
    #     #----------------------------------------------------

    # season_sh = map_fnc.find_season(mnth,'Southern')
    # if season_sh is 'Winter':
    #     sh_winter_rf_atsnw.append(rf_atsnw_arr_sh)
    #     sh_winter_orig_atsnw.append(orig_atsnw_arr_sh)
    #     #----------------------------------------------------

    # elif season_sh is 'Spring':
    #     sh_spring_rf_atsnw.append(rf_atsnw_arr_sh)
    #     sh_spring_orig_atsnw.append(orig_atsnw_arr_sh)
    #     #----------------------------------------------------

    # elif season_sh is 'Summer':
    #     sh_summer_rf_atsnw.append(rf_atsnw_arr_sh)
    #     sh_summer_orig_atsnw.append(orig_atsnw_arr_sh)        
    #     #----------------------------------------------------
    # elif season_sh is 'Autumn':
    #     sh_autumn_rf_atsnw.append(rf_atsnw_arr_sh)
    #     sh_autumn_orig_atsnw.append(orig_atsnw_arr_sh)        
    #------------------------------------------------------
    # save the estimated autosnow data to disk
    # gmasi_snowice_reproc_v003_2007001_0.1deg_wgs.tif

    # RF_estimated_autosnow_using_alldataclim

    mp_svneme = '_'.join(['RF_estimated_autosnow_using_only_ERA5_data',yr_DOY,'0.1deg_wgs']) + '.tif'

    mmf.rasterio_based_save_array_to_disk(path_to_put_estimated_autosnw,mp_svneme,meta_autosnow,rf_predicted_autosnow)


    # autosnow_rf_diff = dat_auto_array - rf_predicted_autosnow
    # autosnow_rf_diff[autosnow_rf_diff==0.]= np.nan   

    # append estimates and original to baskets
    # rf_estimated_autosnow_bskt.append(rf_predicted_autosnow)

    # original_autosnow_bskt.append(dat_auto_array) 
    # gc.collect()

    #------------------------------------------------------

    # if dt_sve in list_val_dates:

    #     #  plot
        # mmf.plot_autosnow_cat_map(rf_predicted_autosnow,'global')
    #     plt.savefig(os.path.join(path_to_put_plots,'rf_estimated_autosnow_' + dt_sve +'_Sep27-2023.png'))
    #     plt.close()    

        # mmf.plot_autosnow_cat_map(dat_auto_array,'global')
    #     plt.savefig(os.path.join(path_to_put_plots,'autosnow_orig_' + dt_sve +'_Sep27-2023.png'))
    #     plt.close()

    #     # plot diffeerences
    #     plot_me(autosnow_rf_diff,'jet')
    #     plt.savefig(os.path.join(path_to_put_plots,'rf_autosnow_diff_' + dt_sve + '_Sep27-2023.png'))
    #     plt.close()       

    count += 1

    if count % 500 == 0:
        print(str(count) + ' files are read so far')


print('done!') 

#%%
# print('begin doing evaluation')
# print('**********************************************************')
# clms2plt = ['count', 'Hits', 'Miss', 'False alarms']

# # global
# glob_data_lst = [original_autosnow_bskt, rf_estimated_autosnow_bskt]

# eval_df_global = cat_evaluate(glob_data_lst,x_shp)
# svnem_csv = '_'.join(['RF_with_clim_categorical_stats_global',cde_run_dte]) + '.csv'
# eval_df_global.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_global = eval_df_global[clms2plt]

# svnem_plt = '_'.join(['RF_with_clim_categorical_stats_global',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_global,15,13,'Global')
# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))
# #-----------------------------------------------------------------------------------------
# # nh
# nh_data_lst = [orig_atsnw_nh, rf_est_atsnw_nh]

# eval_df_nh = cat_evaluate(nh_data_lst,x_shp)
# svnem_csv = '_'.join(['RF_with_clim_categorical_stats_nh',cde_run_dte]) + '.csv'
# eval_df_nh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_nh = eval_df_nh[clms2plt]

# svnem_plt = '_'.join(['RF_with_clim_categorical_stats_nh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_nh,15,13,'Northern Hemisphere')
# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))
# gc.collect()
# #-----------------------------------------------------

# # sh
# sh_data_lst = [orig_atsnw_sh, rf_est_atsnw_sh]

# eval_df_sh = cat_evaluate(sh_data_lst,x_shp)
# svnem_csv = '_'.join(['RF_with_clim_categorical_stats_sh',cde_run_dte]) + '.csv'
# eval_df_sh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_sh = eval_df_sh[clms2plt]

# svnem_plt = '_'.join(['RF_with_clim_categorical_stats_sh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_sh,15,13, 'Southern Hemisphere')
# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))

# #-----------------------------------------------------------------------------------------

# # seasons: NH
# # winter
# nh_winter_data_lst = [nh_winter_orig_atsnw, nh_winter_rf_atsnw]

# winter_eval_df_nh = cat_evaluate(nh_winter_data_lst,x_shp)
# svnem_csv = '_'.join(['RF_with_clim_categorical_stats_winter_nh',cde_run_dte]) + '.csv'
# winter_eval_df_nh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_winter_nh = winter_eval_df_nh[clms2plt]

# svnem_plt = '_'.join(['RF_with_clim_categorical_stats_winter_nh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_winter_nh,15,13,'NH - DJF')
# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))

# gc.collect()
# #--------------------
# # spring
# nh_spring_data_lst = [nh_spring_orig_atsnw, nh_spring_rf_atsnw]

# spring_eval_df_nh = cat_evaluate(nh_spring_data_lst,x_shp)
# svnem_csv = '_'.join(['RF_with_clim_categorical_stats_spring_nh',cde_run_dte]) + '.csv'
# spring_eval_df_nh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_spring_nh = spring_eval_df_nh[clms2plt]

# svnem_plt = '_'.join(['RF_with_clim_categorical_stats_spring_nh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_spring_nh,15,13,'NH - MAM')
# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))
# #--------------------
# # summer
# nh_summer_data_lst = [nh_summer_orig_atsnw, nh_summer_rf_atsnw]
# summer_eval_df_nh = cat_evaluate(nh_summer_data_lst,x_shp)
# svnem_csv = '_'.join(['RF_with_clim_categorical_stats_summer_nh',cde_run_dte]) + '.csv'
# summer_eval_df_nh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_summer_nh = summer_eval_df_nh[clms2plt]
# svnem_plt = '_'.join(['RF_with_clim_categorical_stats_summer_nh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_summer_nh,15,13,'NH - JJA')
# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))

# gc.collect()

# #--------------------
# # autumn
# nh_autumn_data_lst = [nh_autumn_orig_atsnw, nh_autumn_rf_atsnw]
# autumn_eval_df_nh = cat_evaluate(nh_autumn_data_lst,x_shp)
# svnem_csv = '_'.join(['RF_with_clim_categorical_stats_autumn_nh',cde_run_dte]) + '.csv'
# autumn_eval_df_nh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_autumn_nh = autumn_eval_df_nh[clms2plt]
# svnem_plt = '_'.join(['RF_with_clim_categorical_stats_autumn_nh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_autumn_nh,15,13,'NH - SON')
# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))

# #-----------------------------------------------------------------------------------------

# # seasons: SH
# # winter
# sh_winter_data_lst = [sh_winter_orig_atsnw, sh_winter_rf_atsnw]

# winter_eval_df_sh = cat_evaluate(sh_winter_data_lst,x_shp)
# svnem_csv = '_'.join(['RF_with_clim_categorical_stats_winter_sh',cde_run_dte]) + '.csv'
# winter_eval_df_sh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_winter_sh = winter_eval_df_sh[clms2plt]

# svnem_plt = '_'.join(['RF_with_clim_categorical_stats_winter_sh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_winter_sh,15,13,'SH - JJA')
# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))
# gc.collect()
# #--------------------
# # spring
# sh_spring_data_lst = [sh_spring_orig_atsnw, sh_spring_rf_atsnw]

# spring_eval_df_sh = cat_evaluate(sh_spring_data_lst,x_shp)
# svnem_csv = '_'.join(['RF_with_clim_categorical_stats_spring_sh',cde_run_dte]) + '.csv'
# spring_eval_df_sh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_spring_sh = spring_eval_df_sh[clms2plt]

# svnem_plt = '_'.join(['RF_with_clim_categorical_stats_spring_sh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_spring_sh,15,13,'SH - SON')
# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))
# #--------------------
# # summer
# sh_summer_data_lst = [sh_summer_orig_atsnw, sh_summer_rf_atsnw]
# summer_eval_df_sh = cat_evaluate(sh_summer_data_lst,x_shp)
# svnem_csv = '_'.join(['RF_with_clim_categorical_stats_summer_sh',cde_run_dte]) + '.csv'
# summer_eval_df_sh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_summer_sh = summer_eval_df_sh[clms2plt]
# svnem_plt = '_'.join(['RF_with_clim_categorical_stats_summer_sh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_summer_sh,15,13,'SH - DJF')
# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))
# gc.collect()

# #--------------------
# # autumn
# sh_autumn_data_lst = [sh_autumn_orig_atsnw, sh_autumn_rf_atsnw]
# autumn_eval_df_sh = cat_evaluate(sh_autumn_data_lst,x_shp)
# svnem_csv = '_'.join(['RF_with_clim_categorical_stats_autumn_sh',cde_run_dte]) + '.csv'
# autumn_eval_df_sh.to_csv(os.path.join(path_to_put_df,svnem_csv))

# sub_eval_df_autumn_sh = autumn_eval_df_sh[clms2plt]
# svnem_plt = '_'.join(['RF_with_clim_categorical_stats_autumn_sh',cde_run_dte]) + '.png'
# plot_bar(sub_eval_df_autumn_sh,15,13,'SH - MAM')
# plt.savefig(os.path.join(path_to_put_plots,svnem_plt))


# gc.collect()

# print('done!')
#%%
'''
# do some evaluation
# import evaluation_fucntions_algorithms as my_eval_fucnts
# some info
# cat stats
arr_stck = np.column_stack((big_1d_org_autosnow_arr,big_1d_rf_autosnow_arr))

df_obs_pred = pd.DataFrame(arr_stck,columns=['test','pred'])

n_per_class_in_y = pd.DataFrame(df_obs_pred['test'].value_counts())

n_per_class_in_y.sort_index(inplace=True)

n_per_class_in_y['Class']  = classes
#---------------------#------------------#------------------#---------------------------

lst_cat_stats = []
for i in enumerate(classes):

    tval = float(i[0])

    class_val = i[1]

    lst_cat_stats.append(my_eval_fucnts.binary_cat_metrics(df_obs_pred,'test','pred',tval,class_val))

dfs_cat_stats = pd.concat(lst_cat_stats,axis=0)

dfs_cat_stats = dfs_cat_stats.merge(n_per_class_in_y[['count', 'Class']],right_on='Class',
                                    left_on=dfs_cat_stats.index)

dfs_cat_stats['label'] = dfs_cat_stats.index

dfs_cat_stats.index = dfs_cat_stats['Class']

dfs_cat_stats.drop(columns='Class',inplace=True)

dfs_cat_stats = pd.DataFrame(dfs_cat_stats.filter(items=['count','Hits', 'Miss', 'False alarms',
                                                         'label','POD', 'FAR', 'POFD', 'ACC', 'CSI',
                                                         'ETS']))

dfs_cat_stats.to_csv(os.path.join(path_to_put_df,'RF_with_clim_categorical_stats_model_Oct18.csv'))
'''


#%%
# st_met = np.apply_along_axis(lambda a,b: my_eval_fucnts.binary_cat_metrics(a,b),2,arr=np.array[rf_autosnow_arr,org_autosnow_arr])
# st_met = np.apply_along_axis(my_eval_fucnts.binary_cat_metrics, axis=2, arr=np.array([rf_autosnow_arr,org_autosnow_arr]))
#%%
# plot some features
'''
import cartopy.crs as ccrs

proj=ccrs.PlateCarree()

# albedo_arr_grad_cpy = albedo_arr_grad.copy()
# albedo_arr_grad_cpy[albedo_arr_grad_cpy == 0] = np.nan

c_map = cm.get_cmap('jet') #.gist_rainbow#

c_map.set_bad('white')

ax = plt.axes(projection=proj)

ax_plt = ax.imshow(dat_auto_array, extent=img_extent, cmap=c_map)  #
ax.coastlines()
gls_ax = ax.gridlines(crs=proj,color='grey', linestyle='--', lw = 0.35, draw_labels={"bottom": "x", "left": "y"})
# add these before plotting
gls_ax.xlabel_style={'size':15}   
gls_ax.ylabel_style={'size':15}
ax.text(-0.15, 0.55, 'Latitude', va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',fontsize=15,
            transform=ax.transAxes)
    
ax.text(0.5, -0.2, 'Longitude', va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',fontsize=15,
        transform=ax.transAxes)

ax.set_yticklabels([-80,-40,0,40,80])
plt.tick_params(axis='both',  labelsize=15)
plt.tight_layout()
cbar = plt.colorbar(ax_plt,orientation='horizontal',shrink=0.8,ax=ax)
cbar.ax.tick_params(labelsize=12)

plt.savefig(os.path.join(path_to_put_plots,'albedo_grad' + dt_sve + 'Sep14-2023.png'))
plt.close()
'''

#%%
# rf_mdl_fle_test = os.path.join(path_to_models,'autosnow_estimation_rf_classifier_test_model_Aug232023.pkl')
# with open(rf_mdl_fle_test, 'rb') as rf_mdl_file_test:  
#     rf_model_test = pickle.load(rf_mdl_file_test)
# mmf.plot_autosnow_cat_map(autosnow_knn_predicted)
# plt.savefig(os.path.join(path_to_put_plots,'autosnow_knn_predicted_' + dt_sve + '.png'))
# plt.close()

# mmf.plot_autosnow_cat_map(autosnow_hgbc_predicted)
# plt.savefig(os.path.join(path_to_put_plots,'autosnow_hgbc_predicted_' + dt_sve + '.png'))
# plt.close()

# dat_auto_array_rshp = dat_auto_array.reshape(y_shp*x_shp)
# scat_df = pd.DataFrame(list(zip(dat_auto_array_rshp,rf_predicted)),columns=['Autosnow','RF'])
# #ax = plt.axes()
# plt.scatter(dat_auto_array.ravel(),autosnow_rf_predicted.ravel())


#%%
# code extracts
# knn_mdl_fle = os.path.join(path_to_models,'autosnow_estimation_knn_classifier_model.pkl')
# with open(knn_mdl_fle, 'rb') as knn_mdl_file:  
#     knn_model = pickle.load(knn_mdl_file)

# hgbc_mdl_fle = os.path.join(path_to_models,'autosnow_estimation_hgbc_classifier_model.pkl')
# with open(hgbc_mdl_fle, 'rb') as hgbc_mdl_file:  
#     hgbc_model = pickle.load(hgbc_mdl_file)

# knn_predicted = knn_model.predict(img_rshp)
# autosnow_knn_predicted = knn_predicted.reshape(y_shp, x_shp)

# autosnow_knn_diff = dat_auto_array - autosnow_knn_predicted
# autosnow_knn_diff[autosnow_knn_diff==0.]= np.nan

# hgbc_predicted = hgbc_model.predict(img_rshp)
# autosnow_hgbc_predicted = hgbc_predicted.reshape(y_shp, x_shp)

# autosnow_hgbc_diff = dat_auto_array - autosnow_hgbc_predicted
# autosnow_hgbc_diff[autosnow_hgbc_diff==0.]= np.nan

# plot_me(autosnow_knn_diff,'jet')
# plt.savefig(os.path.join(path_to_put_plots,'knn_autosnow_diff_' + dt_sve + '.png'))
# plt.close()  

# plot_me(autosnow_hgbc_diff,'jet')
# plt.savefig(os.path.join(path_to_put_plots,'hgbc_autosnow_diff_' + dt_sve + '.png'))
# plt.close()  

# knn_out_vector_file = os.path.join(path_to_intermediates,'autosnow_knn_pred_' + dt_sve + '.shp')

# hgbc_out_vector_file = os.path.join(path_to_intermediates,'autosnow_hgbc_pred_' + dt_sve + '.shp')

 # # autosnow_knn_pred_plgn = mmf.array_to_vector(autosnow_knn_predicted,knn_out_vector_file,
# #                                             None,crs,trns)
# # autosnow_knn_pred_plgn.to_crs(crs={'proj':'cea'},inplace=True)
# # autosnow_knn_pred_plgn['area'] = autosnow_knn_pred_plgn['geometry'].area/10**6
# # knn_area_list = autosnow_knn_pred_plgn.groupby('raster_val')['area'].sum().to_list()

# # autosnow_hgbc_pred_plgn = mmf.array_to_vector(autosnow_hgbc_predicted,hgbc_out_vector_file,
# #                                             None,crs,trns)
# autosnow_hgbc_pred_plgn.to_crs(crs={'proj':'cea'},inplace=True)
# autosnow_hgbc_pred_plgn['area'] = autosnow_hgbc_pred_plgn['geometry'].area/10**6
# hgbc_area_list = autosnow_hgbc_pred_plgn.groupby('raster_val')['area'].sum().to_list()


