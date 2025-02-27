'''
Goal: Extend autosnow data to before 1987 using EAR5-based surface variables

Code trains ML with EAR5 data vairiables:'2m_dewpoint_temperature', '2m_temperature', 'sea_ice_cover',
                    'sea_surface_temperature', 'skin_temperature', 'forecast_albedo'
for estimating autosnow data
'''
#%%
# import packages
import os
import pandas as pd
import numpy as np
from datetime import date
import pickle

# import cartopy.crs as ccrs
# import cartopy as cart
# import cartopy.feature as cfeature
# import seaborn as sns
import matplotlib.pyplot as plt

import ml_algs as mls
import evaluation_fucntions_algorithms as my_eval_fucnts

# from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import  train_test_split
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay # accuracy_score, classification_report,

#%%
path_to_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/era5_training_df_that_estimated_autosnow'

path_to_put_plots = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/plots'

path_to_save_model = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/trained_models'

#%%
# declare global variables
# read data
big_df = pd.read_pickle(os.path.join(path_to_data,'big_df_clim_autosn_included_code_run_on_20231130.pkl')) # _Aug112023 ,big_df_Sep122023


# projc = ccrs.PlateCarree()#ccrs.epsg(4326)

classes = ['Water','Snow free land', 'Snow covered land','Ice']

target_names = classes#['0','1','2','3']

cde_run_dte = str(date.today().strftime('%Y%m%d'))
#%%
# # do some analysis
# # check for anomalies: compare areas where autosnow classified as snow free land (1) and snow covered land with era5 sst
# sncl = big_df.loc[(big_df['autosnow'] == 1) | (big_df['autosnow'] == 2)]

# sncl_ = sncl.loc[~sncl['sst'].isnull()]


#%%
# stats
df_stats = big_df.groupby('autosnow')[['2m_dewpt','2m_temp','sea_ice','sst','skin_temp','albedo_mean']].mean()


df_stats['Class']  = classes

df_stats = pd.DataFrame(df_stats.filter(['Class','2m_dewpt','2m_temp','sea_ice','sst','skin_temp','albedo_mean'])).round(2)

#df_stats.to_csv(os.path.join(path_to_data,'data_stats.csv'),header=True)

n_per_class = pd.DataFrame(big_df['autosnow'].value_counts())

n_per_class['Class']  = classes

#%%

# preparing data for the ML training and prediction
# Split the data into features (X) and target (y)
# features = ['longitude','latitude','DOY', '2m_dewpt','2m_temp','sst', 'skin_temp', 'sea_ice', 
#             'albedo_mean' , 'clim_auto', 'wtr_cls_prob', 'snw_fre_lnd_cls_prob', 
#             'snw_cvrd_lnd_cls_prob', 'ice_cls_prob', 'lake_cover', 'land_sea_mask']

features = ['longitude','latitude','DOY', '2m_dewpt','2m_temp','sst', 'skin_temp', 'sea_ice', 
            'albedo_mean', 'lake_cover', 'land_sea_mask']
'''
['longitude','latitude','DOY','2m_dewpt','2m_temp', 'sst', 
            'skin_temp','sea_ice', 'albedo_mean', 'clim_auto_arr',
            'clim_auto_std','lake_cover','land_sea_mask'] # 
# 'albedo_min', 'albedo_max','albedo_grad',
'''
#features = ['latitude','year','DOY','2m_temp','sea_ice', 'sst','skin_temp','albedo']

target = ['autosnow']

big_df_ = big_df.copy()

X = big_df_.loc[:,features].copy()

X.fillna(-99999,inplace=True)

y = big_df_.loc[:,target].copy()

y.fillna(-99999,inplace=True)

# Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)

# n_per_class_in_y = pd.DataFrame(y_test['autosnow'].value_counts())

# n_per_class_in_y['Class']  = classes

# X_train, X_test, y_train, y_test = X_train[:].values,X_test[:].values, y_train[:].values, y_test[:].values

#%%

# the rf model traning and fitting
rfc_model = mls.rf_class(X,y,run_type='basic') # train using all data for retrieveal

# rfc_model = mls.rf_class(X_train,y_train,run_type='basic')

# rf_y_pred = rfc_model.predict(X_test)

# feature_importances = pd.Series(rfc_model.feature_importances_, index=features).sort_values(ascending=False)

# # Plot feature imporatnce
# ax = feature_importances.plot.bar(title ='feature importance', rot=20);
# ax.tick_params('both', labelsize=15)
# plt.tight_layout()
# plt.savefig(os.path.join(path_to_put_plots,'feature_importance.png'))
# plt.close()
#================================================================================================================
# do some evaluation

# cat stats
# df_obs_pred = pd.DataFrame(list(zip(y_test,rf_y_pred)),columns=['test','pred'])
# lst_cat_stats = []
# for i in enumerate(classes):

#     tval = float(i[0])

#     class_val = i[1]
    
#     lst_cat_stats.append(my_eval_fucnts.binary_cat_metrics(df_obs_pred,'test','pred',tval,class_val))

# dfs_cat_stats = pd.concat(lst_cat_stats,axis=0)

# dfs_cat_stats = dfs_cat_stats.merge(n_per_class_in_y[['count', 'Class']],right_on='Class',
#                                     left_on=dfs_cat_stats.index)

# dfs_cat_stats['label'] = dfs_cat_stats.index

# dfs_cat_stats.index = dfs_cat_stats['Class']

# dfs_cat_stats.drop(columns='Class',inplace=True)

# dfs_cat_stats = pd.DataFrame(dfs_cat_stats.filter(items=['count','Hits', 'Miss', 'False alarms',
#                                                          'label','POD', 'FAR', 'POFD', 'ACC', 'CSI',
#                                                          'ETS']))
# dfs_cat_stats.to_csv(os.path.join(path_to_data,'RF_with_clim_categorical_stats_test_model_Oct18.csv'))

# # create confusion matrix
# cm_rf = confusion_matrix(y_test, rf_y_pred)

# # confusion matrix plot
# ConfusionMatrixDisplay(confusion_matrix=cm_rf).plot()
# plt.savefig(os.path.join(path_to_put_plots,'confusion_mat_rf_Sep14.png'))
# plt.close()

# save the the trained RF model as a pickle file
# "RF_classifier_model_trained_with_ERA5_and_autosnowclim_data"
autosnow_estimation_rf_classifier_model = '_'.join(["RF_classifier_model_trained_with_ERA5" ,cde_run_dte]) + '.pkl'  

with open(os.path.join(path_to_save_model,autosnow_estimation_rf_classifier_model), 'wb') as rf_model_file:  
    pickle.dump(rfc_model, rf_model_file)

#%%
# scale the data for the models that require data scaling
# sc=StandardScaler()

# scaler = sc.fit(X_train)
# trainX_scaled = scaler.transform(X_train)
# testX_scaled = scaler.transform(X_test)
# # plots
# # plotting anomaly
# ax = plt.axes(projection=projc)

# resol = '50m'  # use data at this scale
# # bodr = cart.feature.NaturalEarthFeature(category='cultural', 
# #     name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
# land = cart.feature.NaturalEarthFeature('physical', 'land', \
#     scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])

# # ocean = cart.feature.NaturalEarthFeature('physical', 'ocean', \
# #     scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])

# lakes = cart.feature.NaturalEarthFeature('physical', 'lakes', \
#     scale=resol, edgecolor='b', facecolor=cfeature.COLORS['water'])
# # rivers = cart.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', \
# #     scale=resol, edgecolor='b', facecolor='none')

# ax.add_feature(land, facecolor='beige')
# # ax.add_feature(ocean, linewidth=0.2 )
# ax.add_feature(lakes)
# # ax.add_feature(rivers, linewidth=0.5)
# # ax.add_feature(bodr, linestyle='--', edgecolor='k', alpha=1)

# sns.scatterplot(data=sncl_,x='longitude',y='latitude',style='autosnow',
#                 palette=['g','r'],hue=sncl_['autosnow'], ax= ax)

# ax.coastlines(zorder = 3)

# ax.text(-0.15, 0.55, 'Latitude', va='bottom', ha='center',
#         rotation='vertical', rotation_mode='anchor',fontsize=15,
#         transform=ax.transAxes)
# ax.text(0.5, -0.2, 'Longitude', va='bottom', ha='center',
#         rotation='horizontal', rotation_mode='anchor',fontsize=15,
#         transform=ax.transAxes)

# gls_ax = ax.gridlines(crs=projc,color='grey', linestyle='--', lw = 0.35, 
#                       draw_labels={"bottom": "x", "left": "y"})

# gls_ax.xlabel_style={'size':15}   
# gls_ax.ylabel_style={'size':15}

# plt.savefig(os.path.join(path_to_put_plots,'anomaly.png'))
# plt.close()

# #===============================================================================================================================
# # plot classes count

# ax = n_per_class.plot.bar(x='Class',y='count',rot=20,legend=False)
# ax.tick_params('both', labelsize=15)
# plt.tight_layout()

# plt.savefig(os.path.join(path_to_put_plots,'count_per_class.png'))
# plt.close()

# #===============================================================================================================================


# define the NL
#%%
# extracts
# report_rf = classification_report(y_test, y_pred,target_names=target_names)
# print(report_rf)

# report_rf = pd.DataFrame(classification_report(y_test, rf_y_pred, target_names = target_names, output_dict = True)).round(2) #

# report_rf.to_csv(os.path.join(path_to_data,'report_rf.csv'))

#accuracy = accuracy_score(y_test, y_pred)


# Create the confusion matrix