'''
Autosnow data:
Autosnow data files are flat binary, 1-byte per pixel, 9000 elements per record, 4500 records per file,
compressed with linux compress command. Global snow and ice maps are provided on a latitude-longitude
(Plate Carree) grid oriented north to south and west to east with the grid cell size of 1/25 of a
degree or about 4 km at the equator. The location of the upper left corner of the upper left pixel of the
map is 90N, 180W.
With this code, the data is:

(1) the data is preprocessed in its flat binary form and saved to tif in its native 0.04 degree resolution and spatial reference
(2) resample/regrid the 0.04 degree data to desired 0.1 degree data in native cordinate reference system
'''

#%%
# import packages
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import struct
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import my_functions as mf
from osgeo import gdal,gdalconst
import zipfile
#%%
# define path to data
path_to_bin_files = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_raw_files'
path_to_put_tif_files = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_in_geotif'
path_to_put_2007_files = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_geotif_2007'
#%%
# define functions to use
 # function to save map to disk
def save_array_to_disk(patth,svenme,metadata,array2save):
    '''
	functions saves an array to .tif
	requires:
    patth = the path to save the map
    file name for saving the map (add .extension e.g. .tif)
    map meta data
    array to save 
    '''
    with rasterio.open(os.path.join(patth,svenme),'w',**metadata) as mp:
        mp.write(array2save,indexes=1)

# function to resample map from one grid resolution to the other
def resample_map(outfile_name,inputfile_name,output_format,xgrid_res,ygrid_res,
                 dest_crfs,resampling_tech):
    '''
    functions regrids/resample a given map from its native grid resolution to a given grid resolution
    requires:
    outfile_name = the output file name; this should include the path to save the file
    inputfile_name = the input map to be resampled/regrided
    output_format = the format of the output map e.g. geotif; in string
    xgrid_res = grid resolution in x axis / longitude axis
    ygrid_res = grid resolution in y axis / latitude axis
    dest_crfs = spatial reference of the destination map
    resampling_tech = the resampling technique/algorithm e.g. average, mode, median, nearest neighbor, bilinear etc; in string
    '''
    out_ds = gdal.Translate(destName = outfile_name,srcDS = inputfile_name,
                            format = output_format,xRes = xgrid_res,yRes = ygrid_res,
                            outputSRS = dest_crfs, resampleAlg = resampling_tech)
    del(out_ds)

# fucntion to zip a file
def zip_a_file(file_name,file2zip,arnme):
    '''
    file_name =  the ziped file name (path + file name) i.e ends with ".zip"
    file2zip = the file to be zipped
    arnme = the name of the file inside the zipped file
    '''
    with zipfile.ZipFile(file_name, 'w',zipfile.ZIP_DEFLATED) as myzip: # compress_type=
        myzip.write(file2zip,arnme)

#%%
# floating variables
crfs_source = 'ESRI:54001' #'+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs' # coordinate Reference System is esri projection 54001 - world plate carree
crfs_dest = '+proj=longlat +datum=WGS84 +no_defs +type=crs'
trns04 = rasterio.transform.from_origin(-180,90,0.04,0.04)

#%%
# the bin file processing to 2D array
count = 0
for fle in sorted([os.path.join(path_to_bin_files,f) for f in os.listdir(path_to_bin_files) if f.startswith('gmasi_snowice_reproc_v003_2007')]):
    count = count + 1

    # make file names 
    file_save_name = os.path.basename(fle) + '_0.04deg.tif'

    # reample the geotif map from 0.04 to 0.1 degree resolution
    input_file = os.path.join(path_to_put_tif_files,file_save_name) # file to be regrided
    output_file_name = os.path.basename(input_file).replace('0.04deg','0.1deg_wgs')
    output_file = os.path.join(path_to_put_tif_files,output_file_name) # output file in Plate Carree (pc)

    if os.path.isfile(output_file):
       continue
    else:
        print(file_save_name)

    with open(fle, mode='rb') as file: # b is important -> 
        fileContent = file.read()

    arr = np.reshape(np.array([i for i in fileContent]), (4500, 9000))
    #arr = np.where(arr > 3, np.nan,arr)
    arr = arr.astype(np.int16)

    nrows_h,ncols_w = arr.shape[0],arr.shape[1]

    # saving the 2d array to disk as amap
    meta_dat = {'driver':'GTiff','width':ncols_w,'height':nrows_h,
                'count':1,'dtype':np.int16,'crs':crfs_source, 'transform':trns04}
    # save map
    save_array_to_disk(path_to_put_tif_files,file_save_name,meta_dat,arr) 

    # the resampling/regriding command using gdal
    # with this, the coordinate reference system of the output map is also changed  
    resample_map(output_file,input_file,'GTiff',0.1,0.1,crfs_dest,'mode')
    # print(os.path.basename(fle).split('_')[-1][:4])

    #zip the 0.04 degree file
    file_name_of_zipfile = input_file.replace ('.tif','.zip') # + '.zip' #
    arcnnme = os.path.basename(input_file)
    #zip_a_file(file_name_of_zipfile,input_file)
    zip_a_file(file_name_of_zipfile,input_file,arcnnme)

    # # remove files to save disk space
    os.remove(input_file)

    if count % 100 == 0:
        print(str(count) + ' Autosnow files processed so far')    
print('***************** done with code *************')

#%%
'''
# visualy comparing the bin to array autosnow with existing autosnow in hdf to validate the code used
# making some grid for plot
lon = np.arange(-180, 180, 0.04)
#lon_org = np.hstack((np.arange(0, 180, 0.04), np.arange(-180, 0, 0.04)))

lat = np.reshape(np.arange(90, -90, -0.04), (-1, 1))

lon_arr = np.tile(lon, (lat.shape[0],1))
lat_arr = np.tile(lat, lon.shape[0])

plt.figure()
cbar = plt.pcolormesh(lon_arr,lat_arr, arr)

plt.colorbar(mappable=cbar)

#%%
fn = '/ra1/pubdat/autosnow_dave/hdf/2007/autosnow_global.v001_20070101.hdf'

as_lon = mf.AutoSnow_HDF_SDreader(fn, 'LON')
as_lat = mf.AutoSnow_HDF_SDreader(fn, 'LAT')
# as_lat = np.reshape(as_lat, (-1, 1))
as_index = np.flip(mf.AutoSnow_HDF_SDreader(fn, 'LAND_SURF_INDEX'), axis=0)

as_lon_arr = np.tile(as_lon, (as_lat.shape[0],1))
as_lat_arr = np.tile(np.reshape(as_lat, (-1, 1)), as_lon.shape[0])

'''