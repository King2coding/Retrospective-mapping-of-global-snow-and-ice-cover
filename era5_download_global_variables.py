"""
download global vraibles: 
land sea mask: This parameter is the proportion of land, as opposed to ocean or inland waters 
(lakes, reservoirs, rivers and coastal waters), in a grid box. This parameter has values ranging 
between zero and one and is dimensionless. In cycles of the ECMWF Integrated Forecasting System (IFS) 
from CY41R1 (introduced in May 2015) onwards, grid boxes where this parameter has a value above 0.5 
can be comprised of a mixture of land and inland water but not ocean. Grid boxes with a value of 0.5 
and below can only be comprised of a water surface. In the latter case, the lake cover is used to 
determine how much of the water surface is ocean or inland water. In cycles of the IFS before CY41R1, 
grid boxes where this parameter has a value above 0.5 can only be comprised of land and those grid boxes 
with a value of 0.5 and below can only be comprised of ocean. In these older model cycles, there is no 
differentiation between ocean and inland water. This parameter does not vary in time.

lake cover: This parameter is the proportion of a grid box covered by inland water bodies 
(lakes, reservoirs, rivers and coastal waters). Values vary between 0: no inland water, 
and 1: grid box is fully covered with inland water. 
This parameter is specified from observations and does not vary in time.
"""
import cdsapi
import os
# path to put data
path_to_put_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/global_variables'


c = cdsapi.Client()

for yr in ['2007', '2008', '2009','2010',]:
    
    for vr in ['lake_cover', 'land_sea_mask',]:

        c.retrieve(
            'reanalysis-era5-single-levels',

            {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable':vr ,
            'year': yr,
            'time': '23:00',
            'day': '31',
            'month': '12',
        },
        os.path.join(path_to_put_data,vr + '_' + yr +'_download.nc'))