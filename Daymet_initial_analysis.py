# -*- coding: utf-8 -*-
"""
Initial analysis
Goal: open .nc and .shp files in Python

first create conda environment in miniconda:
conda create -n spyder-geo
conda activate spyder-geo
conda install -c conda-forge spyder-kernels
conda install -c conda-forge geopandas 
conda install -c conda-forge fiona 
conda install -c conda-forge netcdf4

"""

import netCDF4
import geopandas as gpd
#import fiona


# load in our nc file
file = netCDF4.Dataset('daymet_data/11748_1980_prcp.nc')
prcp = file.variables['prcp'] # grap precipitation
lat = file.variables["lat"]
lon = file.variables["lon"]
latvals = lat[:]
lonvals = lon[:]

# open shapefile and plot it
gdf = gpd.read_file("shapefiles/DaymetV4_Tiles_Continental_HI_PR.shp") # open it
gdf.plot() # plot it
