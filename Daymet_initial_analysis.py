# -*- coding: utf-8 -*-
"""
Initial analysis
Goal: open .nc and .shp files in Python

first create conda environment in miniconda:
conda create -n spyder-geo
conda activate spyder-geo
conda install -c conda-forge spyder-kernels
conda install -c conda-forge geopandas 
conda install -c conda-forge netcdf4
conda install -c conda-forge xarray


"""
#%%
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr


#%%

# Open our .nc file
ds = xr.open_dataset('daymet_data/11748_1980_prcp.nc')

# plot
plt.contourf(ds['prcp'][364,:,:])
plt.colorbar()
plt.savefig("exports/prcp.svg")
#%%
# Read the shapefile
huc4 = gpd.read_file("shapefiles/NHD_H_0410_HU4_Shape/Shape/WBDHU4.shp")
huc4.plot() # plot it
plt.savefig("exports/shp.svg")

