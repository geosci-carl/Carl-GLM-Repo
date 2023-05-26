# -*- coding: utf-8 -*-
"""
Initial analysis
Goal: open .nc file in Python

first create conda environment in miniconda:
conda create -n spyder-env -y
conda activate spyder-env
conda install spyder-kernels netCDF4 -y
conda install spyder-kernels geopandas -y
"""

import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import geopandas

# set the correct working directory
from os import chdir, getcwd
wd = getcwd()
chdir(wd)

# load in our nc file
file = netCDF4.Dataset('daymet_data/11748_1980_prcp.nc')
prcp = file.variables['prcp'] # grap precipitation
lat = file.variables["lat"]
lon = file.variables["lon"]
latvals = lat[:]
lonvals = lon[:]
