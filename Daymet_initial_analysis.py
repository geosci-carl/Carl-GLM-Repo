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
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import numpy as np
import time
#import netCDF4

#%% x array - open precip datafiles

# Open our .nc files
# Tile 1
ds1a = xr.open_dataset('daymet_data/11748_1980_prcp.nc')
ds1b = xr.open_dataset('daymet_data/11748_1981_prcp.nc')
ds1c = xr.open_dataset('daymet_data/11748_1982_prcp.nc')
ds1d = xr.open_dataset('daymet_data/11748_1983_prcp.nc')


# Tile 2
ds2a = xr.open_dataset('daymet_data/11749_1980_prcp.nc')
ds2b = xr.open_dataset('daymet_data/11749_1981_prcp.nc')
ds2c = xr.open_dataset('daymet_data/11749_1982_prcp.nc')
ds2d = xr.open_dataset('daymet_data/11749_1983_prcp.nc')

# Tile 3
ds3a = xr.open_dataset('daymet_data/11750_1980_prcp.nc')
ds3b = xr.open_dataset('daymet_data/11750_1981_prcp.nc')
ds3c = xr.open_dataset('daymet_data/11750_1982_prcp.nc')
ds3d = xr.open_dataset('daymet_data/11750_1983_prcp.nc')

# Tile 4
ds4a = xr.open_dataset('daymet_data/11928_1980_prcp.nc')
ds4b = xr.open_dataset('daymet_data/11928_1980_prcp.nc')
ds4c = xr.open_dataset('daymet_data/11928_1980_prcp.nc')
ds4d = xr.open_dataset('daymet_data/11928_1980_prcp.nc')

# Tile 5
ds5a = xr.open_dataset('daymet_data/11929_1980_prcp.nc')
ds5b = xr.open_dataset('daymet_data/11929_1981_prcp.nc')
ds5c = xr.open_dataset('daymet_data/11929_1982_prcp.nc')
ds5d = xr.open_dataset('daymet_data/11929_1983_prcp.nc')

# Tile 6
ds6a = xr.open_dataset('daymet_data/11930_1980_prcp.nc')
ds6b = xr.open_dataset('daymet_data/11930_1981_prcp.nc')
ds6c = xr.open_dataset('daymet_data/11930_1982_prcp.nc')
ds6d = xr.open_dataset('daymet_data/11930_1983_prcp.nc')

# grab lats
lats_ds1 = ds1a["lat"]
lats_ds2 = ds2a["lat"]
lats_ds3 = ds3a["lat"]
lats_ds4 = ds4a["lat"]
lats_ds5 = ds5a["lat"]
lats_ds6 = ds6a["lat"]

lats_ds1_numpy = lats_ds1.to_numpy()
lats_ds2_numpy = lats_ds2.to_numpy()
lats_ds3_numpy = lats_ds3.to_numpy()
lats_ds4_numpy = lats_ds4.to_numpy()
lats_ds5_numpy = lats_ds5.to_numpy()
lats_ds6_numpy = lats_ds6.to_numpy()

# grab lons
lons_ds1 = ds1a["lon"]
lons_ds2 = ds2a["lon"]
lons_ds3 = ds3a["lon"]
lons_ds4 = ds4a["lon"]
lons_ds5 = ds5a["lon"]
lons_ds6 = ds6a["lon"]

lons_ds1_numpy = lons_ds1.to_numpy()
lons_ds2_numpy = lons_ds2.to_numpy()
lons_ds3_numpy = lons_ds3.to_numpy()
lons_ds4_numpy = lons_ds4.to_numpy()
lons_ds5_numpy = lons_ds5.to_numpy()
lons_ds6_numpy = lons_ds6.to_numpy()

#%% Grab station data   

# Grab station data
station_data = pd.read_csv('station_data/OH_Weather_Stations.csv')

#%% Build a function to find index of nearest 

# find_nearest requires two inputs: target latitude and target longitude
# the function returns (index of tile, lat ri, lat ci, lon ri, lon ci)
# where ri = row index and ci= column index
def find_nearest(target_lat, target_lon): 
  
  
  shape = (2,6)
  indices_lat = np.ones(shape) # initialize indices results
  min_lat = np.ones(6) # initialize min results
  
  # step through lats for each of the six tiles:
  dif = np.absolute(lats_ds1_numpy-target_lat) #
  ri, ci = dif.argmin()//dif.shape[1], dif.argmin()%dif.shape[1] #
  indices_lat[0,0] = ri #
  indices_lat[1,0] = ci #
  min_lat[0] = dif[ri,ci] #
  
  dif = np.absolute(lats_ds2_numpy-target_lat)
  ri, ci = dif.argmin()//dif.shape[1], dif.argmin()%dif.shape[1]
  indices_lat[0,1] = ri
  indices_lat[1,1] = ci
  min_lat[1] = dif[ri,ci]
  
  dif = np.absolute(lats_ds3_numpy-target_lat)
  ri, ci = dif.argmin()//dif.shape[1], dif.argmin()%dif.shape[1]
  indices_lat[0,2] = ri
  indices_lat[1,2] = ci
  min_lat[2] = dif[ri,ci]
  
  dif = np.absolute(lats_ds4_numpy-target_lat)
  ri, ci = dif.argmin()//dif.shape[1], dif.argmin()%dif.shape[1]
  indices_lat[0,3] = ri
  indices_lat[1,3] = ci
  min_lat[3] = dif[ri,ci]
  
  dif = np.absolute(lats_ds5_numpy-target_lat)
  ri, ci = dif.argmin()//dif.shape[1], dif.argmin()%dif.shape[1]
  indices_lat[0,4] = ri
  indices_lat[1,4] = ci
  min_lat[4] = dif[ri,ci]
  
  dif = np.absolute(lats_ds6_numpy-target_lat)
  ri, ci = dif.argmin()//dif.shape[1], dif.argmin()%dif.shape[1]
  indices_lat[0,5] = ri
  indices_lat[1,5] = ci
  min_lat[5] = dif[ri,ci]

  print(indices_lat)
  print(min_lat)
  
  # same for lons
  indices_lon = np.ones(shape) # initialize indices results
  min_lon = np.ones(6) # initialize min results
  
  # step through lats for each of the six tiles:
  dif = np.absolute(lons_ds1_numpy-target_lon)
  ri, ci = dif.argmin()//dif.shape[1], dif.argmin()%dif.shape[1]
  indices_lon[0,0] = ri
  indices_lon[1,0] = ci
  min_lon[0] = dif[ri,ci]
  
  dif = np.absolute(lons_ds2_numpy-target_lon)
  ri, ci = dif.argmin()//dif.shape[1], dif.argmin()%dif.shape[1]
  indices_lon[0,1] = ri
  indices_lon[1,1] = ci
  min_lon[1] = dif[ri,ci]
  
  dif = np.absolute(lons_ds3_numpy-target_lon)
  ri, ci = dif.argmin()//dif.shape[1], dif.argmin()%dif.shape[1]
  indices_lon[0,2] = ri
  indices_lon[1,2] = ci
  min_lon[2] = dif[ri,ci]
  
  dif = np.absolute(lons_ds4_numpy-target_lon)
  ri, ci = dif.argmin()//dif.shape[1], dif.argmin()%dif.shape[1]
  indices_lon[0,3] = ri
  indices_lon[1,3] = ci
  min_lon[3] = dif[ri,ci]
  
  dif = np.absolute(lons_ds5_numpy-target_lon)
  ri, ci = dif.argmin()//dif.shape[1], dif.argmin()%dif.shape[1]
  indices_lon[0,4] = ri
  indices_lon[1,4] = ci
  min_lon[4] = dif[ri,ci]
  
  dif = np.absolute(lons_ds6_numpy-target_lon)
  ri, ci = dif.argmin()//dif.shape[1], dif.argmin()%dif.shape[1]
  indices_lon[0,5] = ri
  indices_lon[1,5] = ci
  min_lon[5] = dif[ri,ci]
  

  print(indices_lon)
  print(min_lon)
  
  dif_lat_lon = min_lat + min_lon
  n = dif_lat_lon.argmin()
  
  my_tile = n
  my_lat_ri = indices_lat[0,n]
  my_lat_ci = indices_lat[1,n]
  my_lon_ri = indices_lon[0,n]
  my_lon_ci = indices_lon[1,n]

  nearest = np.ones(5) # initialize final results
  nearest = (my_tile,my_lat_ri,my_lat_ci,my_lon_ri,my_lon_ci)
  return(nearest)

#%% Use our new function to find indices for our stations 

# Configure dataframe for indexing
# Need columns for:
  # my_tile = n
  # my_lat_ri = indices_lat[0,n] # latitude row index
  # my_lat_ci = indices_lat[1,n] # latitude column index
  # my_lon_ri = indices_lon[0,n] # longitude row index
  # my_lon_ci = indices_lon[1,n] # longitude column index
empty_col = np.ones(len(station_data.axes[0]))*-999
station_data['TileIndex'] = empty_col.copy()
station_data['LatRi'] = empty_col.copy()
station_data['LatCi'] = empty_col.copy()
station_data['LonRi'] = empty_col.copy()
station_data['LonCi'] = empty_col.copy()

n = len(station_data.axes[0]) # number of iterations

# Initialize
TileIndex = empty_col.copy()
LatRi = empty_col.copy()
LatCi = empty_col.copy()
LonRi = empty_col.copy()
LonCi = empty_col.copy()

for x in range(n):
  target_lat = station_data['Latitude'][x]
  target_lon = station_data['Longitude'][x]
  
  nearest = find_nearest(target_lat, target_lon)
  
  TileIndex[x] = nearest[0]
  LatRi[x] = nearest[1]
  LatCi[x] = nearest[2]
  LonRi[x] = nearest[3]
  LonCi[x] = nearest[4]
  
# Assign our answers!
station_data['TileIndex'] = TileIndex.copy()
station_data['LatRi'] = LatRi.copy()
station_data['LatCi'] = LatCi.copy()
station_data['LonRi'] = LonRi.copy()
station_data['LonCi'] = LonCi.copy()
 
 
#%% Let's grab precip data from each station for 1980:
    
# Let's initialize a dataframe:
ones_data = np.ones(shape=(365,len(station_data)))*-999
PrecipData = pd.DataFrame(ones_data, columns=station_data["STID"])

for x in range(len(station_data)):
    # Grab the right station key.
    MyColumn = station_data["STID"][x] 
    
    # First, select the right tile.
    MyTileIndex = station_data["TileIndex"][x]
    if MyTileIndex == 0:
        MyTile = ds1a
        MyLats = lats_ds1_numpy
        MyLons = lons_ds1_numpy
        
    if MyTileIndex == 1:
        MyTile = ds2a    
        MyLats = lats_ds2_numpy
        MyLons = lons_ds2_numpy
      
    if MyTileIndex == 2:
        MyTile = ds3a
        MyLats = lats_ds3_numpy
        MyLons = lons_ds3_numpy
        
    if MyTileIndex == 3:
        MyTile = ds4a 
        MyLats = lats_ds4_numpy
        MyLons = lons_ds4_numpy
    
    if MyTileIndex == 4:
        MyTile = ds5a 
        MyLats = lats_ds5_numpy
        MyLons = lons_ds5_numpy
        
    if MyTileIndex == 5:
        MyTile = ds6a
        MyLats = lats_ds6_numpy
        MyLons = lons_ds6_numpy     
        
    # Second, grab the right indices
    MyLatRi = station_data["LatRi"][x]
    MyLatCi = station_data["LatCi"][x]
    
    # Third, grab the temperature series
    MyPrecip = MyTile["prcp"][:,int(MyLatRi),int(MyLatCi)]
    #MyPrecipNumpy = MyPrecip.to_numpy()
    
    # Assign the results
    PrecipData[MyColumn] = MyPrecip

# Finally, let's drop columns with nans
PrecipDataClean = PrecipData.dropna(axis=1)

# Now, Let's create a correlation matrix for our stations.
f = plt.figure(figsize=(19, 15))
plt.matshow(PrecipDataClean.corr(), fignum=f.number)
plt.xticks(range(PrecipDataClean.select_dtypes(['number']).shape[1]), PrecipDataClean.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(PrecipDataClean.select_dtypes(['number']).shape[1]), PrecipDataClean.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix for daily precipitation, 1980', fontsize=16);

# save the figure
plt.savefig("exports/correlation_matrix.svg")

#%% Let's try building a master data dictionary over all 42 years (1980 - 2021)

# CREATE MASTER DICTIONARIES ####################################################

## Create Data Dictionary
MyData = {11748:{},11749:{},11750:{},11928:{},11929:{},11930:{}} # create empty MyData dictionary
MyYears = list(range(1980,2022,)) # create sequence of years for year loop
MyTiles = [11748,11749,11750,11928,11929,11930]

## Populate data dictionary
for n in range(0,len(MyTiles)): # Loop over tiles
    MyTile = MyTiles[n]
    print('starting tile '+str(MyTile))
    
    for x in range(0,len(MyYears),): # Within each tile, loop over years
        MyYear = MyYears[x]
        MyFile_precip = 'daymet_data/'+ str(MyTile) + '_' + str(MyYear) + '_prcp.nc'
        MyFile_tmin = 'daymet_data/'+ str(MyTile) + '_' + str(MyYear) + '_tmin.nc'
        MyFile_tmax = 'daymet_data/'+ str(MyTile) + '_' + str(MyYear) + '_tmax.nc'
        nc_precip = xr.open_dataset(MyFile_precip) # grab precip
        nc_tmin = xr.open_dataset(MyFile_tmin) # grab tmin
        nc_tmax = xr.open_dataset(MyFile_tmax) # grab tmax
        
        MyData[MyTile][MyYear]={'prcp': nc_precip['prcp'], 'tmin':nc_tmin['tmin'], 'tmax':nc_tmax['tmax']} # Make dictionary!
    
    if n==5:
        print('MISCHIEF MANAGED!')        

#%% Let's try building a master array data structure using arrays over all 42 years (1980 - 2021)

MyYears = list(range(1980,2022,)) # create sequence of years for year loop
MyTiles = [11748,11749,11750,11928,11929,11930]

# I need 3D matrices:
    # each entry in order is x-index, y-index, and the year (or year and day)
    # float is 8 bytes: 8 bytes * 237*196*42*365

# Create our float arrays (x-index, y-index, years, dates) # generalize this later
tile1_prcp = np.zeros((237,196,42,365)) # 11748 (y,x,year,day)
tile2_prcp = np.zeros((239,200,42,365)) # 11749 (y,x,year,day)  
tile3_prcp = np.zeros((243,204,42,365)) # 11750 (y,x,year,day)
tile4_prcp = np.zeros((236,190,42,365)) # 11928 (y,x,year,day)
tile5_prcp = np.zeros((239,195,42,365)) # 11929 (y,x,year,day)
tile6_prcp = np.zeros((242,199,42,365)) # 11930 (y,x,year,day)

MyTileNames = [tile1_prcp, tile2_prcp, tile3_prcp, tile4_prcp, tile5_prcp, tile6_prcp]

## Populate our arrays
start = time.time() # let's time this (Run 1: 160.166 seconds)
for n in range(0,len(MyTiles)): # Loop over tiles
    MyTile = MyTiles[n]
    print('starting tile '+str(MyTile))
    
    for x in range(0,len(MyYears),): # Within each tile, loop over years
        MyYear = MyYears[x]
        print(MyYear)
        MyFile_precip = 'daymet_data/'+ str(MyTile) + '_' + str(MyYear) + '_prcp.nc'
        #MyFile_tmin = 'daymet_data/'+ str(MyTile) + '_' + str(MyYear) + '_tmin.nc'
        #MyFile_tmax = 'daymet_data/'+ str(MyTile) + '_' + str(MyYear) + '_tmax.nc'
        nc_precip = xr.open_dataset(MyFile_precip) # grab precip
        #nc_tmin = xr.open_dataset(MyFile_tmin) # grab tmin
        #nc_tmax = xr.open_dataset(MyFile_tmax) # grab tmax
        
        for i in range(0,365):
            MyTileNames[n][:,:,x,i] = nc_precip['prcp'][i,:,:]
            


        #MyData[MyTile][MyYear]={'prcp': nc_precip['prcp'], 'tmin':nc_tmin['tmin'], 'tmax':nc_tmax['tmax']} # Make dictionary!
    
    if n==5:
        end = time.time()
        print('duration:')
        print(end - start)
        print('MISCHIEF MANAGED!')        
        


#%% Let's grab precip data from each station for all 41 years (1980 - 2021):
## Sum precip across the days of each year for all tiles

# initialize new summed tiles
tile1_prcp_summed = np.zeros((237,196,42)) # 11748 (y,x,year,day)
tile2_prcp_summed = np.zeros((239,200,42)) # 11749 (y,x,year,day)  
tile3_prcp_summed = np.zeros((243,204,42)) # 11750 (y,x,year,day)
tile4_prcp_summed = np.zeros((236,190,42)) # 11928 (y,x,year,day)
tile5_prcp_summed = np.zeros((239,195,42)) # 11929 (y,x,year,day)
tile6_prcp_summed = np.zeros((242,199,42)) # 11930 (y,x,year,day)

MyTileNames_Summed = [tile1_prcp_summed, tile2_prcp_summed, tile3_prcp_summed, tile4_prcp_summed, tile5_prcp_summed, tile6_prcp_summed]

# sum across the days of each year
start = time.time() # let's time this (Run 1: 160.166 seconds)
for i in range(0,len(MyTileNames)):
    MyTileNames_Summed[i][:,:,:] = np.sum(MyTileNames[i],axis=3)

end = time.time()
print('duration:')
print(end - start) # run 1: 4.744 seconds!!!
print('MISCHIEF MANAGED!')  
    
## Grab data for each station

# Let's initialize a dataframe:
ones_data = np.ones(shape=(len(MyYears),len(station_data)))*-999
PrecipData = pd.DataFrame(ones_data, columns=station_data["STID"])
MyYears = list(range(1980,2022,)) # create sequence of years for year loop
PrecipData.index =  MyYears

for n in range(len(station_data)):
    # Grab the right station key.
    MyColumn = station_data["STID"][n] 
    print('Starting '+MyColumn)
    
    # select the right tile.
    #MyTiles = [11748,11749,11750,11928,11929,11930]
    MyTileIndex = station_data["TileIndex"][n]
    if MyTileIndex == 0:
        MyTile = 11748
        
    if MyTileIndex == 1:
        MyTile = 11749    
     
    if MyTileIndex == 2:
        MyTile = 11750
        
    if MyTileIndex == 3:
        MyTile = 11928
    
    if MyTileIndex == 4:
        MyTile = 11929

    if MyTileIndex == 5:
        MyTile = 11930  
        
    # grab the right indices
    MyLatRi = station_data["LatRi"][n]
    MyLatCi = station_data["LatCi"][n]
    
    
    


# Finally, let's drop columns with nans
PrecipDataClean = PrecipData.dropna(axis=1)

# Now, Let's create a correlation matrix for our stations.
f = plt.figure(figsize=(19, 15))
plt.matshow(PrecipDataClean.corr(), fignum=f.number)
plt.xticks(range(PrecipDataClean.select_dtypes(['number']).shape[1]), PrecipDataClean.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(PrecipDataClean.select_dtypes(['number']).shape[1]), PrecipDataClean.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix for yearly precipitation, 1980-2021', fontsize=16);

# save the figure
plt.savefig("exports/correlation_matrix.svg")


    
#%% Try printing a precip stack

ds1a_grab = ds1a['prcp'][364,:,:]
ds2a_grab = ds2a['prcp'][364,:,:]
ds3a_grab = ds3a['prcp'][364,:,:]
ds4a_grab = ds4a['prcp'][364,:,:]
ds5a_grab = ds5a['prcp'][364,:,:]
ds6a_grab = ds6a['prcp'][364,:,:]

ds1a_numpy = ds1a_grab.to_numpy()
ds2a_numpy = ds2a_grab.to_numpy()
ds3a_numpy = ds3a_grab.to_numpy()
ds4a_numpy = ds4a_grab.to_numpy()
ds5a_numpy = ds5a_grab.to_numpy()
ds6a_numpy = ds6a_grab.to_numpy()

# ds1a_numpy = ds1a_numpy[np.isfinite(ds1a_numpy)]
# ds2a_numpy = ds2a_numpy[np.isfinite(ds2a_numpy)]
# ds3a_numpy = ds3a_numpy[np.isfinite(ds3a_numpy)]
# ds4a_numpy = ds4a_numpy[np.isfinite(ds4a_numpy)]
# ds5a_numpy = ds5a_numpy[np.isfinite(ds5a_numpy)]
# ds6a_numpy = ds6a_numpy[np.isfinite(ds6a_numpy)]

# take subsets
# hstack first 3, hstack the second 3 [:236,:190]
ds_stack = np.hstack((ds1a_numpy[:236,:190], ds2a_numpy[:236,:190], ds3a_numpy[:236,:190]))
ds_stack2 = np.hstack((ds4a_numpy[:236,:190], ds5a_numpy[:236,:190], ds6a_numpy[:236,:190]))

# do the vstack for two stacks
ds_stack_both = np.vstack((ds_stack,ds_stack2))

plt.contourf(ds_stack_both)
plt.colorbar()
plt.savefig("exports/prcp_stack.svg")

#%% Let's plot time series of our stations

# FIRST, PRACTICE ON STATION 1
myTileIndex = 2
myLatRi = 117
myLatCi = 102
myLonRi = 240
myLonCi = 89

station_target_prcp = ds1a['prcp'][:,myLatRi,myLatCi]
station_target_prcp_numpy = station_target_prcp.to_numpy()

plt.plot(station_target_prcp_numpy,'.')
plt.title("KAKR Akron Fulton Intl. Airport - Daily Precip for 1980") 
plt.xlabel("Day") 
plt.ylabel("Precipitation [mm]") 
plt.savefig("exports/KAKR_precip1.svg")

station_target_prcp = ds1b['prcp'][:,myLatRi,myLatCi]
station_target_prcp_numpy = station_target_prcp.to_numpy()

plt.plot(station_target_prcp_numpy,'.')
plt.title("KAKR Akron Fulton Intl. Airport - Daily Precip for 1981") 
plt.xlabel("Day") 
plt.ylabel("Precipitation [mm]") 
plt.savefig("exports/KAKR_precip2.svg")

station_target_prcp = ds1c['prcp'][:,myLatRi,myLatCi]
station_target_prcp_numpy = station_target_prcp.to_numpy()

plt.plot(station_target_prcp_numpy,'.')
plt.title("KAKR Akron Fulton Intl. Airport - Daily Precip for 1982") 
plt.xlabel("Day") 
plt.ylabel("Precipitation [mm]") 
plt.savefig("exports/KAKR_precip3.svg")

station_target_prcp = ds1d['prcp'][:,myLatRi,myLatCi]
station_target_prcp_numpy = station_target_prcp.to_numpy()

plt.plot(station_target_prcp_numpy,'.')
plt.title("KAKR Akron Fulton Intl. Airport - Daily Precip for 1983") 
plt.xlabel("Day") 
plt.ylabel("Precipitation [mm]") 
plt.savefig("exports/KAKR_precip4.svg")





#%%    
# A Python dictionary is a key-value pair. You input the key, 
# it gives you a value. 
# I could take the first tile, for non-nan every pixel, find this lat/lon. 
# Lat/lon will be the key for the dictionary, the value will be precipitation. 

# Simple version: dictionary for foods. Maybe the key is the name of food (string).
# Apple would be 'fruit'. {Apple: 'fruit', Broccoli: 'vegetable', Pear: 'fruit'} 
# Squiggly brackets = dictionary named 'Food'
# To add a new dictionary entry: Food['Rice'] = 'grain'

# Convert a tile into a dictionary where I look up lat/lon and get back precip. 
# then I can combine all 6 tiles into one dictionary, so I get one lookup place.
# for every lat/lon that doesn't have nan, PrecipMap[(55,-84)] = 8 #[mm precip]
# tuple (55,-84); string '55,-84'

# Do a dictionary within a dictionary.
# For each lat/lon, you get back a dictionary which contains multiple value pairs.
# ie: PrecipMap is the dictionary. Put in key (55,84), get back a dictionary PRCP.
# 

# each lat lon returns a dictionary where each key is a year
# each year will give you a dictionary where each key is a day
# each day is a dictionary where each key is 'prcp', 'HUC6', 'tempmin', 'tempmax'
# initialize each of these as empty dictionaries, and then add keys as you go:
    
MyMap = {'41,-84':{}} # create empty MyMap dict
MyMap['41,-84'][1980]={} # add key 1980 empty dictionary to MyMap
MyMap['41,-84'][1980][364]={'prcp': 8, 'HUC6':True, 'TempMin':12, 'TempMax':32}


MyMap = {}

#%%
#Plot
plt.contourf(ds1a['prcp'][364,:,:])
#plt.contourf(ds1a_numpy)
#plt.contourf(ds2a_numpy)

#%%
# Read the shapefile
huc4 = gpd.read_file("shapefiles/NHD_H_0410_HU4_Shape/Shape/WBDHU4.shp")
huc4.plot() # plot it
#plt.savefig("exports/shp.svg")



