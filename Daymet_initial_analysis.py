# -*- coding: utf-8 -*-
"""
Initial analysis
Goal: open .nc and .shp files in Python

first create conda environment in miniconda:
conda create -n spyder-geo
conda activate spyder-geo
conda install -c conda-forge spyder-kernels
conda install -c conda-forge spyder
conda install -c conda-forge xarray
conda install -c conda-forge hmmlearn

"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import time
#from hmmlearn.hmm import GaussianHMM


#%% x array - open precip datafiles

# Open our .nc files
# Tile 1
ds1a = xr.open_dataset('daymet_data/11748_1980_prcp.nc')

# Tile 2
ds2a = xr.open_dataset('daymet_data/11749_1980_prcp.nc')

# Tile 3
ds3a = xr.open_dataset('daymet_data/11750_1980_prcp.nc')

# Tile 4
ds4a = xr.open_dataset('daymet_data/11928_1980_prcp.nc')

# Tile 5
ds5a = xr.open_dataset('daymet_data/11929_1980_prcp.nc')

# Tile 6
ds6a = xr.open_dataset('daymet_data/11930_1980_prcp.nc')

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

#%% Grab location data 

stations = False # load station data?
huc12s = True # load huc 12 centroid data?

if stations:
    # Grab station data
    locations = pd.read_csv('station_data/OH_Weather_Stations.csv')

if huc12s:
    # Grab centroid data
    locations = pd.read_csv('centroids/Centroids_HUC12_lite.csv')

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
empty_col = np.ones(len(locations.axes[0]))*-999
locations['TileIndex'] = empty_col.copy()
locations['LatRi'] = empty_col.copy()
locations['LatCi'] = empty_col.copy()
locations['LonRi'] = empty_col.copy()
locations['LonCi'] = empty_col.copy()

n = len(locations.axes[0]) # number of iterations

# Initialize
TileIndex = empty_col.copy()
LatRi = empty_col.copy()
LatCi = empty_col.copy()
LonRi = empty_col.copy()
LonCi = empty_col.copy()

for x in range(n):
  target_lat = locations['Latitude'][x]
  target_lon = locations['Longitude'][x]
  
  nearest = find_nearest(target_lat, target_lon)
  
  TileIndex[x] = nearest[0]
  LatRi[x] = nearest[1]
  LatCi[x] = nearest[2]
  LonRi[x] = nearest[3]
  LonCi[x] = nearest[4]
  
# Assign our answers!
locations['TileIndex'] = TileIndex.copy()
locations['LatRi'] = LatRi.copy()
locations['LatCi'] = LatCi.copy()
locations['LonRi'] = LonRi.copy()
locations['LonCi'] = LonCi.copy()
 
 
#%% Let's try building a master array data structure using arrays over all 42 years (1980 - 2021)

MyYears = list(range(1980,2022,)) # create sequence of years for year loop
MyTiles = [11748,11749,11750,11928,11929,11930]

# I need 3D matrices:
    # each entry in order is x-index, y-index, and the year (or year and day)
    # float is 8 bytes: 8 bytes * 237*196*42*365 ~ 6GB of RAM per array

# Create our float arrays (x-index, y-index, years, dates) # generalize this later
tile1_prcp = np.zeros((237,196,42,365)) # 11748 (y,x,year,day)
tile2_prcp = np.zeros((239,200,42,365)) # 11749 (y,x,year,day)  
tile3_prcp = np.zeros((243,204,42,365)) # 11750 (y,x,year,day)
tile4_prcp = np.zeros((236,190,42,365)) # 11928 (y,x,year,day)
tile5_prcp = np.zeros((239,195,42,365)) # 11929 (y,x,year,day)
tile6_prcp = np.zeros((242,199,42,365)) # 11930 (y,x,year,day)

tile1_tmin = np.zeros((237,196,42,365)) # 11748 (y,x,year,day)
tile2_tmin = np.zeros((239,200,42,365)) # 11749 (y,x,year,day)  
tile3_tmin = np.zeros((243,204,42,365)) # 11750 (y,x,year,day)
tile4_tmin = np.zeros((236,190,42,365)) # 11928 (y,x,year,day)
tile5_tmin = np.zeros((239,195,42,365)) # 11929 (y,x,year,day)
tile6_tmin = np.zeros((242,199,42,365)) # 11930 (y,x,year,day)

tile1_tmax = np.zeros((237,196,42,365)) # 11748 (y,x,year,day)
tile2_tmax = np.zeros((239,200,42,365)) # 11749 (y,x,year,day)  
tile3_tmax = np.zeros((243,204,42,365)) # 11750 (y,x,year,day)
tile4_tmax = np.zeros((236,190,42,365)) # 11928 (y,x,year,day)
tile5_tmax = np.zeros((239,195,42,365)) # 11929 (y,x,year,day)
tile6_tmax = np.zeros((242,199,42,365)) # 11930 (y,x,year,day)

MyTileNames_Prcp = [tile1_prcp, tile2_prcp, tile3_prcp, tile4_prcp, tile5_prcp, tile6_prcp]
MyTileNames_Tmin = [tile1_tmin, tile2_tmin, tile3_tmin, tile4_tmin, tile5_tmin, tile6_tmin]
MyTileNames_Tmax = [tile1_tmax, tile2_tmax, tile3_tmax, tile4_tmax, tile5_tmax, tile6_tmax]

## Populate our arrays
start = time.time() # let's time this
for n in range(0,len(MyTiles)): # Loop over tiles
    MyTile = MyTiles[n]
    print('starting tile '+str(MyTile))
    
    for x in range(0,len(MyYears),): # Within each tile, loop over years
        MyYear = MyYears[x]
        print(MyYear)
        MyFile_precip = 'daymet_data/'+ str(MyTile) + '_' + str(MyYear) + '_prcp.nc'
        MyFile_tmin = 'daymet_data/'+ str(MyTile) + '_' + str(MyYear) + '_tmin.nc'
        MyFile_tmax = 'daymet_data/'+ str(MyTile) + '_' + str(MyYear) + '_tmax.nc'
        nc_precip = xr.open_dataset(MyFile_precip) # grab precip
        nc_tmin = xr.open_dataset(MyFile_tmin) # grab tmin
        nc_tmax = xr.open_dataset(MyFile_tmax) # grab tmax
        
        for i in range(0,365):
            MyTileNames_Prcp[n][:,:,x,i] = nc_precip['prcp'][i,:,:]
            MyTileNames_Tmin[n][:,:,x,i] = nc_tmin['tmin'][i,:,:]
            MyTileNames_Tmax[n][:,:,x,i] = nc_tmax['tmax'][i,:,:]
      
# Let's save our numpy arrays
np.save('tile1_prcp.npy', tile1_prcp)
np.save('tile2_prcp.npy', tile2_prcp)
np.save('tile3_prcp.npy', tile3_prcp)
np.save('tile4_prcp.npy', tile4_prcp)
np.save('tile5_prcp.npy', tile5_prcp)
np.save('tile6_prcp.npy', tile6_prcp)

np.save('tile1_tmin.npy', tile1_tmin)
np.save('tile2_tmin.npy', tile2_tmin)
np.save('tile3_tmin.npy', tile3_tmin)
np.save('tile4_tmin.npy', tile4_tmin)
np.save('tile5_tmin.npy', tile5_tmin)
np.save('tile6_tmin.npy', tile6_tmin)

np.save('tile1_tmax.npy', tile1_tmax)
np.save('tile2_tmax.npy', tile2_tmax)
np.save('tile3_tmax.npy', tile3_tmax)
np.save('tile4_tmax.npy', tile4_tmax)
np.save('tile5_tmax.npy', tile5_tmax)
np.save('tile6_tmax.npy', tile6_tmax)

end = time.time()
print('duration:')
print(end - start)
print('MISCHIEF MANAGED!') 

#%% Load our arrays
start = time.time() # let's time this # Run 1: 1.40 mins # Run 2: 1.11 mins

tile1_prcp = np.load('tile1_prcp.npy')
tile2_prcp = np.load('tile2_prcp.npy')
tile3_prcp = np.load('tile3_prcp.npy')
tile4_prcp = np.load('tile4_prcp.npy')
tile5_prcp = np.load('tile5_prcp.npy')
tile6_prcp = np.load('tile6_prcp.npy')

tile1_tmin = np.load('tile1_tmin.npy')
tile2_tmin = np.load('tile2_tmin.npy')
tile3_tmin = np.load('tile3_tmin.npy')
tile4_tmin = np.load('tile4_tmin.npy')
tile5_tmin = np.load('tile5_tmin.npy')
tile6_tmin = np.load('tile6_tmin.npy')

tile1_tmax = np.load('tile1_tmax.npy')
tile2_tmax = np.load('tile2_tmax.npy')
tile3_tmax = np.load('tile3_tmax.npy')
tile4_tmax = np.load('tile4_tmax.npy')
tile5_tmax = np.load('tile5_tmax.npy')
tile6_tmax = np.load('tile6_tmax.npy')

end = time.time()
print('duration:')
print(end - start)
print('MISCHIEF MANAGED!') 

#%% Let's grab precip data from one centroid for all 41 years (1980 - 2021)
start = time.time()

# we'll choose the portage river, HUC12 41000100502
MyTileNames_Prcp = [tile1_prcp, tile2_prcp, tile3_prcp, tile4_prcp, tile5_prcp, tile6_prcp]
MyTileXs         = [ds1a.x, ds2a.x, ds3a.x, ds4a.x, ds5a.x, ds6a.x]
MyTileYs         = [ds1a.y, ds2a.y, ds3a.y, ds4a.y, ds5a.y, ds6a.y]
MyTileLats       = [lats_ds1, lats_ds2, lats_ds3, lats_ds4, lats_ds5, lats_ds6]
MyTileLons       = [lons_ds1, lons_ds2, lons_ds3, lons_ds4, lons_ds5, lons_ds6]
 
mylocation = locations.loc[locations['Code']=='Portage River'] # portage river, HUC12 41000100502
locationindex = 252 # for portage river
mylocationname = locations.iloc[locationindex,1]

## Grab precip data for each station

# Let's initialize a dataframe:
MyYears = list(range(1980,2022,)) # create sequence of years for year loop
ones_data = np.ones(shape=(len(MyYears)*365,1))*-999
PrecipData = pd.DataFrame(ones_data)
PrecipData.columns = [mylocationname]


# select the right tile.
#MyTiles = [11748,11749,11750,11928,11929,11930]
MyTileIndex = int(mylocation["TileIndex"][locationindex])

# grab the right indices
MyLatRi = int(mylocation["LatRi"][locationindex])
MyLatCi = int(mylocation["LatCi"][locationindex])
MyLonRi = int(mylocation["LonRi"][locationindex])
MyLonCi = int(mylocation["LonCi"][locationindex])

MyLat = MyTileLats[MyTileIndex][MyLatRi,MyLatCi]
MyLon = MyTileLons[MyTileIndex][MyLonRi,MyLonCi]

MyLatY = float(MyLat.y)
MyLonX = float(MyLon.x)

MyIndexX = int(np.where(MyTileXs[MyTileIndex] == MyLonX)[0])
MyIndexY = int(np.where(MyTileYs[MyTileIndex] == MyLatY)[0])


# grab the data

for i in range(len(MyYears)):
    print('Starting '+str(MyYears[i]))
    n=0+i*365 # start at the beginning of our time series
    m=n+365 # end at the 365th day
    PrecipData['Portage River'][n:m] = MyTileNames_Prcp[MyTileIndex][MyIndexY,MyIndexX,i]
    
# plot a time series
plt.title("Daily Precipitation for Portage River HUC12 1980-2021 ")
plt.plot(PrecipData.index, PrecipData['Portage River'],'.')    
plt.xlabel("Day")
plt.ylabel("Precipiation [mm]")

plt.savefig("exports/Portage_River_Precip.svg")
    
end = time.time()
print('duration:')
print(end - start)
print('MISCHIEF MANAGED!')     

#%% Let's grab precip data from each location for all 41 years (1980 - 2021)

start = time.time() # let's time this (Run 1: 10.514518022537231 seconds)

## Sum precip across the days of each year for all tiles
# initialize new summed tiles
tile1_prcp_summed = np.zeros((237,196,42)) # 11748 (y,x,year,day)
tile2_prcp_summed = np.zeros((239,200,42)) # 11749 (y,x,year,day)  
tile3_prcp_summed = np.zeros((243,204,42)) # 11750 (y,x,year,day)
tile4_prcp_summed = np.zeros((236,190,42)) # 11928 (y,x,year,day)
tile5_prcp_summed = np.zeros((239,195,42)) # 11929 (y,x,year,day)
tile6_prcp_summed = np.zeros((242,199,42)) # 11930 (y,x,year,day)


MyYears = list(range(1980,2022,)) # create sequence of years for year loop
MyTiles = [11748,11749,11750,11928,11929,11930]

MyTileNames_Prcp = [tile1_prcp, tile2_prcp, tile3_prcp, tile4_prcp, tile5_prcp, tile6_prcp]
MyTileNames_Tmin = [tile1_tmin, tile2_tmin, tile3_tmin, tile4_tmin, tile5_tmin, tile6_tmin]
MyTileNames_Tmax = [tile1_tmax, tile2_tmax, tile3_tmax, tile4_tmax, tile5_tmax, tile6_tmax]
MyTileNames_Prcp_Summed = [tile1_prcp_summed, tile2_prcp_summed, tile3_prcp_summed, tile4_prcp_summed, tile5_prcp_summed, tile6_prcp_summed]

MyTileXs         = [ds1a.x, ds2a.x, ds3a.x, ds4a.x, ds5a.x, ds6a.x]
MyTileYs         = [ds1a.y, ds2a.y, ds3a.y, ds4a.y, ds5a.y, ds6a.y]
MyTileLats       = [lats_ds1, lats_ds2, lats_ds3, lats_ds4, lats_ds5, lats_ds6]
MyTileLons       = [lons_ds1, lons_ds2, lons_ds3, lons_ds4, lons_ds5, lons_ds6]

# sum across the days of each year

for i in range(0,len(MyTileNames_Prcp)):
    MyTileNames_Prcp_Summed[i][:,:,:] = np.sum(MyTileNames_Prcp[i],axis=3)

## Grab data for each station

# Let's initialize a dataframe:
ones_data = np.ones(shape=(len(MyYears),len(locations)))*-999
PrecipData = pd.DataFrame(ones_data, columns=locations["Code"])
MyYears = list(range(1980,2022,)) # create sequence of years for year loop

for n in range(len(locations)):
    # Grab the right station key.
    MyColumn = locations["Code"][n] 
    print('Starting '+MyColumn)
    
    # select the right tile.
    #MyTiles = [11748,11749,11750,11928,11929,11930]
    MyTileIndex = int(locations["TileIndex"][n] )
    
    # grab the right indices
    MyLatRi = int(locations["LatRi"][n])
    MyLatCi = int(locations["LatCi"][n])
    MyLonRi = int(locations["LonRi"][n])
    MyLonCi = int(locations["LonCi"][n])

    MyLat = MyTileLats[MyTileIndex][MyLatRi,MyLatCi]
    MyLon = MyTileLons[MyTileIndex][MyLonRi,MyLonCi]

    MyLatY = float(MyLat.y)
    MyLonX = float(MyLon.x)

    MyIndexX = int(np.where(MyTileXs[MyTileIndex] == MyLonX)[0])
    MyIndexY = int(np.where(MyTileYs[MyTileIndex] == MyLatY)[0])
    # grab the summed data
    PrecipData[MyColumn] = MyTileNames_Prcp_Summed[int(MyTileIndex)][MyIndexY,MyIndexX,:]

# Finally, let's drop columns with nans
PrecipDataClean = PrecipData.dropna(axis=1)
    
end = time.time()
print('duration:')
print(end - start)
print('MISCHIEF MANAGED!')  

#%%  Now, Let's create a precip correlation matrix for our stations.
f = plt.figure(figsize=(19, 15))
plt.matshow(PrecipDataClean.corr(), fignum=f.number)
plt.xticks(range(PrecipDataClean.select_dtypes(['number']).shape[1]), PrecipDataClean.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(PrecipDataClean.select_dtypes(['number']).shape[1]), PrecipDataClean.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix for yearly precipitation, 1980-2021', fontsize=16);

# save the figure
plt.savefig("exports/correlation_matrix.svg")
