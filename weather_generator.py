# -*- coding: utf-8 -*-
"""
Initial analysis
Goal: open .nc and .shp files in Python

first create conda environment in miniconda:
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
from hmmlearn.hmm import GaussianHMM
import seaborn as sns
from scipy import stats as ss

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
# If you've already built this, skip to the next section!    
start = time.time() # let's time this

build = True # build a master array data structure? (could take 20-30 minutes or more)

if (build):   
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
    np.save('load_data/tile1_prcp.npy', tile1_prcp)
    np.save('load_data/tile2_prcp.npy', tile2_prcp)
    np.save('load_data/tile3_prcp.npy', tile3_prcp)
    np.save('load_data/tile4_prcp.npy', tile4_prcp)
    np.save('load_data/tile5_prcp.npy', tile5_prcp)
    np.save('load_data/tile6_prcp.npy', tile6_prcp)
    
    np.save('load_data/tile1_tmin.npy', tile1_tmin)
    np.save('load_data/tile2_tmin.npy', tile2_tmin)
    np.save('load_data/tile3_tmin.npy', tile3_tmin)
    np.save('load_data/tile4_tmin.npy', tile4_tmin)
    np.save('load_data/tile5_tmin.npy', tile5_tmin)
    np.save('load_data/tile6_tmin.npy', tile6_tmin)
    
    np.save('load_data/tile1_tmax.npy', tile1_tmax)
    np.save('load_data/tile2_tmax.npy', tile2_tmax)
    np.save('load_data/tile3_tmax.npy', tile3_tmax)
    np.save('load_data/tile4_tmax.npy', tile4_tmax)
    np.save('load_data/tile5_tmax.npy', tile5_tmax)
    np.save('load_data/tile6_tmax.npy', tile6_tmax)
    
end = time.time()
print('duration:')
print(end - start)
print('MISCHIEF MANAGED!') 

#%% Load our arrays, useful if you've already built and saved them in the previous section!
start = time.time() # let's time this # Run 1: 1.40 mins # Run 2: 1.11 mins

load = False # load in our master data arrays? (should take 1-2 minutes)
if (load):

    
    tile1_prcp = np.load('load_data/tile1_prcp.npy')
    tile2_prcp = np.load('load_data/tile2_prcp.npy')
    tile3_prcp = np.load('load_data/tile3_prcp.npy')
    tile4_prcp = np.load('load_data/tile4_prcp.npy')
    tile5_prcp = np.load('load_data/tile5_prcp.npy')
    tile6_prcp = np.load('load_data/tile6_prcp.npy')
    
    tile1_tmin = np.load('load_data/tile1_tmin.npy')
    tile2_tmin = np.load('load_data/tile2_tmin.npy')
    tile3_tmin = np.load('load_data/tile3_tmin.npy')
    tile4_tmin = np.load('load_data/tile4_tmin.npy')
    tile5_tmin = np.load('load_data/tile5_tmin.npy')
    tile6_tmin = np.load('load_data/tile6_tmin.npy')
    
    tile1_tmax = np.load('load_data/tile1_tmax.npy')
    tile2_tmax = np.load('load_data/tile2_tmax.npy')
    tile3_tmax = np.load('load_data/tile3_tmax.npy')
    tile4_tmax = np.load('load_data/tile4_tmax.npy')
    tile5_tmax = np.load('load_data/tile5_tmax.npy')
    tile6_tmax = np.load('load_data/tile6_tmax.npy')

end = time.time()
print('duration:')
print(end - start)
print('MISCHIEF MANAGED!') 

#%% Let's grab precip data from all centroids for all 42 years (1980 - 2021)
start = time.time() # Takes about 40 seconds

# Set up my tiles
MyTileNames_Prcp = [tile1_prcp, tile2_prcp, tile3_prcp, tile4_prcp, tile5_prcp, tile6_prcp]
MyTileXs         = [ds1a.x, ds2a.x, ds3a.x, ds4a.x, ds5a.x, ds6a.x]
MyTileYs         = [ds1a.y, ds2a.y, ds3a.y, ds4a.y, ds5a.y, ds6a.y]
MyTileLats       = [lats_ds1, lats_ds2, lats_ds3, lats_ds4, lats_ds5, lats_ds6]
MyTileLons       = [lons_ds1, lons_ds2, lons_ds3, lons_ds4, lons_ds5, lons_ds6]
 
# Let's initialize a dataframe:
MyYears = list(range(1980,2022,)) # create sequence of years for year loop
ones_data = np.ones(shape=(len(MyYears)*365,len(locations)))*-999
PrecipData = pd.DataFrame(ones_data, columns=locations["HUC12"]) # each column is a HUC12

for x in range(len(locations)):
    
    MyTileIndex = int(locations["TileIndex"][x])
    MyHUC = locations["HUC12"][x]
    
    print('//////////Starting '+str(MyHUC)+'//////////')
    
    # grab the right indices
    MyLatRi = int(locations["LatRi"][x])
    MyLatCi = int(locations["LatCi"][x])
    MyLonRi = int(locations["LonRi"][x])
    MyLonCi = int(locations["LonCi"][x])
    
    MyLat = MyTileLats[MyTileIndex][MyLatRi,MyLatCi]
    MyLon = MyTileLons[MyTileIndex][MyLonRi,MyLonCi]
    
    MyLatY = float(MyLat.y)
    MyLonX = float(MyLon.x)
    
    MyIndexX = np.where(MyTileXs[MyTileIndex] == MyLonX)[0]
    MyIndexY = np.where(MyTileYs[MyTileIndex] == MyLatY)[0]
    
    # grab the data
    
    
    
    if (np.isnan(np.sum(MyTileNames_Prcp[MyTileIndex][MyIndexY,MyIndexX,]))):
        PrecipData[MyHUC] = np.nan
    
    else:
        for i in range(len(MyYears)):
            print('Starting '+str(MyYears[i]))
            n=0+i*365 # start at the beginning of our time series
            m=n+365 # end at the 365th day
            PrecipData[MyHUC][n:m] = np.reshape(MyTileNames_Prcp[MyTileIndex][MyIndexY,MyIndexX,i], -1)
            

# Finally, let's drop columns with nans
PrecipDataClean = PrecipData.dropna(axis=1)

#convert zeros to 1e-12 in case we want to work in logspace (daily) later
PrecipDataClean = PrecipDataClean.replace(to_replace=0, value=1e-12)

# Convert to monthly
ones_data = np.ones(shape=(len(MyYears)*12,len(PrecipDataClean.columns)))*-999
PrecipDataMonthly = pd.DataFrame(ones_data)
PrecipDataMonthly.columns = [PrecipDataClean.columns] # each column is a HUC12

for x in range(len(PrecipDataClean.columns)):
    MyHUC = PrecipDataClean.columns[x]
    print('//////////Starting '+str(MyHUC)+'//////////')
    

    for i in range(len(MyYears)): # for each year, calculate monthly values
        n=0+i*365 # start at the beginning of our time series
        print('Starting '+str(MyYears[i]))
        
        for k in range(12): # for each month, add up daily values 
            if k==0: # january
                monthstart = int(n) # jan starts on the 1st day of the year, or the 0th day
                PrecipDataMonthly.loc[k+i*12,MyHUC]=sum(PrecipDataClean[MyHUC][monthstart:monthstart+31]) # jan has 31 days
            
            if k==1: # february
                monthstart = n+31 # feb starts on the 32nd day of the year, or the 31st python day
                PrecipDataMonthly.loc[k+i*12,MyHUC]=sum(PrecipDataClean[MyHUC][monthstart:monthstart+28]) # feb has 28 days
                
            if k==2: # march
                monthstart = n+59 # mar starts on the 60th day of the year, or the 59th python day
                PrecipDataMonthly.loc[k+i*12,MyHUC]=sum(PrecipDataClean[MyHUC][monthstart:monthstart+31]) # mar has 31 days
                
            if k==3: # april
                monthstart = n+90 # apr starts on the 91st day of the year, or the 90th python day
                PrecipDataMonthly.loc[k+i*12,MyHUC]=sum(PrecipDataClean[MyHUC][monthstart:monthstart+30]) # apr has 30 days   
                
            if k==4: # may
                monthstart = n+120 # may starts on the 121st day of the year, or the 120th python day
                PrecipDataMonthly.loc[k+i*12,MyHUC]=sum(PrecipDataClean[MyHUC][monthstart:monthstart+31]) # may has 31 days    
    
            if k==5: # june
                monthstart = n+151 # jun starts on the 152nd day of the year, or the 151st python day
                PrecipDataMonthly.loc[k+i*12,MyHUC]=sum(PrecipDataClean[MyHUC][monthstart:monthstart+30]) # jun has 30 days     
                
            if k==6: # july
                monthstart = n+181 # jul starts on the 182nd day of the year, or the 181st python day
                PrecipDataMonthly.loc[k+i*12,MyHUC]=sum(PrecipDataClean[MyHUC][monthstart:monthstart+31]) # jul has 31 days 
    
            if k==7: # august
                monthstart = n+212 # aug starts on the 213th day of the year, or the 212th python day
                PrecipDataMonthly.loc[k+i*12,MyHUC]=sum(PrecipDataClean[MyHUC][monthstart:monthstart+31]) # aug has 31 days 
    
            if k==8: # september
                monthstart = n+243 # sep starts on the 244th day of the year, or the 243rd python day
                PrecipDataMonthly.loc[k+i*12,MyHUC]=sum(PrecipDataClean[MyHUC][monthstart:monthstart+30]) # sep has 30 days             
     
            if k==9: # october
                monthstart = n+273 # oct starts on the 274th day of the year, or the 273rd python day
                PrecipDataMonthly.loc[k+i*12,MyHUC]=sum(PrecipDataClean[MyHUC][monthstart:monthstart+31]) # oct has 31 days 
    
            if k==10: # november
                monthstart = n+304 # nov starts on the 305th day of the year, or the 304th python day
                PrecipDataMonthly.loc[k+i*12,MyHUC]=sum(PrecipDataClean[MyHUC][monthstart:monthstart+30]) # oct has 30 days 
                
            if k==11: # december
                monthstart = n+334 # dec starts on the 335th day of the year, or the 334th python day
                PrecipDataMonthly.loc[k+i*12,MyHUC]=sum(PrecipDataClean[MyHUC][monthstart:monthstart+31]) # dec has 31 days
                   
# Export the files
PrecipDataMonthly.to_csv("centroids/Centroids_PrecipDataMonthly.csv", index=False)
    
end = time.time()
print('duration:')
print(end - start)
print('MISCHIEF MANAGED!')

#%% Let's build a weather generator for all of our centroids
start = time.time() # takes about 15 seconds

# Functions first
def fitHMM(Q, nSamples):
    # fit Gaussian HMM to Q
    model = GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(Q,[len(Q),1]))
     
    # classify each observation as state 0 or 1
    hidden_states = model.predict(np.reshape(Q,[len(Q),1]))
 
    # find parameters of Gaussian HMM
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]),np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)
 
    # find log-likelihood of Gaussian HMM
    logProb = model.score(np.reshape(Q,[len(Q),1]))
 
    # generate nSamples from Gaussian HMM
    samples = model.sample(nSamples)
 
    # re-organize mus, sigmas and P so that first row is lower mean (if not already)
    if mus[0] > mus[1]:
        mus = np.flipud(mus)
        sigmas = np.flipud(sigmas)
        P = np.fliplr(np.flipud(P))
        hidden_states = 1 - hidden_states
 
    return hidden_states, mus, sigmas, P, logProb, samples

#Plot hidden states
def plotTimeSeries(Q, hidden_states, ylabel, filename):
 
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)
 
    flag=True
    if (flag):  
        xs = np.arange(len(Q))+1909
        masks = hidden_states == 0
        ax.scatter(xs[masks], Q[masks], c='r', label='Dry State')
        masks = hidden_states == 1
        ax.scatter(xs[masks], Q[masks], c='b', label='Wet State')
        #ax.plot(xs, Q, c='k')
         
        ax.set_xlabel('Month')
        ax.set_ylabel(ylabel)
        plt.title("Hidden States: Monthly Precip for Portage River 1980-2021 ")
        fig.subplots_adjust(bottom=0.2)
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
        fig.savefig(filename)
        fig.clf()
    
    else:
        xs = np.arange(len(Q))+1909
        masks = hidden_states == 0
        ax.scatter(xs[masks], Q[masks], c='r', label='Dry State')
        masks = hidden_states == 1
        ax.scatter(xs[masks], Q[masks], c='b', label='Wet State')
        ax.plot(xs, Q, c='k')
         
        ax.set_xlabel('Day')
        ax.set_ylabel(ylabel)
        plt.title("Hidden States: Daily Precip for Portage River 1980-2021 ")
        fig.subplots_adjust(bottom=0.2)
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
        fig.savefig(filename)
        fig.clf()
 
    return None


# Initialize some dataframes
# mus
ones_data = np.ones(shape=(len(PrecipDataClean.columns),2))*-999
WeatherMus = pd.DataFrame(ones_data)
WeatherMus.columns = ['mu1','mu2']

# sigmas
ones_data = np.ones(shape=(len(PrecipDataClean.columns),2))*-999
WeatherSigmas = pd.DataFrame(ones_data)
WeatherSigmas.columns = ['sigma1','sigma2']

# Ps
ones_data = np.ones(shape=(len(PrecipDataClean.columns),4))*-999
WeatherPs = pd.DataFrame(ones_data)
WeatherPs.columns = ['P1','P2','P3','P4']

# pi
ones_data = np.ones(shape=(len(PrecipDataClean.columns),1))*-999
WeatherPi = pd.DataFrame(ones_data)
WeatherPi.columns = ['pi']

# Load 42 years of daily precip data for all centroids
data = pd.read_csv('centroids/Centroids_PrecipDataMonthly.csv')

for x in range(len(PrecipDataClean.columns)):
    MyHUC = PrecipDataClean.columns[x]
    print('//////////Starting '+str(MyHUC)+'//////////')
    
    Q = data[str(MyHUC)]

    # log transform the data and fit the HMM
    logQ = np.log10(Q)
    hidden_states, mus, sigmas, P, logProb, samples = fitHMM(logQ, 100)
     
    # calculate stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(P))
    one_eigval = np.argmin(np.abs(eigenvals-1))
    pi = eigenvecs[:,one_eigval] / np.sum(eigenvecs[:,one_eigval])
 
    # save variables
    WeatherMus['mu1'][x]=mus[0]
    WeatherMus['mu2'][x]=mus[1]
    WeatherSigmas['sigma1'][x] = sigmas[0]
    WeatherSigmas['sigma2'][x] = sigmas[1]
    WeatherPs['P1'][x]=P[0,0]
    WeatherPs['P2'][x]=P[0,1]
    WeatherPs['P3'][x]=P[1,0]
    WeatherPs['P4'][x]=P[1,1]
    WeatherPi['pi'][x]=pi[0]
    
end = time.time()
print('duration:')
print(end - start)
print('MISCHIEF MANAGED!')
#%% Let's generate some monthly weather for all of our centroids

start = time.time() # takes about 35 seconds

# create a dataframe
ones_data = np.ones(shape=(12*len(MyYears),len(PrecipDataClean.columns)))*-999  # 1 for now
GeneratedWeather = pd.DataFrame(ones_data)
GeneratedWeather.columns = [PrecipDataClean.columns] # each column is a HUC12

wet=0 #initialize our state variable
options = ['dry','wet'] # initialize our state options

for x in range(len(PrecipDataClean.columns)):
    MyHUC = PrecipDataClean.columns[x]
    print('//////////Starting '+str(MyHUC)+'//////////')
    
    # pull variables   
    pi = WeatherPi['pi'][x]
    mus = [WeatherMus['mu1'][x],WeatherMus['mu2'][x]]
    sigmas = [WeatherSigmas['sigma1'][x], WeatherSigmas['sigma2'][x]]

    for i in range(len(GeneratedWeather)):
        
        if i==0: # start with month 0
            
            #determine dry or wet        
            draw = np.random.choice(options, 1, p=(pi,1-pi)) # sample from `options` 1 time according to `pi`
            
            # assign state variable
            if draw=='wet': 
                wet=True
            if draw!='wet':
                wet=False        
           
        if i!=0: # if not month 0, we transition
        
            if wet==False:    
               draw = np.random.choice(options, 1, p=[WeatherPs['P1'][x], WeatherPs['P2'][x]]) # sample from `options` 1 time according to P first row
           
            if wet==True:
               draw = np.random.choice(options, 1, p=[WeatherPs['P3'][x], WeatherPs['P4'][x]]) # sample from `options` 1 time according to P second row
        
            # assign state variable    
            if draw=='wet': 
                wet=True
            if draw!='wet':
                wet=False 
               
        # now get rain values
        if wet==False:    
            log_rain = float(np.random.normal(loc=mus[0], scale=sigmas[0],size=1)) 
            GeneratedWeather.loc[i,MyHUC]=10**(log_rain)
            
        if wet==True: # if wet, we draw from our distribution
            log_rain = float(np.random.normal(loc=mus[1], scale=sigmas[1],size=1)) 
            GeneratedWeather.loc[i,MyHUC]=10**(log_rain)
        

# Export the files
GeneratedWeather.to_csv("exports/Centroids_GeneratedWeather.csv", index=False)

end = time.time()
print('duration:')
print(end - start)
print('MISCHIEF MANAGED!')         