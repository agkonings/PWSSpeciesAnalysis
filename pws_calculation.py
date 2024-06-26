# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 21:17:42 2022

@author: kkrao
"""


import os
import datetime
import sys
import pickle

from osgeo import gdal, osr
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt



#%% helper functions
def get_dates(date, maxLag = 4):
    subsetDates = []
    for delta in range(1, maxLag):
        shiftedDate = pd.to_datetime(date,format = "%Y-%m-%d") - DateOffset(months = delta)
        shiftedDate = shiftedDate.date().strftime(format = "%Y-%m-%d")                    
        for day in [1,15]:
            subsetDates+= [shiftedDate[:-2]+"%02d"%day]
    
    subsetDates = pd.to_datetime(subsetDates,format = "%Y-%m-%d")
    subsetDates = list(subsetDates.sort_values(ascending = False).strftime(date_format = "%Y-%m-%d") )
    
    return subsetDates   

def create_time_df(dir_data, dates, maxLag = 4, hr = "100hr", folder = "lfmc_dfmc_anomalies"):
    date = "2016-01-01"
    filename = os.path.join(dir_data, folder,"lfmc_map_%s.tif"%date)
    ds = gdal.Open(filename)
    array = np.array(ds.GetRasterBand(1).ReadAsArray())
    
    x_loc, y_loc = np.meshgrid(range(array.shape[1]),range(array.shape[0]) )
    
    master = pd.DataFrame()
    for date in dates:
        df = pd.DataFrame()
        filename = os.path.join(dir_data, folder,"lfmc_map_%s.tif"%date)
        ds = gdal.Open(filename)
        lfmc = np.array(ds.GetRasterBand(1).ReadAsArray())
        dfmc = np.array(ds.GetRasterBand(dfmcDict[hr]).ReadAsArray())
        
        df['lfmc(t)'] = lfmc.flatten()
        df['dfmc(t)'] = dfmc.flatten()
        df['x_loc'] = x_loc.flatten()
        df['y_loc'] = y_loc.flatten()
        df['pixel_index'] = df.index
        
        df['date'] = date
        ctr = 1
        sys.stdout.write('\r'+'[INFO] Time step %s'%date)
        sys.stdout.flush()
        # print(date)
        subsetDates = get_dates(date, maxLag = maxLag)
        for t in subsetDates:
            shiftedFile = os.path.join(dir_data,folder,"lfmc_map_%s.tif"%t)
            ds = gdal.Open(shiftedFile)
            df['dfmc(t-%d)'%ctr] = np.array(ds.GetRasterBand(dfmcDict[hr]).ReadAsArray()).flatten()
            ctr+=1
        df.dropna(inplace = True)
        master = master.append(df,ignore_index = True) 
    master = master.dropna()
    return master
    

def regress(df,norm = "lfmc_dfmc_norm", coefs_type = "unrestricted"):            
    cols = [col for col in df.columns if "dfmc" in col]        
    X = df.loc[:,cols]
    y = df.iloc[:,0] ### 
    if norm=="lfmc_norm":
        y = (y-y.mean())/y.std()
    elif norm=="dfmc_norm":
        X = (X - X.mean())/X.std()
    elif norm == "lfmc_dfmc_norm":
        y = (y-y.mean())/y.std()
        X = (X - X.mean())/X.std()
    
    if coefs_type=="positive":
        reg = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
                positive=True, random_state=9999, selection='random').fit(X,y)
    else:
        reg = LinearRegression().fit(X, y)
    r2 = reg.score(X, y)
    coefs = [reg.intercept_]+list(reg.coef_)
    
    return r2, coefs, df['x_loc'].iloc[0],df['y_loc'].iloc[0]  


def save_tif(data, geotransform, savepath = None):
    
    nrows, ncols = data.shape
    
    output_raster = gdal.GetDriverByName('GTiff').Create(savepath,ncols, nrows, 1 ,gdal.GDT_Float32)  # Open the file
    output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
    srs = osr.SpatialReference()                 # Establish its coordinate encoding
    srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
                                                # Anyone know how to specify the 
                                                 # IAU2000:49900 Mars encoding?
    output_raster.SetProjection(srs.ExportToWkt() )   # Exports the coordinate system 
                                                       # to the file
    output_raster.GetRasterBand(1).WriteArray(data)   # Writes my array to the raster
    
    output_raster.FlushCache() 
    output_raster = None 
    print("output_saved")

#%% Create data frame for calculating PWS

dir_data = "G:/My Drive/0000WorkComputer/dataStanford/" # Where is data located?
hr = "100hr" #what hr DFMC data do you want to calculate PWS with?
folder = "lfmc_dfmc_anomalies" #where are the maps of LFMC-DFMC located within dir_data?
lag = 4  #what is the max lag to be considered between LFMC and DFMC in 
# number of months including the current month?
norm = "lfmc_dfmc_norm" #how to normalize the LFMC and DFMC anomalies?
#see regress() function for more options
coefs_type = "positive" #Are there any constraints on the PWS regression coefficients?

years = range(2016, 2022) #[inclusive, exclusive)
months = range(1,13) #months = list(range(12,13)) + list(range(1,6))
days = [1,15]

dates = []
for year in years:
    for month in months:
        if year == 2016 and month <(lag+1): # first 6 months ignored because DFMC is lagged by 5 months
            continue
        if year == 2022 and month >=(lag+1): #do exactly even number of 5 years
            continue
        for day in days:
            dates+=["%s-%02d-%02d"%(year, month, day)]
dfmcDict = {"100hr":2, "1000hr":3} #mapping dfmc hour to band location    
master = create_time_df(hr = hr, folder = folder, maxLag = lag, dates = dates, dir_data = dir_data)

#Good idea to save the data frame because it takes a lot of time to make again
# master.to_pickle(os.path.join(dir_data, f"pws_input_data_{str(datetime.date.today())}.pickle"))
# master = pd.read_pickle(os.path.join(dir_data, f"pws_input_data_{str(datetime.date.today())}.pickle"))
master.date = pd.to_datetime(master.date)
# Possibly apply seasonal filter (e.g. wildfire season only)
master = master.loc[(master.date.dt.month>=1) & (master.date.dt.month<=12)]
# Remove pixels which have less than 75 data points
master = master.groupby("pixel_index").filter(lambda df: df.shape[0] > 75 )

#%% Calculate PWS
# print('\r')
print('[INFO] Regressing')
out = master.groupby('pixel_index').apply(regress,norm = norm, coefs_type = coefs_type)

#Good idea to save the coefficients because it takes a lot of time to make again
# out.to_pickle(os.path.join(dir_data, f"pws_coefficients_{str(datetime.date.today())}.pickle"))
# out = pd.read_pickle(os.path.join(dir_data, f"pws_coefficients_{str(datetime.date.today())}.pickle"))
pwsR2 = [x[0] for x in out]
coefs = [x[1] for x in out]
x_loc = [x[2] for x in out]
y_loc = [x[3] for x in out]
coefSum = [np.sum(x[1:]) for x in coefs]


#%% Plot PWS
filename = os.path.join("G:/My Drive/0000WorkComputer/dataStanford/PWSCalc/PWS_6_jan_2021.tif") #load an old PWS file. 
# This can be any old file at 4 km resolution of the western US to write the 
# new pws information on the matrix of required size.
# If you don't have such a map use this: 
# https://github.com/kkraoj/wildfire_from_lfmc/tree/master/maps/PWS_6_jan_2021.tif

ds = gdal.Open(filename)
geotransform = ds.GetGeoTransform()
pws = np.array(ds.GetRasterBand(1).ReadAsArray())
pws[:,:] = np.nan
pws[y_loc, x_loc] = coefSum

fig, ax = plt.subplots()
ax.imshow(pws)

#%% Save PWS map
savepath = os.path.join(dir_data, f"PWSCalc/PWS_through2021_allSeas_wnorm_4monthslag.tif")
save_tif(pws, geotransform, savepath)

#save other output
df_pws = pd.DataFrame(data={'pws': coefSum, 'x_loc': x_loc, 'y_loc': y_loc, 'R2': pwsR2,'coefs': coefs} )
picklePath = os.path.join(dir_data, f"PWSCalc/PWSinfo_through2021_allSeas_wnorm_4monthslag.pkl")
with open(picklePath,'wb') as file:
    pickle.dump(df_pws, file)

#%% Output maps of various DFMC statistics to check for cross-correlations
mnDFMC = master.groupby('pixel_index')['dfmc(t)'].apply(np.mean)
stdDFMC = master.groupby('pixel_index')['dfmc(t)'].apply(np.std)
q10DFMC = master.groupby('pixel_index')['dfmc(t)'].apply(np.quantile, q=0.10)
q5DFMC = master.groupby('pixel_index')['dfmc(t)'].apply(np.quantile, q=0.05)

def assignToMap(ds, theseCoef, y_loc, x_loc):
    '''
    inputs: dataset with right size, coefficients, y_loc, x_loc
    outputs: array
    '''
    thisMap = np.array(ds.GetRasterBand(1).ReadAsArray()) #get size
    thisMap[:,:] = np.nan
    thisMap[y_loc, x_loc] = theseCoef
    
    return thisMap


#could probably put this in function but skip for efficiency for now
mnDFMCMap = assignToMap(ds, mnDFMC, y_loc, x_loc)
stdDFMCMap = assignToMap(ds, stdDFMC, y_loc, x_loc)
q5DFMCMap = assignToMap(ds, q5DFMC, y_loc, x_loc)
q10DFMCMap = assignToMap(ds, q10DFMC, y_loc, x_loc)

fig, ax = plt.subplots()
ax.imshow(mnDFMCMap)
plt.title('mean DFMC map')

save_tif(mnDFMCMap, geotransform, os.path.join(dir_data, f"meanDFMC.tif"))
save_tif(stdDFMCMap, geotransform, os.path.join(dir_data, f"stdDFMC.tif"))
save_tif(q5DFMCMap, geotransform, os.path.join(dir_data, f"q5DFMC.tif"))
save_tif(q10DFMCMap, geotransform, os.path.join(dir_data, f"q10DFMC.tif"))

