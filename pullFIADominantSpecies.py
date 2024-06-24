# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:08:51 2022

@author: konings
 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import pickle

#notes: seems like you can find one plot ID # that corresponds to tally different (10 degrees apart!)
#lat and lon dependign on year. So that value is basically useless.
# for now just make site index with year and sort on unique values for those

flpth = 'G:/My Drive/0000WorkComputer/dataStanford/fiaAnna/'

#read two csv files
varsdf = pd.read_csv(flpth + 'CONDVars_LL.csv', sep=',', header=0, 
                     usecols=['CN','INVYR', 'PLOT', 'MEASYEAR', 'CONDID', \
					 'LAT', 'LON', 'FORTYPCD', 'FLDTYPCD', 'CONDPROP_UNADJ','BALIVE'])       
spdf = pd.read_csv(flpth + 'COND_Spp_Live_Ac.csv', sep=',', header=0, 
                     usecols=['COND_CN','SPCD','BALiveAc','TPALiveAc'])
p50df = pd.read_csv(flpth + 'FIA_traits_Alex.csv', sep=',', header=0)

#varsdf has 1,084,063 unique CN values, out of the same number of rows
#spdf has 413,987 unique COND_CN values, out of 1,926,679

#merge dataset to add lat lon information to condition information
combdf = spdf.merge(varsdf, left_on=['COND_CN'], right_on=['CN'], how='inner')
print( 'Minimum lat: ' + str(np.min(combdf['LAT'])) )
print( 'Maximum lat: ' + str(np.max(combdf['LAT'])) )
print( 'Minimum lon: ' + str(np.min(combdf['LON'])) )
print( 'Maximum lon: ' + str(np.max(combdf['LON'])) )

#Load PWS lats and lons, geotransform to figure out how to aggregate
ds = gdal.Open('C:/repos/data/pws_features/PWS_through2021_allSeas.tif')
gt = ds.GetGeoTransform()
pws = ds.GetRasterBand(1).ReadAsArray()
pws_y,pws_x = pws.shape
wkt_projection = ds.GetProjection()
ds = None

#just brute force search for locations where one species is dominant
#can't do this by condition because identical plots measured 10 years apart will 
#have different condition numbers.
#instead, approach by making cheap site index that maps to each location within 4km


lonInd = np.floor( (combdf['LON']-gt[0])/gt[1] ).to_numpy().astype(int)
latInd = np.floor( (combdf['LAT']-gt[3])/gt[5] ).to_numpy().astype(int)
cheapSiteID = lonInd*1e5 + latInd 
combdf['siteID'] = cheapSiteID

#here is the brute forcing. When you have a site-index
#if two latitudes, add basal areas per species...whole mess
#if only one, pick latest year
dominantThresh = 0.75
dominantLocs = pd.DataFrame()
noGoodCnt = 0
for unSite in combdf['siteID'].unique(): 
    sitedf = combdf[combdf['siteID'] == unSite]
    #if only species listed, assume dominates
    if len(sitedf) == 1:
        dominantLocs = pd.concat([dominantLocs, sitedf])
    else:
        #if multiple years of meas at same site, pick the most recent
        if sitedf['INVYR'].nunique() > 1:
            sitedf = sitedf[sitedf['INVYR'] == sitedf['INVYR'].max()]
        #if mulitple condition IDs, pick the lowest
        if sitedf['CONDID'].nunique() > 1:
            sitedf = sitedf[sitedf['CONDID'] == sitedf['CONDID'].min()]
        #note that if there's two sites within 4 km, they can have the same 
        #species across the two sites, and harder to check which is dominant
        #check for this and treat the whole set-up differently then
        if sitedf['SPCD'].nunique() < len(sitedf):
            foundSite = False
            #could probably replace this loop with groupby 
            #priortize programmer time for nwo
            for unSpecies in sitedf['SPCD'].unique():
                theseSpec = sitedf[sitedf['SPCD'] == unSpecies]            
                BARat = theseSpec['BALiveAc'].sum()/sitedf['BALiveAc'].sum()
                if BARat > dominantThresh:
                    #have two locations now, but otherwise identical for our
                    #later purpose. So just add one randomly 
                    dominantLocs = pd.concat([dominantLocs, theseSpec.head(1)])
                    foundSite = True
            #if you make it through this loop, no species is dominant
            if foundSite == False:
                noGoodCnt += 1
        else:
            BARat = sitedf['BALiveAc']/sitedf['BALiveAc'].sum()
            #check if one species is dominant in terms of basal area
            if BARat.max()>dominantThresh: 
                if len(sitedf[BARat>dominantThresh])>1:
                    raise Exception('Multiple dominant species are impossible')
                dominantLocs = pd.concat([dominantLocs, sitedf[BARat>dominantThresh]])
            else:
                noGoodCnt += 1            
#store as pickle file for use elsewhere in mapping/exploring the data sources
pickleLoc = './data/dominantLocs.pkl'
with open(pickleLoc, 'wb') as file:
    pickle.dump(dominantLocs, file)
                
