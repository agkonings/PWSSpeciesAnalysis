# -*- coding: utf-8 -*-
"""

Calculate cross-correlation between random forest features

@author: konings
"""
import pandas as pd
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def prettify_names(names):
    new_names = {"ks":"K$_{s,max}$",
                 "ndvi":"$NDVI_{mean}$",
                 "vpd_mean":"VPD$_{mean}$",
                 "vpd_cv":"VPD$_{CV}$",
                 "vpd_std":"VPD$_{std}$",
                 "ppt_mean":"Precip$_{mean}$",
                 "ppt_cv":"Precip$_{CV}$",
                 "agb":"Biomass",
                 "canopy_height": "Canopy height",
                 "root_depth":"Root depth",
                 "bulk_density":"Bulk density",
                 "nlcd": "Land cover",                
                 "aspect":"Aspect",
                 "slope":"Slope",
                 "twi":"TWI",
                 "t_mean":"Temp$_{mean}$",
                 "t_std":"Temp$_{st dev}$",
                 "lon":"Lon", "lat":"Lat",
                 "theta_third_bar": "$\psi_{0.3}$",,
                 "AWS":"Avail water storage",
                 "AI":"Aridity index",
                 "Sr": "RZ storage",
                 "species":"species",
                 "mnDFMC": "$DFMC_{mean}$",
                 "stdDFMC": "$DFMC_{std}$",
                 "HAND":"HAND",
                 "pws": "PWS"
                 }
    return [new_names[key] for key in names]

# Load data consistent with way it's done in rf_regression.py
pickleLoc = 'C:/repos/pws_drivers/data/df_wSpec.pkl'
df_wSpec = pd.read_pickle(pickleLoc)

''' 
First make a separate figure for cross-correlations with DFMC values and climate
'''
#load mnDMFC
dirDFMC = 'C:/repos/data/DFMCstats/'
ds = gdal.Open(dirDFMC + "meanDFMC.tif")
gt = ds.GetGeoTransform()
mnDFMCMap = np.array(ds.GetRasterBand(1).ReadAsArray())
ds = None
ds = gdal.Open(dirDFMC + "stdDFMC.tif")
stdDFMCMap = np.array(ds.GetRasterBand(1).ReadAsArray())
ds = None


#note that these are lat, lons of associated gri dcells with FIA sites in them, so should be at corners.
#can therefore round
latInd = np.round( (df_wSpec['lat'].to_numpy() - gt[3])/gt[5] ).astype(int)
lonInd = np.round( (df_wSpec['lon'].to_numpy() - gt[0])/gt[1] ).astype(int)
dfDFMC = df_wSpec.copy()
dfDFMC['mnDFMC'] = mnDFMCMap[latInd, lonInd]
dfDFMC['stdDFMC'] = stdDFMCMap[latInd, lonInd]
dfDFMC = dfDFMC[['mnDFMC','stdDFMC','vpd_mean','AI','ppt_cv', 'pws']]        

corrMatDFMC = dfDFMC.corr()
corrMatDFMC = corrMatDFMC.drop(['vpd_mean','AI','ppt_cv'], axis=1) 
r2bcmap = sns.color_palette("vlag", as_cmap=True)
fig, ax = plt.subplots(figsize = (3,3))
sns.heatmap(np.round(corrMatDFMC, decimals=2),
        xticklabels=prettify_names(corrMatDFMC.columns.values),
        yticklabels=prettify_names(corrMatDFMC.index.values),
        cmap = r2bcmap, vmin=-0.4, vmax=0.4,
        annot=True,  fmt=".2f", annot_kws={'size': 10})
plt.savefig("../figures/PWSDriversPaper/crossCorrDFMCStats.jpeg", dpi=300, bbox_inches = "tight")

'''
Make general cross-correlation map
'''
df_wSpec.drop(columns=['species','lat','lon','nlcd'], inplace=True)
columnOrder = ['pws', 'vpd_mean', 'AI', 'ppt_cv', 'ndvi', 'bulk_density', 'ks', 'Sr', 'aspect',
       'slope', 'twi']
df_wSpec = df_wSpec[columnOrder] #re-order manually to make easier to read
corrMat = df_wSpec.corr()
mask = np.triu(np.ones_like(corrMat, dtype=bool))
fig, ax = plt.subplots()
sns.heatmap(corrMat, mask=mask,
        xticklabels=prettify_names(corrMat.columns.values),
        yticklabels=prettify_names(corrMat.index.values),
        cmap = r2bcmap, vmin=-0.65, vmax=0.65)
        #annot=True,  fmt=".1f", annot_kws={'size': 10})
plt.savefig("../figures/PWSDriversPaper/crossCorr.jpeg", dpi=300, bbox_inches = "tight")


'''
Make climate cross-correlation map
'''
df_clim = df_wSpec[['ndvi','vpd_mean','AI','ppt_cv']]
corrClim = df_clim.corr()
mask = np.triu(np.ones_like(corrClim, dtype=bool))
fig, ax = plt.subplots()
sns.heatmap(corrClim, mask=mask,
        xticklabels=prettify_names(corrClim.columns.values),
        yticklabels=prettify_names(corrClim.columns.values),
        cmap = r2bcmap, vmin=-0.75, vmax=0.75,
        annot=True,  fmt=".2f", annot_kws={'size': 10})

