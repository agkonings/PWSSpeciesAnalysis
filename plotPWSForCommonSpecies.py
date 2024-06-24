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
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale = 1, style = "ticks")
plt.rcParams.update({'font.size': 16})

flpth = 'G:/My Drive/0000WorkComputer/dataStanford/fiaAnna/'

#Load PWS lats and lons, geotransform
ds = gdal.Open('G:/My Drive/0000WorkComputer/dataStanford/PWSCalc/PWS_through2021_allSeas_nonorm_4monthslag_exact6years.tif')
gt = ds.GetGeoTransform()
pws = ds.GetRasterBand(1).ReadAsArray()
pws_y,pws_x = pws.shape
wkt_projection = ds.GetProjection()
ds = None

#look at common species at actual sites used in study (also filtered for NLCD, data availability)
pickleLoc = './data/df_wSpec.pkl'
rfLocs = pd.read_pickle(pickleLoc)
vc = rfLocs['species'].value_counts()
#five most common among actual sites
#756 = honey mesquite
#65 = Utah juniper
#122 = ponderosa pine
#202 = Douglas-fir
#69 = oneseed juniper (juniperus monisperma)

'''
Ok, ready to plot! Manually checked five species are most common
'''

plotSpecList = [202, 756, 122, 65, 69]
legLabels = ["Douglas-fir", "Honey mesquite", "Ponderosa pine", "Utah juniper", "Oneseed juniper", "All"]

#filter dataframe to be used for plotting purposes so that it only 
#contains sites where the dominant cover is one of the species to be plotted
topDomLocs = rfLocs.copy()
noDataRows = topDomLocs.loc[~topDomLocs.species.isin(plotSpecList)]
topDomLocs.drop(noDataRows.index, inplace=True)


fig, ax = plt.subplots()
ax1 = sns.displot( topDomLocs, x='pws', hue='species', kind='kde', common_norm=False, \
            palette=sns.color_palette(n_colors=5), fill=False, bw_adjust=0.75, legend=False)
sns.kdeplot(rfLocs['pws'], ax=ax1, color='k', bw_adjust=0.75)
plt.ylabel("Density", size=18); plt.xticks(fontsize=16)
plt.xlabel("PWS", size=18); plt.yticks([], fontsize=16)
plt.xlim(0,6)
plt.legend(legLabels, loc="lower center", bbox_to_anchor=(0.5,-0.5), ncol=2, title=None, fontsize=18)
#plt.savefig("../figures/PWSDriversPaper/PWSkdesbyspecies.jpeg", dpi=300)

'''
Calculate standard deviation of species means vs mean standard deviation per species
for use in paper text
'''
stdPerSpecList = []
mnPerSpecList = []
for spec in np.unique(rfLocs['species']):
    if vc[spec] > 50:
        thisMean = rfLocs[rfLocs['species'] == spec].pws.mean()
        thisStd = rfLocs[rfLocs['species'] == spec].pws.std()
        mnPerSpecList.append(thisStd)
        stdPerSpecList.append(thisStd)

print( 'Standard deviation across species means is ', str(np.std(mnPerSpecList)) )
print( 'Mean value of within-species standard deviations is ', str(np.mean(stdPerSpecList)) )

#plot bar chart of top species for paper
#don't overcomplicat ethings, just do manually
fig, ax = plt.subplots(figsize=(3,3))
ax = sns.barplot(y=[0,1,2,3,4], x=vc[plotSpecList], palette=sns.color_palette(n_colors=5), orient='h')
ax.set_yticks([0,1,2,3,4])
ax.set_yticklabels(legLabels[:-1]) 
ax.set(xlabel='number of sites\n(out of 21,455)')
plt.savefig("../figures/PWSDriversPaper/nSites_keySpecies.svg", dpi=300, bbox_inches='tight')
plt.show()