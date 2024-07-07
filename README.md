This repository contains the analysis code that led to the results presented in the article in Global Change Biology titled “Tree species explain only half of explained spatial variability in plant water sensitivity" (Konings et al., 2024). This README is written assuming the reader has read that article. 

Most of the data files associated with this analysis (including the input plant water sensitivity map, the subsetting of theis map to FIA sites, the biogeographic predictor values at the FIA sites, their dominant species, and the predicted plant water sensitivity values from the species-based and random forest models) are uploaded to an associated Dryad repository. The Dryad repository can be found at [doi:10.5061/dryad.g1jwstr05](doi:10.5061/dryad.g1jwstr05) This may be helpful for running some of the code in this repository. 


Much of the core analysis code is contained in the file rf_regression.py, using an HDF5 file of input maps for the random forest that is created in make_data.py. The file pullFIADominantSpecies.py identifies the FIA sites where a single species covers more than 75% of the basal area.  The file pws_calculation.py calculations the initial PWS maps, as described in the article. The file checkCrossCorrs.py is used to determine which potentially inputs have sufficiently low cross-correlated to be useable for the random forest model, as described in the article’s methods file. It also is used to calculate some of the supplementary figures. Lastly, the file writeDataToNETCDF.py is used to write the output of the scripts to the main netCDF output file stored in the Dryad repository. It uses as input a python dill file (packaging the entire contents/state of the Python console session at the end of running rf_regression.py). 


The main figures of the article are derived from the following scripts:  
Figure 1: plotPWSForCommonSpecies.py  
Figure 2: plotPWSForCommonSpecies.py  
Figure 3: rf_regression.py  
Figure 4: rf_regression.py  
Figure 5: rf_regression.py  


The pickle files are supplementary: rf_regression_dill.py was created using the Python dill package and contains the entire state of the console after running rf_regression.py, including most of the finished calculations used for the article’s main results. The df_wSpeckle file is the input pandas dataframe used in the random forest model in rf_regression.py. 


If you have any further questions, please contact the lead author Alexandra Konings at lastname at stanford dot edu
