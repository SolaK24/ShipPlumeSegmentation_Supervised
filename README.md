# ShipPlumeSegmentation_Supervised

In this repository we provide the code used for the experiments on ship plume segmentation using 
multivariate classifiers on TROPOMI/S5P satellite data.

## Data set
In the direction ``./data`` we provide a labeled data set that can be use for supervised learning experiments.
The data set combines TROPOMI/S5P NO2 data, ECMWF 10m reanalyzis wind data, information about the dimensions and speed
of the ships in the area. The borders of the images were defined based on AIS data of ship positions.
Due to the privacy reasons, we removed all information about the uniqueidentifiers of the ships,
for which the images were generated. To further protect the shipping companies 
from all kinds unjustifiable accusations, we do not provide the exact geografical coordinates 
for the studied TROPOMI measurements. 
All TROPOMI measurements reported in the dataset were retrieved from the area limited by the following 
coordinate range: lon: [19.5, 29.5], lat: [31.5, 34.2] - eastern Mediterranean region. Further details of data preprocessing can be found in the respective paper. The DOI will be added here upon the publication.

## Python scripts
 - To check the performance of the models reported in the article,
run the script ``classification_experiments_cv.py``
 - To generate feature importance reports for the studied models, run the script ``feature_importance_exp.py``










