# Data Access

The data used for this project was downloaded from a variety of sources and preprocessed for modeling in the Jupyter notebooks found in the `bayareaco2/notebooks` folder. Due to storage limitations, raw data files were not uploaded to the repository. Only the `.csv` files that were created during preprocessing by extracting data from the raw files are included in the `Data` folder. As a result,  `bayareaco2/notebooks/training` and `bayarea/notebooks/results` notebooks can be run  immediately upon cloning the repository and installing the `co2` environment and the `bayareaco2` module; the notebooks in `bayareaco2/notebooks/preprocessing` can only be run if the additional necessary data files are downloaded.

## Data Sources
The relevent data sources are summarized in the table below:

| Feature    | Year   | Source   |
|-------------|-------------|-------------|
| CO2 & Meteorology  | 2012-2024  | [BEACO2N](http://beacon.berkeley.edu/about/) |
| Land Use  | 2021  | [ESRI Sentinel-2 10m LU/LC](https://www.arcgis.com/apps/instant/media/index.html?appid=fc92d38533d440078f17678ebc20e8e2) |
| Industrial (Land Use)  | 2021-2023 | [California General Plan Land Use](https://gis.data.ca.gov/datasets/Gov-OPR::california-general-plan-land-use/about) |
| Roads  | 2022  | [U.S. Census Bureau 2022 California Roads TIGER/Line](https://catalog.data.gov/dataset/tiger-line-shapefile-2022-state-california-primary-and-secondary-roads)  |
| AADT  | 2022  | [Caltrans 2022 Traffic Volumes](https://gis.data.ca.gov/datasets/d8833219913c44358f2a9a71bda57f76_0/about)  |
| NDVI  | 2022 | [USGS EarthExplorer: Landsat 8-9 OLI/TIRS C2 L2 Surface Reflectance-derived NDVI](https://earthexplorer.usgs.gov)  |
| Population Density  | 2022  | [California Hard-to-Count Index](https://cacensus.maps.arcgis.com/apps/webappviewer/index.html?id=48be59de0ba94a3dacff1c9116df8b37) & [U.S. Census Bureau 2022 Census Tract TIGER/Line](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2022&layergroup=Census+Tracts) |

## Data Directory Structure
The files used as inputs for preprocessing and data extraction were organized as follows:
```
irp-acs223
├─ bayareaco2
└─ Data
   ├─ AADT
   │  └─ Traffic_Volumes_AADT
   │     └─ HWY_Traffic_Volumes_AADT.shp
   ├─ Califorinia_General_Plan_La
   │  └─ California_General_Plan_Land_Use.shp
   ├─ CO2
   │  └─ BEACO2N_2012_01_01_to_2024_06_03
   ├─ Land_Use
   │  └─ 10S_20210101-20220101.tif
   ├─ NDVI
   │  ├─ LC08_L2SP_044034_20220409_20220419_02_T1_SR_B4.TIF
   │  └─ LC08_L2SP_044034_20220409_20220419_02_T1_SR_B5.TIF
   ├─ Population_Density
   │  ├─ California Hard-to-Count Index by Census Tract.csv
   │  └─ tl_2022_06_tract
   │     └─ tl_2022_06_tract.shp
   └─ Roads
      └─ tl_2022_06_prisecroads
         └─ tl_2022_06_prisecroads.shp
```
This directory structure is directly compatible with filepaths specified in the notebooks in this repository.

## Compressed Data Download
The complete [Data folder](https://imperiallondon-my.sharepoint.com/:f:/g/personal/acs223_ic_ac_uk/EtXHs1IbYyROuU5kBWQCmUQBdoRrs1soloS-EOybjkE9-w?e=zyMnhw) is available via OneDrive. The size of the decompressed folder is 2.85 GB. 