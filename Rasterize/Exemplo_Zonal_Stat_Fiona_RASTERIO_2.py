# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:16:07 2018

@author: Philipe Leal
"""

import rasterio
import numpy as np
import geopandas as gpd

from shapely.geometry import mapping
from rasterio.mask import mask


def extract_zonal_statistics(src, geom, stat, band):
    geom_map = mapping(geom)
    out_image, out_transform = mask(src, [geom_map], crop=True, nodata=np.nan, all_touched=True)
    return stat(out_image[band - 1])

gdf = gpd.read_file(r"C:\Doutorado\2_Trimestre\Disciplinas\Climatologia\ZEEs\ZEE_Antares.shp")

Lista_de_estatisticas = ['np.nansum', 'np.nanmean', 'np.std', 'np.var']
with rasterio.open(r"C:\Doutorado\2_Trimestre\Disciplinas\Climatologia\Anomalia_simples\Full_Year\Anomalia_Integrada_para_1_ano_85_2055.tif") as src:
    
    gdf['zonal_stat_mean'] = gdf.geometry.apply(
        lambda geom: extract_zonal_statistics(src, geom, stat=np.nanmean, band=1))
    
    gdf['zonal_stat_max'] = gdf.geometry.apply(
lambda geom: extract_zonal_statistics(src, geom, stat=np.nanmax, band=1))
 
    gdf['zonal_stat_min'] = gdf.geometry.apply(
lambda geom: extract_zonal_statistics(src, geom, stat=np.nanmin, band=1))
  
    gdf['zonal_stat_std'] = gdf.geometry.apply(
lambda geom: extract_zonal_statistics(src, geom, stat=np.nanstd, band=1))
       

    
gdf.head()