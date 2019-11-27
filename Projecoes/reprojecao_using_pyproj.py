# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:46:04 2019

@author: lealp
"""


import pyproj
from functools import partial
from shapely.ops import transform
import geopandas as gpd


def reproject(gdf, source_crs='gdf.crs', to_crs='epsg:31983'):
    project = partial(
          pyproj.transform,
          pyproj.Proj(gdf.crs), # source coordinate system
          pyproj.Proj(init=to_crs))

    Result = [transform(project, geom) for geom in gdf.geometry]

    return {'transformed_geometries':Result, 'geoseries_do_novo_crs':gpd.GeoSeries(Result, crs=pyproj.transform)}

