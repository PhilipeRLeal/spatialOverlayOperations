# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:16:53 2019

@author: lealp
"""

import pandas as pd
pd.set_option('display.width', 50000)
pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 5000)


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib
import seaborn as sn
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

import geopandas as gpd



# funções efetivas


def _explode(indata):
    indf = indata
    outdf = gpd.GeoDataFrame(columns=indf.columns)
    for idx, row in indf.iterrows():
        if type(row.geometry) == Polygon:
            outdf = outdf.append(row,ignore_index=True)
        if type(row.geometry) == MultiPolygon:
            multdf = gpd.GeoDataFrame(columns=indf.columns)
            recs = len(row.geometry)
            multdf = multdf.append([row]*recs,ignore_index=True)
            for geom in range(recs):
                multdf.loc[geom,'geometry'] = row.geometry[geom]
            outdf = outdf.append(multdf,ignore_index=True)
    return outdf


def distance_between_two_geometries(geometry1, geometry2):
    distance = geometry1.distance(geometry2)
    
    
    return distance
    


def check_spatial_distance_between_geometries_of_two_GeoDataFrames(GDF1, GDF2):
    

    distance = np.vectorize(distance_between_two_geometries)


    Result = {}
    
    GDF1 = _explode(GDF1)
    
    
    Result = GDF1.geometry.map(lambda x: distance(x, GDF2.geometry))

    Result = pd.Series(Result).reset_index().set_index(gdf.index).rename({'geometry':'distance'}, axis=1)['distance'].apply(pd.Series)
    
    Result.columns = gdf.index
    
    return Result


if '__main__' == __name__:
        
    path = r'F:\Philipe\Doutorado\BD\IBGE\IBGE_Estruturas_cartograficas_Brasil\2017\Unidades_Censitarias\Municipios\MUNICIPIOS_PARA.shp'
    
    gdf = gpd.read_file(path , encoding='UTF-8').head(10)
    
    
    
    gdf.head()
    
    gdf.to_crs(epsg=5880, inplace=True)
    
    
    
    gdf.set_index('NM_MUNICIP', inplace=True)
    
    
    print(' teste 1 (deverá ser mais rápido) \n')
    
    
    Result = check_spatial_distance_between_geometries_of_two_GeoDataFrames(gdf, gdf)
    
        
    print(' teste 2 (deverá ser mais lento) \n')
    

    from itertools import permutations 
    
    
    
    def lento(gdf):
        permutacoes = permutations(gdf.index, 2)
        
        
        distances = {}
        
        
        for idx1, idx2 in permutacoes:
           
            
            distance = gdf.loc[idx1].geometry.distance(gdf.loc[idx2].geometry)
            
            distances[ (idx1, idx2) ] = distance
            
            
        distances = pd.Series(distances).to_frame().unstack()
        
        return distances
    
    %timeit Result2 = lento(gdf)
    