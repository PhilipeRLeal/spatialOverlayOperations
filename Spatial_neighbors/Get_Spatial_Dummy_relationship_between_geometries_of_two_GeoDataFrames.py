#!/usr/bin/env python
# coding: utf-8

# In[1]:


# references in :

# https://www.mdh.se/polopoly_fs/1.49051!/Menu/general/column-content/attachment/MAA704_ht12_ce_2_bbversion.pdf


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import sys
import os
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon


# In[3]:



# In[84]:
def explode(indata):
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

def touches_between_two_geometries(geometry1, geometry2, spatial_operation='touches'):
    
    Funct = getattr(geometry1, spatial_operation)

    if Funct(geometry2):
        return 1
    else:
        return 0


# In[81]:


def check_spatial_relationship_between_geometries_of_two_GeoDataFrames(GDF1, GDF2):
    

    touches = np.vectorize(touches_between_two_geometries)


    Result = {}

    for i in range(len(GDF1)):
        Result[GDF1.index[i]] = touches(explode(GDF1).geometry[i], GDF2.geometry, spatial_operation='intersects')


    Result = pd.DataFrame(Result)

    Result.index = GDF2.index

    return Result


# # Costs between positions in the matrix:

# In[7]:


def minCost_towards_XY(cost, m, n): 
  
    # Instead of following line, we can use int tc[m+1][n+1] or 
    # dynamically allocate memoery to save space. The following 
    # line is used to keep te program simple and make it working 
    # on all compilers. 
    cost = np.array(cost)
    R, C = np.shape(cost)
    
    tc = [[0 for x in range(C)] for x in range(R)] 
  
    tc[0][0] = cost[0][0] 
  
    # Initialize first column of total cost(tc) array 
    for i in range(1, m+1): 
        tc[i][0] = tc[i-1][0] + cost[i][0] 
  
    # Initialize first row of tc array 
    for j in range(1, n+1): 
        tc[0][j] = tc[0][j-1] + cost[0][j] 
  
    # Construct rest of the tc array 
    for i in range(1, m+1): 
        for j in range(1, n+1): 
            tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] 
  
    return tc[m][n]


# # Número mínimo de conexões para conexão de todos os polígonos:

# In[10]:


def min_conection_steps(connection):
    
    connection = np.array(connection)
    
    Steps = connection[np.sum(connection, axis=1) > 1].shape[1] -1
    
    return Steps


# In[ ]:


if '__main__' == __name__:
    Municipios_Para = gpd.read_file(r'F:\Philipe\Doutorado\BD\IBGE\IBGE_Estruturas_cartograficas_Brasil\2017\Unidades_Censitarias\Municipios\MUNICIPIOS_PARA.shp')


    Municipios_Para.set_index('CD_GEOCMU', inplace=True)

    cost1 = check_spatial_relationship_between_geometries_of_two_GeoDataFrames(Municipios_Para, Municipios_Para)

    
    print("N° de passos mínimos para conectar todos os polígonos: ", min_conection_steps(cost1))
    
    ####### Exemplo 2:
    
    print("\n\nExemplo 2.")
    
    
    cost = [[1,1,0,0,0,0,0,0,1], # row 0 
        [1,1,1,0,0,0,0,0,0],   # row 1 
        [0,1,1,1,0,0,0,1,0],   # row 2 
        [0,0,1,1,1,0,0,0,0],   # row 3 
        [0,0,0,1,1,0,0,0,0],   # row 4 
        [0,0,0,0,1,1,1,0,1],
        [0,0,0,0,0,1,1,0,0],
        [0,0,1,0,0,0,0,1,0],
        [0,0,0,0,0,1,0,0,1]]   # row 5 
    
    cost = pd.DataFrame(cost)
    
    print("Matriz cost simples: \n", cost)



    # Exemplo1: 
    print('Distância entre os polígonos {0} e {1} : {2}'.format(cost.index[5],
                                                                cost1.index[4],
                                                                minCost_towards_XY(cost1.replace(0, 999999), 5, 4)))

    # Exemplo1: 

    print('Distância entre os polígonos {0} e {1} : {2}'.format(cost.index[7],
                                                                cost.index[7],
                                                                minCost_towards_XY(cost1.replace(0, 999999), 7, 7)))


# In[11]:




