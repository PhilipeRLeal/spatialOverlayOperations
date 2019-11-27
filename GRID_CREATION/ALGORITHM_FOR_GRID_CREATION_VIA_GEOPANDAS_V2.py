# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:28:19 2019

@author: Philipe Leal
"""


import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

def generate_grid_from_gdf(feature, dx=100, dy=100, verbose=False):
    
    
    xmin,ymin,xmax,ymax = feature.bounds

    lenght = dx
    wide = dy

    cols = list(range(int(np.floor(xmin)), int(np.ceil(xmax)), wide))
    rows = list(range(int(np.floor(ymin)), int(np.ceil(ymax)), lenght))
    rows.reverse()

    polygons = []
    for i, x in enumerate(cols):
        for j, y in enumerate(rows):
                        
            if verbose==True:
                print("Processing poligon: [xo {0} xf {1} -- y0{2}, yf {3} ]".format(x, x+dx, y, y+dy))
            
            polygons.append( Polygon([(x,y), (x+wide, y), (x+wide, y-lenght), (x, y-lenght)]) )

    GRID = gpd.GeoDataFrame({'geometry':polygons}, crs=GDF.crs)
    
    return GRID
    



def main(GDF,  dx=100, dy=100, to_file_path=None, verbose=False):
    """
    GDF: geodataframe
    
    dx: dimension in the x coordinate to make the grid
    dy: dimenion in the y coordinate to make the grid)
    
    """
    
    
    for i in GDF.iterrows():
        
        j = gpd.GeoSeries(i[1])
        
        GRID_GEOM = generate_grid_from_gdf(i[1].geometry, dx=dx, dy=dy, verbose=verbose)
        
        for key, value in j.items():
            GRID_GEOM[key] = value
        
        GRID_GEOM.to_file(to_file_path)
    

    

if "__main__" == __name__:
    
        
    try:
        GDF = gpd.read_file(r'C:\Doutorado\BD\IBGE_Estruturas_cartograficas_Brasil\2017\Unidades_Censitarias\Municipios\MUNICIPIOS_PARA.shp')
        GDF.crs
        GDF.to_crs('+proj=poly +lat_0=0 +lon_0=-54 +x_0=5000000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs', inplace=True)
    except:
        print("File not found")
        
    to_file_path = r'C:\Doutorado\BD\IBGE_Estruturas_cartograficas_Brasil\2017\Unidades_Censitarias\Municipios\PARA_MUNICIPIOS_GRIDDED_5x5_km2.shp'
    
    main(GDF, dx=5000, dy=5000, to_file_path=to_file_path, verbose=False)
    

    