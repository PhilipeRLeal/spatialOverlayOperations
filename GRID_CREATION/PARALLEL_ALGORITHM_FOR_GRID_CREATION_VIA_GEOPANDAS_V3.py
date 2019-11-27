# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:03:26 2019

@author: Philipe_Leal
"""



import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from multiprocessing import Pool, Lock, freeze_support
import os
from functools import partial
import time
import multiprocessing

def info(time_value):
    
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    print("Time spent: ", time.time() - time_value)

def init(l):
    
    global lock
    
    lock=l
    
def Data_Arranger(to_filename):
    
    """This function concatenates and deletes temporary files. It is an arranger 
        of the multicessing data results"
    """
    
    Base = os.path.join(os.path.dirname(to_filename), 'temp')
    
    
    Strings = [file for file in os.listdir(Base)]
        
    Strings = [os.path.join(Base, S) for S in Strings]
    
    if not os.path.exists(os.path.dirname(to_filename)):
        os.mkdir(os.path.dirname(to_filename))
    
    Sq = [S for S in Strings if S.endswith('.shp')]
    
    gpd.GeoDataFrame(pd.concat([gpd.read_file(sq1) for sq1 in Sq]), crs=GDF.crs).to_file(to_filename)
    
    for sq1 in Sq:
        os.remove(sq1) 
    
    import shutil

    shutil.rmtree(Base, ignore_errors=True) 
    
    
 
    
def parallelize_df(gdf, func, n_cores, dx=100, dy=100, verbose=False, to_filename=None):
    
    
    
    Geometries= gdf.loc[:, 'geometry'].values
    crs = gdf.crs

    pool = Pool(processes=n_cores, initializer=init, initargs=(Lock(), ) )
    
    func_partial=partial(func, dx, dy, verbose, to_filename, crs) # prod_x has only one argument x (y is fixed to 10) 
    
    
    pool.map(func_partial, Geometries)
    
    pool.close()
    pool.join()
    
    
def generate_grid_from_gdf(dx=100, dy=100, verbose=False, to_filename=None, crs=None, polygon=None):
    if verbose == True:
        info(time.time())
    else:
        None
    
    xmin,ymin,xmax,ymax = polygon.bounds

    lenght = dx
    wide = dy

    cols = list(np.arange(int(np.floor(xmin)), int(np.ceil(xmax)), wide))
    rows = list(np.arange(int(np.floor(ymin)), int(np.ceil(ymax)), lenght))
    rows.reverse()

    subpolygons = []
    for x in cols:
        for y in rows:
            subpolygons.append( Polygon([(x,y), (x+wide, y), (x+wide, y-lenght), (x, y-lenght)]) )

    
    
    lock.acquire()
    
    print('parent process: ', os.getppid(), ' has activated the Lock')
    GDF = gpd.GeoDataFrame(geometry=subpolygons, crs=crs)
    
    
    to_filename = os.path.join(os.path.dirname(to_filename), 'temp',  str(os.getpid()) + '_' + str(time.time()) + '.' + os.path.basename(to_filename).split('.')[-1])
    
    if not os.path.exists(os.path.dirname(to_filename)):
        os.mkdir(os.path.dirname(to_filename))
    
    try:
        print("to_filename: ", to_filename)
        GDF.to_file(to_filename)
    except:
        print("error in the file saving")
    lock.release()
    
    print('parent process: ', os.getppid(), ' has unlocked')
    
    
    

def main(GDF, n_cores='standard', dx=100, dy=100, verbose= False, to_filename=None):
    """
    GDF: geodataframe
    n_cores: use standard or a positive numerical (int) value. It will set the number of cores to use in the multiprocessing
    
    dx: dimension in the x coordinate to make the grid
    dy: dimenion in the y coordinate to make the grid)
    verbose: whether or not to show info from the processing. Appliable only if applying the function not
            in Windows (LINUX, UBUNTU, etc.), or when running in separte console in Windows.
    
    to_filename: the path which will be used to save the resultant file.
    """
    
    if isinstance(n_cores, str):
        
        N_cores = multiprocessing.cpu_count() -1

    elif isinstance(n_cores, int):
        
        N_cores =n_cores
        
    parallelize_df(GDF, generate_grid_from_gdf, n_cores=N_cores, dx=dx, dy=dy, verbose=verbose, to_filename=to_filename)
    Data_Arranger(to_filename)
    

####################################################################################################

if "__main__" == __name__:
    freeze_support()
    GDF = gpd.read_file(r'F:\Philipe\Doutorado\BD\IBGE\IBGE_Estruturas_cartograficas_Brasil\2017\Unidades_Censitarias\Municipios\MUNICIPIOS_PARA.shp')
    
    GDF.to_crs('+proj=poly +lat_0=0 +lon_0=-54 +x_0=5000000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs', inplace=True)
    
    to_filename = r'F:\Philipe\Doutorado\BD\IBGE\IBGE_Estruturas_cartograficas_Brasil\2017\Unidades_Censitarias\Municipios\GRIDDED\MUNICIPIOS_PARA_GRIDDED_1000x1000m.shp'
    
    
    main(GDF, dx=1000, dy=1000, verbose=True, to_filename=to_filename)
    
    
    GDF = gpd.read_file(to_filename)

    