# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:03:26 2019

@author: Philipe_Leal
"""



import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from multiprocessing import Pool, Lock, freeze_support
import os
from functools import partial
import time

def info(time_value):
    
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    print("Time spent: ", time.time() - time_value)

def init(l):
    
    global lock
    
    lock=l

    
 
    
def parallelize_df(gdf, func, n_cores, dx=100, dy=100, verbose=False, to_filename=None):
    
    
    
    Geometries= gdf.loc[:, 'geometry']
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

    
    
    print('Child process: ', os.getpid(), ' has activated the Lock')
    GDF = gpd.GeoSeries(subpolygons, crs=crs)
    
        
    if not os.path.exists(os.path.dirname(to_filename)):
        os.mkdir(os.path.dirname(to_filename))
    
    lock.acquire()
    
    print("\n\n\n")
    print("Saving file")
    print("\n\n\n")
    GDF.to_file(to_filename, mode='a')

    
    lock.release()
    
    print('Child process: ', os.getpid(), ' has unlocked')
    
    
    

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
        import multiprocessing
        N_cores = multiprocessing.cpu_count() -1

    elif isinstance(n_cores, int):
        
        N_cores =n_cores
        
    
    
    parallelize_df(GDF, generate_grid_from_gdf, n_cores=N_cores, dx=dx, dy=dy, verbose=verbose, to_filename=to_filename)

    print("Process has well finished!!!!!!!!!! \n You may close the TAB!")


####################################################################################################

if "__main__" == __name__:
    freeze_support()
    GDF = gpd.read_file(r'F:\Philipe\Doutorado\BD\IBGE\IBGE_Estruturas_cartograficas_Brasil\2017\Unidades_Censitarias\Municipios\MUNICIPIOS_PARA.shp')
    GDF.head()
    
    GDF.set_index('CD_GEOCMU', inplace=True)
    GDF.to_crs('+proj=poly +lat_0=0 +lon_0=-54 +x_0=5000000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs', inplace=True)
    to_filename = r'F:\Philipe\Doutorado\BD\IBGE\IBGE_Estruturas_cartograficas_Brasil\2017\Unidades_Censitarias\Municipios\GRIDDED\MUNICIPIOS_PARA_SIRGAS_2000_polyconic_GRIDDED_5000x5000m.shp'
    
    
    main(GDF, dx=5000, dy=5000, verbose=True, to_filename=to_filename)
    
    import sys
    
    sys.path.insert(0, r'C:\Users\lealp\Dropbox\Profissao\Python\OS_e_System')
    
    