# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:17:10 2018

@author: Philipe Leal
"""

# importacao de Bibliotecas:
    



def Zonal_Stat(Shapefile_path, Raster_Base_Path, Raster_Burn_Path=None):
    """ Esta função retorna um objeto geopandas com os atributos zonais espaciais.
        
        Shapefile_path: caminho do shapefile a ser rasterizado
        
        Raster_Base_Path: caminho do raster para calcular as estatísticas zonais
        
        Raster_Burn_Path: permite definir onde será salvo o array rasterizado do vetor (burned_array).
            
            Por padrão, o vetor rasterizado será salvo na mesma pasta do vetor base, e com o mesmo nome do arquivo vetorial.
            Caso o Raster_Burn_Path seja definido, ele deverá ser definido completamente (contendo até mesmo a terminação '.tif')
    """
    
    import geopandas as gpd
    import rasterio
    from rasterio import features

    import numpy as np
    import os
    
    
    Shapefile_path = Shapefile_path
    Raster_Base_Path = Raster_Base_Path
    
    if Raster_Burn_Path == None:
        
        Raster_Burn_Path = str(os.path.abspath(Shapefile_path)[:-4]) + str('.tif')
        
    else:
        Raster_Burn_Path = Raster_Burn_Path
        
        
    Vetores = gpd.read_file(Shapefile_path)
    
    rst = rasterio.open(Raster_Base_Path)
    
    # copy and update the metadata from the input raster for the output
    
    meta = rst.meta.copy()
    meta.update(compress='lzw')
    
    # Checando indice sobre vetores. Caso já tenha indice 'ID': 
        
    try:    
        Vetores.contains(Vetores.ID)
        
        for i in range(len(Vetores)):
    
            Vetores['ID'][i] = (i + 1)
    
    except AttributeError:
        
       # Caso ainda nao haja: cria-se Indice sobre Vetores:

        Vetores['ID'] = (0)
    
        for i in range(len(Vetores)):
        
            Vetores['ID'][i] = (i + 1)
            
    # Fase de rasterização do Vetor.
    
    out= None
    out_arr=None
    
    Raster_base=None
    
    with rasterio.open(Raster_Burn_Path, 'w', **meta) as out:
        out_arr = out.read(1)
        shapes = []
        # this is where we create a generator of geom, value pairs to use in rasterizing
        for geom, value in zip(Vetores.geometry, Vetores.ID):
            shapes.append((geom,value))
        
      
            
        print("\n\n\n")
        
        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform, dtype=np.float32)
        out.write_band(1, burned)
        
        Raster_base_DT = rasterio.open(Raster_Base_Path)
    
        Raster_base = Raster_base_DT.read(1)
    
    
    # Agora fazendo algebra de matrizes para obtenção das estatísticas zonais:
        
    out_arr = np.where(out_arr<(-0.1), np.nan, out_arr)



    Raster_base=None
    
    Raster_base_DT = rasterio.open(Raster_Base_Path)
    
    Raster_base = Raster_base_DT.read(1)
    
    Raster_base = np.nan_to_num(Raster_base)
    
    print(np.max(Raster_base))

    
    Base_name = os.path.basename(Raster_Base_Path)
    
    Base_name = Base_name[:-4]
    
    Vetores['MAX_' + str(Base_name)] = (0)
    Vetores['MIN_'+ str(Base_name)] = (0)
    Vetores['MEAN_' + str(Base_name)] = (0)
    Vetores['STD_' + str(Base_name)] = (0)
    Vetores['SUM_' + str(Base_name)] = (0)
    
    for i in range(len(Vetores)):
        ID = Vetores['ID'][i]
        ID = ID * 1.0
        
        out_arr_cor = np.where(out_arr==ID, 1.0, np.nan)
        
        Raster_Stat = Raster_base * out_arr_cor
        
        
        
        print(i+1, np.nanmax(Raster_Stat))
        
        Vetores['MAX_' + str(Base_name)][i] = np.nanmax(Raster_Stat)
        Vetores['MIN_'+ str(Base_name)][i] = np.nanmin(Raster_Stat)
        Vetores['MEAN_' + str(Base_name)][i] = np.nanmean(Raster_Stat)
        Vetores['STD_' + str(Base_name)][i] = np.nanstd(Raster_Stat)
        Vetores['SUM_' + str(Base_name)][i] = np.nansum(Raster_Stat)

    return Vetores