
# coding: utf-8



import sys
import geopandas as gdp
import os


Modulo_path = r'C:\Users\Philipe Leal\Dropbox\Profissao\Python\Shapefile\Geopandas\Rasterize'

sys.path.insert(0, Modulo_path)

# uma vez definido o path de importação do modulo de interesse, é só importá-lo normalmente!

from Funcao_Zonal_Stat_Fiona_RASTERIO import Zonal_Stat as ZS

shp_fn = r'C:\Doutorado\2_Trimestre\Disciplinas\Climatologia\ZEEs\ZEE_Antares.shp'
rst_fn = r'C:\Doutorado\2_Trimestre\Disciplinas\Climatologia\Anomalia_simples\Full_Year\Anomalia_Integrada_para_1_ano_45_2055.tif'
out_fn = str(os.path.abspath(shp_fn)[:-4]) + str('.tif')


Diretorio_Tif_Path = os.path.dirname(rst_fn)

Lista_Tif_Files = []
for file in os.listdir(Diretorio_Tif_Path):
    if file.endswith(".tif"):
        
        Lista_Tif_Files.append(os.path.join(Diretorio_Tif_Path, file))

for i in range(len(Lista_Tif_Files)):
    
    print("\nCalculando Estatistica Zonal para: ", Lista_Tif_Files[i],"\n")
    
    A=ZS(shp_fn, Lista_Tif_Files[i])

       
    # uma vez definido o path de importação do modulo de interesse, é só importá-lo normalmente!
    
    Alternate_shapefile_path = shp_fn[:-4] + str('_zonal_stat')
     
    try: 
        os.remove(Alternate_shapefile_path)
    except OSError:
        pass
    
    
    
    A.to_file(driver='ESRI Shapefile',filename=Alternate_shapefile_path)
    
    print('Alternate_shapefile_path: ', Alternate_shapefile_path)



A = gdp.read_file(r'C:\Doutorado\2_Trimestre\Disciplinas\Climatologia\ZEEs\ZEE_Antares_zonal_stat\ZEE_Antares_zonal_stat.shp')


A.keys()



