# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 17:11:13 2018

@author: Philipe Leal
"""

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

import cartopy.crs as ccrs




file_name = r'C:\Doutorado\Tese\SINAN\Casos_hepatite_A_por_estado_por_ano\Por_Regioes_BR_por_Ano.xlsx'

## Fluxo temporal 1 ano em 1 ano:


df = pd.read_excel(file_name, sheet_name='prevalencias', header=[1,2])


stacked = df.stack()
stacked.reset_index(inplace=True)


stacked_keys = stacked.keys()

Keys_dict = {'level_0':'ANO', 'Ano':'REGIAO', 'REGIAO':'Prevalencias'}

stacked = stacked.rename(columns=Keys_dict)

stacked.set_index('REGIAO', inplace=True)


# Filtrando apenas os anos acima de 2006:
stacked = stacked[stacked['ANO'] >= 2007]


stacked['Prevalencias_relativas_%'] = stacked['Prevalencias']/np.sum(stacked['Prevalencias'])*100


SHP_path = r'C:\Doutorado\Tese\SHP\2017\UF\BRUFE250GC_SIR.shp'

SHP = gpd.read_file(SHP_path)

SHP.head()


SHP.set_index('NM_REGIAO', inplace=True)

SHP.index = SHP.index.map(str.lower)

stacked.set_index('REGIAO', inplace=True)

stacked.index = stacked.index.map(str.lower)

SHP_joined = SHP.join(stacked)

SHP_joined = SHP_joined[SHP_joined['ANO'] >=2007]

SHP_joined = SHP_joined.to_crs({'proj': 'latlong', 'ellps': 'WGS84', 'datum': 'WGS84', 'no_defs': True}) ## ellipsoide wgs84

SHP_joined.crs = {'init' :'epsg:4326'}

#SHP_joined = SHP_joined.to_crs({'init': 'epsg:3395'}) ## Mercator

xmin,ymin, xmax, ymax = SHP_joined.total_bounds
    
Latitude_central = (ymin+ ymax) /2.

Longitude_central = (xmin + xmax) /2.


Anos = np.unique(SHP_joined['ANO'])

Years = []
for Ano in Anos:
    if Ano == np.nan:
        None
    elif str(Ano) == 'nan':
        None
    else:
        Years.append(Ano)
        
Years = np.array(Years,np.int16) 
       

###### ------------------------------------------#############
import sys
sys.path.insert(0, r'C:\Users\Philipe Leal\Dropbox\Profissao\Python\Matplot_estudo_e_aplicacoes\Imagens')

from Adiciona_Ruler_e_North_arrow import scale_bar as Map_items




fig, Ax = plt.subplots(nrows=4,ncols=3, sharex='col', sharey='row',
                       subplot_kw={'projection': ccrs.Mercator()})
fig.suptitle('Prevalência da Hepatite-A por Região', fontsize=16)

# definindo Vmin e Vmax para garantir range entre todos os subplots:
    # para ajuste local por subplot, deletar Vmin e Vmax.
    # ex: https://gis.stackexchange.com/questions/273273/reducing-space-in-geopandas-and-matplotlib-pyplots-subplots


Vmin = SHP_joined['Prevalencias_relativas_%'].min()
Vmax = SHP_joined['Prevalencias_relativas_%'].max()


for i in range(len(Years)):
    Ano = Years[i]
    print(Ano)
    
    Axes = Ax.ravel()[i]
    
    
    Axes.set_extent([xmin, xmax, ymin, ymax], ccrs.Mercator())
    
    
    SHP_joined[SHP_joined['ANO']==Ano].plot(ax=Axes,
                                            column='Prevalencias_relativas_%', 
                                            legend=False,
                                            cmap='viridis',
                                            vmin=Vmin, vmax=Vmax,
                                            label=str(Ano))
    
    
    Axes.set_aspect('equal')
    Axes.set_title(str(Ano), fontsize=8)
    Axes.grid()
    
   # Map_items(Axes, ccrs.Mercator(), 100)

    
Axes11 = Ax.ravel()[11] 
Axes11.set_aspect('equal')
Axes11.grid()

Map_items(Axes11, ccrs.Mercator(), 100)

cax = fig.add_axes([0.9, 0.17, 0.02, 0.65])
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=Vmin, vmax=Vmax))
sm._A = []
cbar = fig.colorbar(sm, cax=cax)
cbar.ax.set_title('Prevalencia\n relativa (%)')
    

#im = plt.gca().get_children()[0]
#cax = fig.add_axes([0.90,0.1,0.03,0.8]) 
#fig.colorbar(im, cax=cax)


fig.subplots_adjust(top=0.855,
                    bottom=0.065,
                    left=1.21e-17,
                    right=0.850,
                    hspace=0.5,
                    wspace=0.005)



plt.show()



## Média de 10 anos:


df = pd.read_excel(file_name, sheet_name="SINAN_summary_DF")

df = df.rename(columns={'Regiões':'REGIAO'})

df.set_index('REGIAO', inplace=True)

df.rename({'Centro-Oeste':'Centro Oeste'}, axis=0, inplace=True)


SHP_path = r'C:\Doutorado\Tese\SHP\2017\UF\Merges\Brasil_UTF_8.shp'

SHP = gpd.read_file(SHP_path)

SHP.head()

SHP.set_index('REGIAO', inplace=True)

SHP_joined = SHP.join(df)



fig, Ax = plt.subplots(nrows=1,ncols=2, sharex='col', sharey='row')


fig.suptitle('Prevalência da Hepatite-A por Região', fontsize=16)
SHP_joined[SHP_joined['Date']=='1999-2016'].plot(ax=Ax[0], 
          column='Prevalência (%) por Região (1999-2016)', 
          legend=False,
          cmap=plt.cm.get_cmap('viridis', 10))


SHP_joined[SHP_joined['Date']==2017].plot(ax=Ax[1], 
          column='Prevalência (%) por Região (1999-2016)', 
          legend=False,
          cmap=plt.cm.get_cmap('viridis', 10))

im = plt.gca().get_children()[0]
cax = fig.add_axes([0.90,0.1,0.03,0.8]) # criando um subplot com as respectivas dimensões da figura
fig.colorbar(im, cax=cax) # adicionando à figura o subplot do colorbar

for i in range(len(Ax)):
    Ax[i].grid()
    Ax[i].set_aspect('equal')
Ax[0].set_title('1999-2016')
Ax[1].set_title('2017')

fig.subplots_adjust(top=0.860,
                    bottom=0.141,
                    left=0.096,
                    right=0.876,
                    hspace=0.31,
                    wspace=0.171)
plt.show()





#### Um colorbar per image:

fig, (Ax, Ax1) = plt.subplots(nrows=1,ncols=2, sharex='col', sharey='row')


fig.suptitle('Prevalência da Hepatite-A por Região', fontsize=16)


Ax.grid()
Ax.set_aspect('equal')

Ax1.grid()
Ax1.set_aspect('equal')
 

Mappable = SHP_joined[SHP_joined['Date']=='1999-2016'].plot(ax=Ax, 
          column='Prevalência (%) por Região (1999-2016)', 
          legend=False,
          cmap=plt.cm.get_cmap('viridis', 10))
Mappable.set_title('Prevalência (%) (1999-2016)')
fig = Mappable.get_figure()

# Posicao_x >>
# Posicao y >>
# largura das barras
cax = fig.add_axes([0.45, 0.2, 0.03, 0.5])

sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('viridis', 10))
sm._A = []
fig.colorbar(sm, cax=cax)

Mappable2 = SHP_joined[SHP_joined['Date']==2017].plot(ax=Ax1, 
          column='Prevalência (%) por Região (1999-2016)', 
          legend=False,
          cmap=plt.cm.get_cmap('viridis', 10))
Mappable2.set_title('Prevalência (%) (2017)')
fig2 = Mappable2.get_figure()
# Posicao_x >>
# Posicao y >>
# largura das barras
cax2 = fig2.add_axes([0.9, 0.2, 0.03, 0.5])

sm2 = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('viridis', 10))
sm2._A = []
fig.colorbar(sm2, cax=cax2)

fig.subplots_adjust(top=0.95,
                    bottom=0.0,
                    left=0.066,
                    right=0.886,
                    hspace=0.48,
                    wspace=0.45)
plt.show()



