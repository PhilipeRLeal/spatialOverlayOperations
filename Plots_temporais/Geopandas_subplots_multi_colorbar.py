# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 17:11:13 2018

@author: Philipe Leal
"""

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.patches as mpatches
import numpy as np


file_name = r'C:\Doutorado\Tese\SINAN\Casos_hepatite_A_por_estado_por_ano\Por_Regioes_BR_por_Ano.xlsx'

## Fluxo temporal 1 ano em 1 ano:


df = pd.read_excel(file_name, sheet_name='prevalencias', header=[1,2])


### Stacking:



stacked = df.stack()
stacked.reset_index(inplace=True)


stacked_keys = stacked.keys()

Keys_dict = {'level_0':'ANO', 'Ano':'REGIAO', 'REGIAO':'Prevalencias'}

stacked = stacked.rename(columns=Keys_dict)

stacked.set_index('ANO', inplace=True)




######## Plotting time series boxplot variation for Northern region:

Nordeste = stacked.loc[stacked['REGIAO']=='Nordeste']

## Figuras gráficas:


import seaborn as sn

plt.figure(figsize=(7, 4))

sn.lineplot(x=Nordeste.index, y="Prevalencias", data=Nordeste)
plt.xticks(rotation=90, fontsize=10)
plt.ylabel("Indice acumulado de esgoto\nsem tratamento")
plt.tight_layout()
plt.show()
#plt.savefig(r'C:\Doutorado\Tese\ANA\Esgotamento_Sanitario\Indice_acumulado_de_esgoto_sem_tratamento_por_UF.png')


## Testando para varias localizações:



stacked.reset_index(inplace=True)

stacked.set_index(["ANO"], inplace=True)

## Figuras gráficas:

Keys_dict2 = {'REGIAO':'Regiões'}

stacked = stacked.rename(columns=Keys_dict2)


import seaborn as sn

Fig = plt.figure(figsize=(7, 4))
Ax = Fig.add_subplot(111)
Ax = sn.lineplot(ax= Ax, x=stacked.index, y="Prevalencias", hue="Regiões", 
            data=stacked)


Ax_x_ticks = Ax.get_xticks()

Ax_x_ticks = list(Ax_x_ticks)
Ax_x_ticks2 = []
for tick in Ax_x_ticks:
    tick = str(tick)
    tick = tick.replace('.0','')
    tick = tick.replace('.5','')
    print(tick)
    Ax_x_ticks2.append(tick)
Ax_x_ticks = Ax_x_ticks2
del(Ax_x_ticks2)
Ax.set_xticklabels(Ax_x_ticks, rotation=90, fontsize=10)
Ax_y_ticks = Ax.get_yticks()
Ax.set_yticklabels(Ax_y_ticks*10000)
plt.ylabel("Indice acumulado de esgoto\nsem tratamento\n*10.000")
plt.tight_layout()
plt.show()



Keys_dict3 = {'Regiões':'REGIAO'}

stacked = stacked.rename(columns=Keys_dict3)


stacked.reset_index(inplace=True)
stacked.set_index('REGIAO', inplace=True)


Keys_dict_index = {'Centro-Oeste': 'Centro Oeste'}

stacked = stacked.rename(index=Keys_dict_index)

# Filtrando apenas os anos acima de 2006:
stacked = stacked[stacked['ANO'] >= 2007]


stacked['Prevalencias_relativas_%'] = stacked['Prevalencias']/np.sum(stacked['Prevalencias'])*100


SHP_path = r'C:\Doutorado\Tese\SHP\2017\UF\Merges\Brasil_UTF_8.shp'

SHP = gpd.read_file(SHP_path)

SHP.head()


SHP.set_index('REGIAO', inplace=True)


SHP_joined = SHP.join(stacked)

SHP_joined = SHP_joined[SHP_joined['ANO'] >=2007]


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



### Um plot com todos os anos normalizados para apenas um colorbar para de toda série historica:       

fig, Ax = plt.subplots(nrows=4,ncols=3, sharex='col', sharey='row',
                       )
fig.suptitle('Prevalência da Hepatite-A por Região', fontsize=16)


Vmin = SHP_joined['Prevalencias_relativas_%'].min()
Vmax = SHP_joined['Prevalencias_relativas_%'].max()


for i in range(len(Years)):
    Ano = Years[i]
    print(Ano)
    
    Axes = Ax.ravel()[i]
    
    SHP_joined[SHP_joined['ANO']==Ano].plot(ax=Axes,
                                            column='Prevalencias_relativas_%', 
                                            legend=False,
                                            cmap='viridis',
                                            vmin=Vmin, vmax=Vmax,
                                            label=str(Ano))
    Axes.set_aspect('equal')
    Axes.set_title(str(Ano), fontsize=8)
    Axes.grid()

Axes11 = Ax.ravel()[11] 
Axes11.set_aspect('equal')
Axes11.grid()
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






### Um plot com todos os anos para de toda série historica:       

import matplotlib as mpl


fig, Ax = plt.subplots(nrows=4,ncols=3, sharex='col', sharey='row',
                       )
fig.suptitle('Prevalência da Hepatite-A por Região', fontsize=16)
cmap = mpl.cm.plasma

for i in range(len(Years)):
    Ano = Years[i]
    print(Ano)
    
    Axes = Ax.ravel()[i]
    
    
    
    Vmin = SHP_joined[SHP_joined['ANO']==Ano]['Prevalencias_relativas_%'].min()
    Vmax = SHP_joined[SHP_joined['ANO']==Ano]['Prevalencias_relativas_%'].max()
    
    Ticks_list, step = np.linspace(Vmin, Vmax, num=4, endpoint=True, retstep=True)
    Ticks_list = np.round(Ticks_list,2)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=Vmin-(step/2),vmax=Vmax+(step/2)))

    sm._A = []
    
    
    
  
    SHP_joined[SHP_joined['ANO']==Ano].plot(ax=Axes,
                                            column='Prevalencias_relativas_%', 
                                            legend=False,
                                            cmap=cmap,
                                            vmin=Vmin, vmax=Vmax,
                                            label=False)
    
    Axes.set_aspect('equal')
    Axes.set_title(str(Ano), fontsize=8)
    Axes.grid()
    cbar = fig.colorbar(sm, ax=Axes, ticks=Ticks_list)   

    cbar.set_ticklabels(Ticks_list)

### Adding a mean of the whole period:
    


SHP_Historical_mean = SHP_joined.groupby('UF').mean()


SHP.reset_index(inplace=True)

SHP.set_index('UF', inplace=True)

SHP_Historical_mean = SHP.merge(SHP_Historical_mean, on='UF')

    


    
Vmin = SHP_Historical_mean['Prevalencias_relativas_%'].min()
Vmax = SHP_Historical_mean['Prevalencias_relativas_%'].max()

Ticks_list, step = np.linspace(Vmin, Vmax, num=4, endpoint=True, retstep=True)
Ticks_list = np.round(Ticks_list,2)    
    
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=Vmin-(step/2),vmax=Vmax+(step/2)))

sm._A = []



Axes11 = Ax.ravel()[11] 
Axes11.set_aspect('equal')
Axes11.grid()

Tmin = SHP_joined['ANO'].min()
Tmax = SHP_joined['ANO'].max()

Axes11.set_title("Média Temporal\n({0}-{1})".format(Tmin, Tmax), fontsize=8)
SHP_Historical_mean.plot(ax=Axes11,
                column='Prevalencias_relativas_%', 
                legend=False,
                cmap=cmap,
                vmin=Vmin, vmax=Vmax,
                label=False)
fig.colorbar(sm, ax=Axes11, ticks=Ticks_list)   
cbar.set_ticklabels(Ticks_list)
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














### Um plot por ano com respectivo colorbar:


for i in range(len(Years)):

    fig, Ax = plt.subplots(nrows=1,ncols=1, sharex='col', sharey='row')
    fig.suptitle('Prevalência da Hepatite-A por Região', fontsize=16)


    Ano = Years[i]
    print(Ano)
    
    Axes = Ax
    SHP_Ano = SHP_joined[SHP_joined['ANO']==Ano]
    SHP_Ano.plot(ax=Axes,
                 column='Prevalencias_relativas_%', 
                 legend=False,
                 cmap='viridis',
                 #vmin=Vmin, vmax=Vmax,
                 label=str(Ano))
    Axes.set_aspect('equal')
    Axes.set_title(str(Ano), fontsize=8)
    Axes.grid()
    
#im = plt.gca().get_children()[0]
#cax = fig.add_axes([0.90,0.1,0.03,0.8]) 
#fig.colorbar(im, cax=cax)

    fig.subplots_adjust(top=0.855,
                    bottom=0.065,
                    left=1.21e-17,
                    right=0.850,
                    hspace=0.5,
                    wspace=0.005)
    Nome = 'C:\Doutorado\Tese\SINAN\Casos_hepatite_A_por_estado_por_ano\Ano_a_Ano\Casos_hepatite_A_por_estado_para_' +str(Ano) +'.png'
    
    
    plt.savefig(Nome, dpi=400)
    plt.show()

# Um plot com a média de todo o período;


SHP_Historical_mean = SHP_joined.groupby('UF').mean()

SHP_Historical_mean.set_index('UF', inplace=True)

SHP.reset_index(inplace=True)

SHP.set_index('UF', inplace=True)

SHP_Historical_mean = SHP.merge(SHP_Historical_mean, on='UF')




fig, Ax = plt.subplots(nrows=1,ncols=1, sharex='col', sharey='row')
fig.suptitle('Prevalência da Hepatite-A por Região\nda Média Histórica (2007-2017)', fontsize=16)


SHP_Historical_mean.plot(ax=Ax,
             column='Prevalencias_relativas_%', 
             legend=True,
             cmap='viridis',
             #vmin=Vmin, vmax=Vmax,
             label=False)
Ax.set_aspect('equal')

Ax.grid()


cbar = fig.colorbar(sm, cax=cax)
cbar.ax.set_title('Prevalencia\n relativa (%)')


#im = plt.gca().get_children()[0]
#cax = fig.add_axes([0.90,0.1,0.03,0.8]) 
#fig.colorbar(im, cax=cax)

fig.subplots_adjust(top=0.82,
                    bottom=0.065,
                    left=0.0,
                    right=0.85,
                    hspace=0.5,
                    wspace=0.005)


Nome = 'C:\Doutorado\Tese\SINAN\Casos_hepatite_A_por_estado_por_ano\Ano_a_Ano\Casos_hepatite_A_por_estado_Media_historica.png'

    
plt.savefig(Nome, dpi=400)

plt.show()




## Média de 10 anos:


df = pd.read_excel(file_name, sheet_name="SINAN_summary_DF")

df = df.rename(columns={'Regiões':'REGIAO'})

df.set_index('REGIAO', inplace=True)
df = df.rename(index={'Centro-Oeste':'Centro Oeste'})

SHP_path = r'C:\Doutorado\Tese\SHP\2017\UF\Merges\Brasil_UTF_8.shp'

SHP = gpd.read_file(SHP_path)

SHP.head()

SHP.set_index('REGIAO', inplace=True)

SHP_joined = SHP.join(df)

SHP_joined[SHP_joined['Date']=='1999-2016'].head()


SHP_joined.set_index('Date', inplace=True)


SHP_joined.index = SHP_joined.index.map(str)
SHP_joined.sort_index(ascending=True, inplace=True)
Serie = pd.Series(SHP_joined.loc['1999-2016', 'Prevalência (%) por Região (1999-2016)']/(2016-1999), name='taxa prevalencia (1999-2016)')

SHP_joined.loc['1999-2016', 'taxa prevalencia (1999-2016)']= Serie.values

SHP_joined = SHP_joined.to_crs({'init': 'epsg:4326'})





import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
proj = ccrs.PlateCarree(central_longitude=0)

fig, Ax = plt.subplots(nrows=1,ncols=2, sharex='col', sharey='row', subplot_kw=dict(projection=proj))

fig.suptitle('Prevalência da Hepatite-A por Região', fontsize=16)

SHP_joined.loc['1999-2016'].plot(ax=Ax[0], 
          column='taxa prevalencia (1999-2016)', 
          legend=False,
          cmap=plt.cm.get_cmap('viridis', 10))


SHP_joined.loc['2017'].plot(ax=Ax[1],  
          column='Prevalência (%) por Região (1999-2016)',
          legend=False,
          cmap=plt.cm.get_cmap('viridis', 10))

im = plt.gca().get_children()[0]

cax = fig.add_axes([0.90,0.24,0.03,0.524]) 
fig.colorbar(im, cax=cax)

for i in range(len(Ax)):
    Ax[i].grid()
    Ax[i].set_aspect('equal')
    gl = Ax[i].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.35, linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right=False
    gl.xlines = True
    if i > 0:
        gl.ylabels_left=False
    xmin, ymin, xmax, ymax = SHP_joined.total_bounds
    X = np.round(np.linspace(xmin-10, xmax+10, 6),0)
    gl.xlocator = mticker.FixedLocator(X)
    
    Y = np.round(np.linspace(ymin-10, ymax+10, 6),0)
    
    gl.ylocator = mticker.FixedLocator(Y)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black', 
                       #'weight': 'bold', 
                       'rotation':90}
    
        
Ax[0].set_title('1999-2016')

Ax[1].set_title('2017')

# Setting gridlines:



fig.subplots_adjust(top=0.99,
                    bottom=0.016,
                    left=0.066,
                    right=0.876,
                    hspace=0.285,
                    wspace=0.066)
plt.show()

fig.savefig(r'C:\Doutorado\Tese\SINAN\Casos_hepatite_A_por_estado_por_ano\1999-2016_and_2017.png')