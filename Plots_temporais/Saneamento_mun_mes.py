# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:06:02 2018

@author: Philipe Leal
"""

# Orientação quanto aos indicadores sanitários: http://www.diarioonline.com.br/add/pdf/relatorio-completoesgoto-18-08-2017-13-34-00.pdf


import matplotlib.pyplot as plt

import numpy as np
import matplotlib as mpl
import geopandas as gpd
import pandas as pd


import sys

sys.path.append(r'C:\Users\Philipe Leal\Dropbox\Profissao\Python\Matplot_estudo_e_aplicacoes\North_arrow')

from Add_north_arrow import North_arrow


###### Quebra de texto automatico:


import sys

sys.path.append(r'C:\Users\Philipe Leal\Dropbox\Profissao\Python\Matplot_estudo_e_aplicacoes')

from Quebra_automatica_de_titulos import split_title_line as QT



File_path = r'C:\Doutorado\Tese\SNIS\BD\Agregado_por_Municipio\PA\Agregado--2018-10-04--15-20-43.csv'
Para = pd.read_csv(File_path,  encoding='latin-1', na_values='', sep=';',decimal=',', skipfooter=1, engine='python')

Para.head()


############ Corrigindo dados:


Para['Código do Município '] = Para['Código do Município '].apply(str)

A = Para['AG006 - Volume de água produzido (1000 m³/ano)'].apply(float)
B =Para['AG007 - Volume de água tratada em ETAs (1000 m³/ano)'].apply(float)

C = pd.concat([A,B], axis=1)


def Percentual_agua_tratada_em_funcao_do_volume_produzido_MN(row):
    
    A = row['AG006 - Volume de água produzido (1000 m³/ano)'] # Volume anual de água disponível para consumo (tratada + não tratada),
    
    
    B = row['AG007 - Volume de água tratada em ETAs (1000 m³/ano)'] # Volume anual de água tratada
    if A != 0.0 or A != 0:
    
        C = (B/A)*100
        
    elif A == np.nan or B == np.nan:
        C = 0
        
    else:
        
        C = 0

    return C


Para['Percentual_agua_tratada_em_funcao_do_volume_produzido_MN'] = Para.apply(Percentual_agua_tratada_em_funcao_do_volume_produzido_MN, axis=1)

Para['Percentual_agua_tratada_em_funcao_do_volume_produzido_MN'].fillna(0, inplace=True)


Para = Para.rename(index=str, columns={'Ano de Referência ': 'Ano de Referência'})

Para.set_index('Ano de Referência', inplace=True)





def Percentual_agua_tratada_em_funcao_do_consumo_tot(row):
    
    # Volume anual de água consumido por todos os usuários, (volume
    # micromedido (AG008) mais o volume de consumo estimado (para as ligações desprovidas
    # de hidrômetro ou com hidrômetro parado). 
    
    A = row['AG010 - Volume de água consumido (1000 m³/ano)'] 
    

    B = row['AG007 - Volume de água tratada em ETAs (1000 m³/ano)'] # Volume anual de água tratada
    
    if A!= 0.0 or A!= 0:
    
        C = (B/A)*100
        
    elif A == np.nan or B == np.nan:
        C = 0
        
    else:
        
        C = 0

    return C



Para['Percentual_de_cobertura_dagua_tratada_em_f_do_consumo_tot_estimado'] = Para.apply(Percentual_agua_tratada_em_funcao_do_consumo_tot, axis=1)



### Atribuindo geometria aos dados:


SHP_path = r'C:\Doutorado\Tese\SHP\2017\Municipios\MUNICIPIOS_PARA.shp'

SHP = gpd.read_file(SHP_path)

SHP.keys()


SHP.set_index('CD_GEOCMU', inplace=True)
SHP = SHP.sort_index(ascending=True)

def f(row):
    
    row = str(row)
    
    row = row[0:6]
    print(row)
    
    return row



SHP['Código do Município'] = SHP.index.map(f)

SHP.keys()


SHP.reset_index(inplace=True)
SHP.set_index('Código do Município', inplace=True)

SHP.sort_index(inplace=True)

SHP.index



# Filtrando somente Parah:

SHP2 = SHP.loc[SHP.index.str[:2] == '15']



Para.reset_index(inplace=True)
Para.set_index('Código do Município ', inplace=True)




SHP2 = SHP2.merge(Para, left_index=True, right_index=True)

Keys_SHP2 = pd.Series(data=SHP2.keys())
Keys_SHP2.loc[81]


SHP2[str(Keys_SHP2.loc[81])] = SHP2[str(Keys_SHP2.loc[81])].fillna(0)    


SHP2 = SHP2.fillna(0)




############# Graficos:


################# Graficos ############################## --------------------------



### Create colormap custom:
from matplotlib.colors import LinearSegmentedColormap

colors = [(1,1,1), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0,0,0)]  # W -> B -> G -> R -> K
n_bins = 10
cmap_name = 'Phi_colormap'
cmap_phi = LinearSegmentedColormap.from_list(
        cmap_name, colors, N=n_bins)

plt.register_cmap(cmap = cmap_phi)

# 'Percentual anual de água não tratada disponibilizada à População em função do consumo total estimado

Para.reset_index(inplace=True)

Para.set_index('Ano de Referência', inplace=True)




    # Boxplot

import seaborn as sn

fig = plt.figure(figsize=(7, 4))
fig.suptitle('Pará', fontsize=16)
Ax2 = sn.boxplot(x=Para.index, y='Percentual_de_cobertura_dagua_tratada_em_f_do_consumo_tot_estimado',
           data=Para)
plt.xticks(rotation=90, fontsize=10)
Ticks = Ax2.get_yticks()
Ticks2 = []
for t in Ticks:
    Temp = str(t)[:-2]
    print(Temp)
    Temp.replace('.',',')

    Ticks2.append( Temp + '%')

Ax2.set_yticklabels(Ticks2)


Ax2.set_ylabel('Percentual anual de água tratada\ndisponibilizada à População\n em função do consumo total\nestimado')
fig.subplots_adjust(top=0.879,
                    bottom=0.219,
                    left=0.204,
                    right=0.965,
                    hspace=0.2,
                    wspace=0.2)
plt.savefig(r'C:\Doutorado\Tese\SNIS\BD\Agregado_por_Municipio\PA\Percentual_agua_tratada_disponibilizada_a_pop_em_funcao_do_consumo_total_municipal.png',dpi=600)
plt.show()



  # Plot Espacial
  
  

### Plotando


fig, Ax = plt.subplots(nrows=5, ncols=4, sharex='col', sharey='row')            
fig.suptitle('Percentual anual de água tratada disponibilizada\nà População em função do consumo total estimado', fontsize=12)
cmap = cmap_phi

Years = np.unique(SHP2['Ano de Referência'])


for i in range(np.size(Years)):
    Ano = Years[i]
    print(Ano)
    
    Axes = Ax.ravel()[i]
    
    
    
    Vmin = SHP2[SHP2[str(Keys_SHP2.loc[16])]==Ano][str(Keys_SHP2.loc[92])].min()
    Vmax = SHP2[SHP2[str(Keys_SHP2.loc[16])]==Ano][str(Keys_SHP2.loc[92])].max()
    
    if Vmax == 0:
        Vmax = 100
    
    Ticks_list, step = np.linspace(Vmin, Vmax, num=5, endpoint=True, retstep=True)
    Ticks_list = np.round(Ticks_list,2)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=Vmin-(step/2),vmax=Vmax+(step/2)))

    sm._A = []
    
    
    
    SHP.plot(ax= Axes, edgecolor='gray', facecolor='white', linewidth=0.23)
  
    SHP2[SHP2[str(Keys_SHP2.loc[16])]==Ano].plot(ax=Axes,
                                            column=str(Keys_SHP2.loc[92]), 
                                            legend=False,
                                            cmap=cmap,
                                            vmin=Vmin, vmax=Vmax,
                                            label=False)
    

    
    Axes.set_aspect('equal')
    Axes.set_title(str(Ano), fontsize=6)
    Axes.grid()
    Axes.tick_params(axis='both', which='major', labelsize=6)
        
    cbar = fig.colorbar(sm, ax=Axes, ticks=Ticks_list)   
    cbar.ax.tick_params(labelsize=4) 

    ticks = [t.get_text().replace('.',',')+'%' for t in cbar.ax.get_yticklabels()]
    
    cbar.set_ticklabels(ticks)
    

### Axes 11:

# mínimo e máximo da média temporal de todos os municípios
    
    
Vmin = SHP2.groupby(SHP2.index)[str(Keys_SHP2.loc[92])].mean().min()
Vmax = SHP2.groupby(SHP2.index)[str(Keys_SHP2.loc[92])].mean().max()

Axes11 = Ax.ravel()[-1]

Ticks_list, step = np.linspace(Vmin, Vmax, num=5, endpoint=True, retstep=True)
Ticks_list = np.round(Ticks_list,2)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=Vmin-(step/2),vmax=Vmax+(step/2)))

sm._A = []



SHP.plot(ax= Axes11, edgecolor='gray', facecolor='white', linewidth=0.23)

SHP2_temporal_mean = SHP2.groupby(SHP2.index)[str(Keys_SHP2.loc[92])].mean()
  

SHP2_temporal_mean = SHP.join(SHP2_temporal_mean)

SHP2_temporal_mean.plot(ax=Axes11,
                        edgecolor='gray',
                        linewidth=0.23,
                        column=str(Keys_SHP2.loc[92]), 
                        legend=False,
                        cmap=cmap,
                        vmin=Vmin, vmax=Vmax,
                        label=False)



Axes11.set_aspect('equal')
Axes11.set_title('Média:'+ '\n'+ '{0} - {1}: '.format(Years.min(), Years.max()), fontsize=6)
Axes11.grid()
Axes11.tick_params(axis='both', which='major', labelsize=6)
    
cbar = fig.colorbar(sm, ax=Axes11, ticks=Ticks_list)   
cbar.ax.tick_params(labelsize=4) 

ticks = [t.get_text().replace('.',',')+'%' for t in cbar.ax.get_yticklabels()]

cbar.set_ticklabels(ticks)
   
    
   
Arrow = North_arrow(Axes11, Arrow_location=(0.978, 0.008), Matplotlib_Transform=fig.transFigure, size=6)

fig.subplots_adjust(top=0.85,
                    bottom=0.06,
                    left=0.035,
                    right=0.95,
                    hspace=0.8,
                    wspace=0.32)

fig.savefig(r'C:\Doutorado\Tese\SNIS\BD\Agregado_por_Municipio\PA\Percentual_agua_tratada_disponibilizada_a_pop_em_funcao_do_consumo_total_municipal_espacializado.png',dpi=600)


plt.show()






#### ----------------------------------------------- #################


# 'IN046 - Índice de esgoto tratado referido à água consumida (percentual)'

# Esse indicador mostra, em relação à água consumida, qual porcentagem do
#esgoto é tratada. Quanto maior for essa porcentagem, melhor deve ser a colocação do
#município no Ranking, pois maior parte do esgoto gerado pelo município é tratada.

Para.reset_index(inplace=True)
Para.set_index('Ano de Referência', inplace=True)



    # Plot - Boxplot
    
    
import seaborn as sn


fig1 = plt.figure(figsize=(7, 4))
fig1.suptitle('Pará', fontsize=16)

Ax1 = sn.boxplot(x=Para.index, y=Keys_SHP2.loc[81], data=Para)
plt.xticks(rotation=90, fontsize=10)
Ax1.set_ylabel(QT(Keys_SHP2.loc[81], max_words=7))
Ticks = Ax1.get_yticks()
Ticks2 = []
for t in Ticks:
    Temp = str(t)[:-2]
    print(Temp)
    Temp.replace('.',',')

    Ticks2.append( Temp + '%')

Ax1.set_yticklabels(Ticks2)
fig1.subplots_adjust(top=0.874,
    bottom=0.2,
    left=0.168,
    right=0.974,
    hspace=0.2,
    wspace=0.2)
fig1.savefig(r'C:\Doutorado\Tese\SNIS\BD\Agregado_por_Municipio\PA\Indice_de_esgoto_tratado_referido_a_agua_consumida_percentual.png', dpi=600)
plt.show()









Para.reset_index(inplace=True)
Para.set_index('Município ', inplace=True)



    
    # Plot Espacial
    
Alpha = 0.8

Grid_line_width = 0.2

fig, Ax = plt.subplots(nrows=4, ncols=5, sharex='col', sharey='row',)                     
fig.suptitle(QT(Keys_SHP2.loc[81], max_words=7), fontsize=12)
cmap = cmap_phi

Years = np.unique(SHP2['Ano de Referência'])


for i in range(len(Years)):
    
    
    Ano = Years[i]
    print(Ano)
    
    Axes = Ax.ravel()[i]
    
    
    
    Vmin = SHP2[SHP2[str(Keys_SHP2.loc[16])]==Ano][str(Keys_SHP2.loc[81])].min()
    Vmax = SHP2[SHP2[str(Keys_SHP2.loc[16])]==Ano][str(Keys_SHP2.loc[81])].max()
    
    if Vmax == Vmin:
        Vmax = 100
    
    Ticks_list, step = np.linspace(Vmin, Vmax, num=5, endpoint=True, retstep=True)
    Ticks_list = np.round(Ticks_list,2)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=Vmin-(step/2),vmax=Vmax+(step/2)))

    sm._A = []
    

    SHP.plot(ax= Axes, edgecolor='gray', facecolor='white', linewidth=0.1)
  
    SHP2[SHP2[str(Keys_SHP2.loc[16])]==Ano].plot(ax=Axes,
                                            column=str(Keys_SHP2.loc[81]), 
                                            legend=False,
                                            cmap=cmap,
                                            vmin=Vmin, vmax=Vmax,
                                            label=False)
    

    
    Axes.set_aspect('equal')
    Axes.set_title(str(Ano), fontsize=8)
    Axes.grid(color='gray', alpha = Alpha, linewidth=Grid_line_width, linestyle='--')
    Axes.tick_params(axis='both', which='major', labelsize=6)
        
    cbar = fig.colorbar(sm, ax=Axes, ticks=Ticks_list)   
    cbar.ax.tick_params(labelsize=4) 

    ticks = [t.get_text().replace('.',',')+'%' for t in cbar.ax.get_yticklabels()]
    
    cbar.set_ticklabels(ticks)
    
    

### Axes 11:

# mínimo e máximo da média temporal de todos os municípios
    
    
Vmin = SHP2.groupby(SHP2.index)[str(Keys_SHP2.loc[81])].mean().min()
Vmax = SHP2.groupby(SHP2.index)[str(Keys_SHP2.loc[81])].mean().max()

Axes11 = Ax.ravel()[i+1]

Ticks_list, step = np.linspace(Vmin, Vmax, num=5, endpoint=True, retstep=True)
Ticks_list = np.round(Ticks_list,2)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=Vmin-(step/2),vmax=Vmax+(step/2)))

sm._A = []



SHP.plot(ax= Axes11, edgecolor='gray', facecolor='white', linewidth=0.23)

SHP2_temporal_mean = SHP2.groupby(SHP2.index)[str(Keys_SHP2.loc[81])].mean()
  

SHP2_temporal_mean = SHP.join(SHP2_temporal_mean)

SHP2_temporal_mean.plot(ax=Axes11,
                        edgecolor='gray',
                        linewidth=0.23,
                        column=str(Keys_SHP2.loc[81]), 
                        legend=False,
                        cmap=cmap,
                        vmin=Vmin, vmax=Vmax,
                        label=False)



Axes11.set_aspect('equal')
Axes11.set_title('Média:'+ '\n'+ '{0} - {1}: '.format(Years.min(), Years.max()), fontsize=7)
Axes11.grid(color='gray', alpha = Alpha, 
            linestyle='--',
            linewidth=Grid_line_width)

Axes11.tick_params(axis='both', which='major', labelsize=6)
    
cbar = fig.colorbar(sm, ax=Axes11, ticks=Ticks_list)   
cbar.ax.tick_params(labelsize=4) 

ticks = [t.get_text().replace('.',',')+'%' for t in cbar.ax.get_yticklabels()]

cbar.set_ticklabels(ticks)
   
       
Arrow = North_arrow(Axes11, Arrow_location=(0.978, 0.008), Matplotlib_Transform=fig.transFigure, size=6)


fig.subplots_adjust(top=0.82,
                    bottom=0.06,
                    left=0.02,
                    right=0.95,
                    hspace=0.65,
                    wspace=0.42)

fig.savefig(r'C:\Doutorado\Tese\SNIS\BD\Agregado_por_Municipio\PA\Indice_de_esgoto_tratado_referido_a_agua_consumida_percentual_espacial.png', dpi=900)

plt.show()








#### O SNIS define “População Total Atendida com Esgoto” como: valor da
#população urbana beneficiada com esgotamento sanitário pelo prestador de serviços, no
#último dia do ano de referência. Corresponde à população urbana que é efetivamente
#servida com os serviços. 

# 'IN024 - Índice de atendimento urbano de esgoto referido aos municípios atendidos com água (percentual)'


Para.reset_index(inplace=True)

Para.set_index('Ano de Referência', inplace=True)

    # plot Boxplot

import seaborn as sn


fig1 = plt.figure(figsize=(7, 4))
fig1.suptitle('Pará', fontsize=14)

Ax1 = sn.boxplot(x=Para.index, y=Keys_SHP2.loc[76], data=Para)
plt.xticks(rotation=90, fontsize=10)
Ax1.set_ylabel(QT(Keys_SHP2.loc[76], max_words=7))
Ticks = Ax1.get_yticks()
Ticks2 = []
for t in Ticks:
    Temp = str(t)[:-2]
    print(Temp)
    Temp.replace('.',',')

    Ticks2.append( Temp + '%')

Ax1.set_yticklabels(Ticks2)
fig1.subplots_adjust(top=0.789,
                    bottom=0.2,
                    left=0.168,
                    right=0.934,
                    hspace=0.2,
                    wspace=0.2)

fig1.savefig(r'C:\Doutorado\Tese\SNIS\BD\Agregado_por_Municipio\PA\Indice_de_atendimento_urbano_de_esgoto_refereido_aos_municipios_atendidos_com_agua_percentual.png', dpi=600)
plt.show()



    
    # Plot Espacial


fig, Ax = plt.subplots(nrows=4, ncols=5, sharex='col', sharey='row',) 
fig.suptitle(QT(Keys_SHP2.loc[76], max_words=7), fontsize=12)
cmap = cmap_phi

Years = np.unique(SHP2['Ano de Referência'])


for i in range(len(Years)):
    Ano = Years[i]
    print(Ano)
    
    Axes = Ax.ravel()[i]
    
    
    
    Vmin = SHP2[SHP2[str(Keys_SHP2.loc[16])]==Ano][str(Keys_SHP2.loc[76])].min()
    Vmax = SHP2[SHP2[str(Keys_SHP2.loc[16])]==Ano][str(Keys_SHP2.loc[76])].max()
    
    Ticks_list, step = np.linspace(Vmin, Vmax, num=5, endpoint=True, retstep=True)
    Ticks_list = np.round(Ticks_list,2)
    
          
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=Vmin-(step/2),vmax=Vmax+(step/2)))

    sm._A = []
    

    SHP.plot(ax= Axes, edgecolor='gray', facecolor='white', linewidth=0.1)
  
    SHP2[SHP2[str(Keys_SHP2.loc[16])]==Ano].plot(ax=Axes,
                                            column=str(Keys_SHP2.loc[76]), 
                                            legend=False,
                                            cmap=cmap,
                                            vmin=Vmin, vmax=Vmax,
                                            label=False)
    

    
    Axes.set_aspect('equal')
    Axes.set_title(str(Ano), fontsize=8)
    Axes.grid(color='gray', alpha = Alpha, linewidth=Grid_line_width, linestyle='--')
    Axes.tick_params(axis='both', which='major', labelsize=6)
        
    cbar = fig.colorbar(sm, ax=Axes, ticks=Ticks_list)   
    cbar.ax.tick_params(labelsize=4) 

    ticks = [t.get_text().replace('.',',')+'%' for t in cbar.ax.get_yticklabels()]
    
    cbar.set_ticklabels(ticks)
    
    

### Axes 11:

# mínimo e máximo da média temporal de todos os municípios
    
    
Vmin = SHP2.groupby(SHP2.index)[str(Keys_SHP2.loc[76])].mean().min()
Vmax = SHP2.groupby(SHP2.index)[str(Keys_SHP2.loc[76])].mean().max()

Axes11 = Ax.ravel()[i+1]

Ticks_list, step = np.linspace(Vmin, Vmax, num=5, endpoint=True, retstep=True)
Ticks_list = np.round(Ticks_list,2)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=Vmin-(step/2),vmax=Vmax+(step/2)))

sm._A = []



SHP.plot(ax= Axes11, edgecolor='gray', facecolor='white', linewidth=0.23)

SHP2_temporal_mean = SHP2.groupby(SHP2.index)[str(Keys_SHP2.loc[76])].mean()
  

SHP2_temporal_mean = SHP.join(SHP2_temporal_mean)

SHP2_temporal_mean.plot(ax=Axes11,
                        edgecolor='gray',
                        linewidth=0.23,
                        column=str(Keys_SHP2.loc[76]), 
                        legend=False,
                        cmap=cmap,
                        vmin=Vmin, vmax=Vmax,
                        label=False)



Axes11.set_aspect('equal')
Axes11.set_title('Média:'+ '\n'+ '{0} - {1}: '.format(Years.min(), Years.max()), fontsize=7)
Axes11.grid(color='gray', alpha = Alpha, 
            linestyle='--',
            linewidth=Grid_line_width)

Axes11.tick_params(axis='both', which='major', labelsize=6)
    
cbar = fig.colorbar(sm, ax=Axes11, ticks=Ticks_list)   
cbar.ax.tick_params(labelsize=4) 

ticks = [t.get_text().replace('.',',')+'%' for t in cbar.ax.get_yticklabels()]

cbar.set_ticklabels(ticks)
   
    

fig.subplots_adjust(top=0.82,
                    bottom=0.06,
                    left=0.02,
                    right=0.95,
                    hspace=0.65,
                    wspace=0.42)
  
Arrow = North_arrow(Axes11, Arrow_location=(0.978, 0.008), Matplotlib_Transform=fig.transFigure, size=6)


fig.savefig(r'C:\Doutorado\Tese\SNIS\BD\Agregado_por_Municipio\PA\Indice_de_esgoto_tratado_referido_a_agua_consumida_percentual_espacial.png', dpi=900)

plt.show()








####################### ------------------------------------ ###############################

# Percentual de água tratada em funcao do volume produzido:

Para.reset_index(inplace=True)

Para.set_index('Ano de Referência', inplace=True)

    # Boxplot
    

import seaborn as sn


fig1 = plt.figure(figsize=(7, 4))
fig1.suptitle('Pará', fontsize=16)

Ax1 = sn.boxplot(x=Para.index, y='Percentual_agua_tratada_em_funcao_do_volume_produzido_MN', data=Para)
plt.xticks(rotation=90, fontsize=10)
Ax1.set_ylabel('Percentual anual de água tratada disponibilizada \nà População em função do volume hídrico total\n disponibilizado')
Ticks = Ax1.get_yticks()
Ticks2 = []
for t in Ticks:
    Temp = str(t)[:-2]
    print(Temp)
    Temp.replace('.',',')

    Ticks2.append( Temp + '%')

Ax1.set_yticklabels(Ticks2)
fig1.subplots_adjust(top=0.874,
    bottom=0.2,
    left=0.168,
    right=0.974,
    hspace=0.2,
    wspace=0.2)
plt.show()








    # Espacial


fig, Ax = plt.subplots(nrows=4, ncols=5, sharex='col', sharey='row',) 
fig.suptitle(QT('Percentual anual de água tratada disponibilizada \nà População em função do volume hídrico total\n disponibilizado', max_words=5), fontsize=12)
cmap = mpl.cm.hot_r
cmap = cmap_phi
Years = np.unique(SHP2['Ano de Referência'])


for i in range(len(Years)):
    Ano = Years[i]
    print(Ano)
    
    Axes = Ax.ravel()[i]
    
    
    
    Vmin = SHP2[SHP2[str(Keys_SHP2.loc[16])]==Ano][str(Keys_SHP2.loc[91])].min()
    Vmax = SHP2[SHP2[str(Keys_SHP2.loc[16])]==Ano][str(Keys_SHP2.loc[91])].max()
    
    Ticks_list, step = np.linspace(Vmin, Vmax, num=5, endpoint=True, retstep=True)
    Ticks_list = np.round(Ticks_list,2)
    
          
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=Vmin-(step/2),vmax=Vmax+(step/2)))

    sm._A = []
    

    SHP.plot(ax= Axes, edgecolor='gray', facecolor='white', linewidth=0.1)
  
    SHP2[SHP2[str(Keys_SHP2.loc[16])]==Ano].plot(ax=Axes,
                                            column=str(Keys_SHP2.loc[91]), 
                                            legend=False,
                                            cmap=cmap,
                                            vmin=Vmin, vmax=Vmax,
                                            label=False)
    

    
    Axes.set_aspect('equal')
    Axes.set_title(str(Ano), fontsize=8)
    Axes.grid(color='gray', alpha = Alpha, linewidth=Grid_line_width, linestyle='--')
    Axes.tick_params(axis='both', which='major', labelsize=6)
        
    cbar = fig.colorbar(sm, ax=Axes, ticks=Ticks_list)   
    cbar.ax.tick_params(labelsize=4) 

    ticks = [t.get_text().replace('.',',')+'%' for t in cbar.ax.get_yticklabels()]
    
    cbar.set_ticklabels(ticks)
    
    

### Axes 11:

# mínimo e máximo da média temporal de todos os municípios
    
    
Vmin = SHP2.groupby(SHP2.index)[str(Keys_SHP2.loc[91])].mean().min()
Vmax = SHP2.groupby(SHP2.index)[str(Keys_SHP2.loc[91])].mean().max()

Axes11 = Ax.ravel()[i+1]

Ticks_list, step = np.linspace(Vmin, Vmax, num=5, endpoint=True, retstep=True)
Ticks_list = np.round(Ticks_list,2)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=Vmin-(step/2),vmax=Vmax+(step/2)))

sm._A = []



SHP.plot(ax= Axes11, edgecolor='gray', facecolor='white', linewidth=0.23)

SHP2_temporal_mean = SHP2.groupby(SHP2.index)[str(Keys_SHP2.loc[91])].mean()
  

SHP2_temporal_mean = SHP.join(SHP2_temporal_mean)

SHP2_temporal_mean.plot(ax=Axes11,
                        edgecolor='gray',
                        linewidth=0.23,
                        column=str(Keys_SHP2.loc[91]), 
                        legend=False,
                        cmap=cmap,
                        vmin=Vmin, vmax=Vmax,
                        label=False)



Axes11.set_aspect('equal')
Axes11.set_title('Média:'+ '\n'+ '{0} - {1}: '.format(Years.min(), Years.max()), fontsize=7)
Axes11.grid(color='gray', alpha = Alpha, 
            linestyle='--',
            linewidth=Grid_line_width)

Axes11.tick_params(axis='both', which='major', labelsize=6)
    
cbar = fig.colorbar(sm, ax=Axes11, ticks=Ticks_list)   
cbar.ax.tick_params(labelsize=4) 

ticks = [t.get_text().replace('.',',')+'%' for t in cbar.ax.get_yticklabels()]

cbar.set_ticklabels(ticks)
   
Arrow = North_arrow(Axes11, Arrow_location=(0.978, 0.008), Matplotlib_Transform=fig.transFigure, size=6)

fig.subplots_adjust(top=0.82,
                    bottom=0.06,
                    left=0.035,
                    right=0.95,
                    hspace=0.65,
                    wspace=0.42)

fig.savefig(r'C:\Doutorado\Tese\SNIS\BD\Agregado_por_Municipio\PA\Indice_de_esgoto_tratado_referido_a_agua_consumida_percentual_espacial.png', dpi=900)

plt.show()




