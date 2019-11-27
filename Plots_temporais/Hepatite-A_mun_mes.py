
# coding: utf-8

# In[10]:



import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.patches as mpatches
import numpy as np



file_name = r'C:\Doutorado\Tese\SINAN\Casos_hepatite_A_por_municipio_por_mes_ano\DATASET_FULL_YEARS.csv'

## Fluxo temporal 1 ano em 1 ano:


df = pd.read_csv(file_name, sep=';')

df.head()



# In[11]:


df.drop('Unnamed: 0', axis=1, inplace=True)

df.head()


# In[12]:


df['MUN_CODE'] = df['Municipio_de_residencia'].apply(lambda x: str(x).split(' ')[0])
df['Municipio_de_residencia'] = df['Municipio_de_residencia'].apply(lambda x: str(x).split(' ')[1:])
df['Municipio_de_residencia'] = df['Municipio_de_residencia'].apply(lambda x: ' '.join(x))
df.head()


# In[15]:


df.DATETIME = pd.DatetimeIndex(df.DATETIME)


df['ESTADO_CODE'] = df['MUN_CODE'].apply(lambda x: str(x)[:2])

df.set_index('ESTADO_CODE', inplace=True)


df.head()


# In[16]:


Estados_por_regiao = pd.read_excel(r'C:\Doutorado\Tese\IBGE_censo\Estados_por_regiao.xlsx',index_col='CODIGO_IBGE', na_values='-')

Estados_por_regiao.index = Estados_por_regiao.index.map(str)

Estados_por_regiao.sort_index(ascending=True, inplace=True)

print(Estados_por_regiao.index[0])
print(type(Estados_por_regiao.index[0]))


Estados_por_regiao.head()


# In[17]:


df = Estados_por_regiao.join(df)

df.head(15)


# In[18]:


df.DATETIME = pd.DatetimeIndex(df.DATETIME)

df.set_index('DATETIME', inplace=True)

df.head()


# # Resampling annuarly

# In[19]:


def Setting_NAN(x):
    if x=='-':
        x=np.nan
    return x

df = df.applymap(Setting_NAN) 

df.head()


# In[20]:


print("Info inicial: \n\n", df.info(), "\n\n\n")
df['N_Hepat_A_por_Mes'] = df['N_Hepat_A_por_Mes'].apply(float)
df['MUN_CODE'] = df['MUN_CODE'].apply(str)
df.info()


# In[21]:


df.reset_index(inplace=True)
df.set_index(['DATETIME', 'MUN_CODE'], inplace=True)
df.head()


# In[22]:
# Resample by year, aggregating by all values (all counties)

df.resample('Y',level=0, closed='left', label='right')['N_Hepat_A_por_Mes'].head()


# In[23]:


df.reset_index(inplace=True)
df.set_index(['DATETIME'], inplace=True)
df.head()


# In[46]:


import os


Ax = df[df['Abrev']=='PA'].pivot_table(values=['N_Hepat_A_por_Mes'], index='DATETIME', columns='Abrev', aggfunc='mean').plot(kind='line', grid=True, label=1, rot=90, legend=True)
plt.title("N casos de Hepatite-A do Pará")

Ax.grid(b=True, color='gray', alpha=0.5, linewidth=0.5)

Path = r'C:\Doutorado\Tese\SINAN\Casos_hepatite_A_por_municipio_por_mes_ano'

patch = mpatches.Patch(color='blue', label='Dados mensais')
plt.legend(handles=[patch])

plt.legend(['Dados Médios Mensais'])
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(Path, 'N casos de Hepatite-A do Parah.png'), dpi=600)


# In[55]:




fig, (ax, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10,5))

df[df['Abrev']=='PA'].pivot_table(values=['N_Hepat_A_por_Mes'], 
                                              columns='Abrev', 
                                              index='DATETIME', 
                                              aggfunc='mean').pct_change().plot(ax=ax,
                                                legend='Hepat - Percent Change')
Y_ticks = ax.get_yticks()
Y_ticks2 = []
for T in Y_ticks:
    T = str(T*100)
    T = T.replace('.', ',')
    
    T = T + '%'
    Y_ticks2.append(T)
    
ax.set_yticklabels(Y_ticks2)

ax.set_title('Pará', fontsize=16)
ax.set_ylabel("Mudança percentual")

Y_mean = df[df['Abrev']=='PA'].pivot_table(values=['N_Hepat_A_por_Mes'], 
                                              columns='Abrev', 
                                              index='DATETIME', 
                                              aggfunc='mean').pct_change().mean()

ax.hlines(Y_mean, xmin =df.index.min() , xmax =df.index.max() , colors='k', linestyles='--', 
          linewidth = 1.75, alpha = 0.6)

ax.legend(['Hepatite-A', 'Linha média'], fontsize=8, bbox_transform=ax.transAxes , bbox_to_anchor=(0.315, 0.9, 0, 0.102))



ax.grid(b=True, color='gray', alpha=0.5, linewidth=0.5)


### Axis 2

df[df['Abrev']=='PA'].pivot_table(values=['N_Hepat_A_por_Mes'], 
                                              columns='Abrev', 
                                              index='DATETIME', 
                                              aggfunc='mean').plot(ax= ax2, 
                                                legend=False)
Y_ticks = ax2.get_yticks()
Y_ticks2 = []
for T in Y_ticks:
    T = str(T)
    T = T.replace('.', ',')

    Y_ticks2.append(T)
    
ax2.set_yticklabels(Y_ticks2)


legend = ax2.legend(['Hepatite-A'], fontsize=8, 
                    bbox_transform=ax2.transAxes,
                    bbox_to_anchor=(0.315, 0.9, 0, 0.102))

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('white')

ax2.set_title('Pará', fontsize=16)
ax2.set_ylabel("N° de casos confirmados")

ax2.grid(b=True, color='gray', alpha=0.5, linewidth=0.5)


fig.subplots_adjust(top=0.917,
bottom=0.117,
left=0.088,
right=0.983,
hspace=0.2,
wspace=0.153)


plt.savefig(os.path.join(Path, 'Evolucao da Incidência de Hepatite-A no Pará.png'), dpi=600)
plt.show()


# In[56]:


Parah = df[df['Abrev']=='PA']

Parah.head(10)


# # Aucorrelação da série para o Pará:

# In[57]:


# autocorrelação temporal:
Auto_corr = []

for i in range(1,7):
    print(i)
    Multiplier = str(i * 6) + 'M'

    
    Parah_Resampled = Parah.resample(rule=Multiplier).last()
    Autocorr_Parah_percentage = Parah_Resampled['N_Hepat_A_por_Mes'].pct_change().autocorr()
    Autocorr_Parah = Parah_Resampled['N_Hepat_A_por_Mes'].autocorr()
    Multiplier = i * 6
    Auto_corr.append([Multiplier, float(Autocorr_Parah_percentage), float(Autocorr_Parah)])
    
Auto_corr = np.array(Auto_corr)


# In[83]:


Parah.groupby(Parah.index).sum()['N_Hepat_A_por_Mes'].head(10)


# # Teste de aleatoriedade da série via Autocorrelação:
# 
# 
# Autocorrelations should be near-zero for randomness. Such is not the case in this example and thus the randomness assumption fails
# 
# ## Inserindo coeficientes de confiança para a série:
# 
# If the autocorrelation plot is being used to test for randomness (i.e., there is no time dependence in the data), the following formula is recommended:
# 
# 
# (Z_(1 - a/2)/N**0.5
# 
#     Z1: função de distribuição acumulada da distribuição normal
#     
#     N = Tamanho da amostra (tamanho da série temporal)
# 
#     a: nível de significância
# 
#     Então para a = 95%, (Z_(1 - a/2) = 1.96
# 
#     Para a = 99, (Z_(1 - a/2) ~ 2.4
# 
# 

# # Referência: https://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm

# In[88]:


Z = 1.96

N = Parah.groupby(Parah.index).sum()['N_Hepat_A_por_Mes'].size
Xmin = Parah.groupby(Parah.index).sum()['N_Hepat_A_por_Mes'].min()
Xmax = Parah.groupby(Parah.index).sum()['N_Hepat_A_por_Mes'].max()
N

IC_upper = Z/(N**0.5)
IC_lower = -Z/(N**0.5)


# In[105]:


Ax[0]


# In[107]:


import matplotlib.mlab as mlab



Ax = plt.acorr(Parah.groupby(Parah.index).sum()['N_Hepat_A_por_Mes'], 
          normed=True, 
          detrend=mlab.detrend_linear,)

plt.hlines(IC_upper, np.min(Ax[0]), np.max(Ax[0]), linestyle = '--', alpha=0.8)

plt.hlines(IC_lower, np.min(Ax[0]), np.max(Ax[0]), linestyle = '--', alpha=0.8, label='a = 5%')
plt.legend()
plt.xlabel('Lag (n° meses)')
plt.ylabel("Autocorrelação")
plt.show()


# The autocorrelation plot can provide answers to the following questions:
# Are the data random?
# Is an observation related to an adjacent observation?
# Is an observation related to an observation twice-removed? (etc.)
# Is the observed time series white noise?
# Is the observed time series sinusoidal?
# Is the observed time series autoregressive?
# What is an appropriate model for the observed time series?
# Is the model
# Y = constant + error
# valid and sufficient?
# 
# Is the formula
# S =Desvio_padrao/√N
# valid?

# In[59]:


Auto_corr = pd.DataFrame(Auto_corr)

Auto_corr.columns=['Lag Mensal',  'Autocorr_PCT_CHANGE', 'Autocorr']

Auto_corr.set_index('Lag Mensal', inplace=True)

Auto_corr.head()


# In[65]:


fig, Ax = plt.subplots(figsize=(10,6))


Ax.set_title("Figura de autocorrelação \npor Percentage Change")

Auto_corr['Autocorr'].plot(kind='line', c='b', legend=True)
Auto_corr['Autocorr_PCT_CHANGE'].plot(kind='line', c='r', legend=True)
Ax.set_title("Figura de autocorrelação")
Ax.legend(['AC da incidência', 'AC da MP'])
Ax.set_xlabel('Lag (n° meses)')
plt.show()

fig.savefig(os.path.join(Path, 'Figura de autocorrelação.png'), dpi=600)


# In[111]:


Autocorr_RS = Parah['N_Hepat_A_por_Mes'].pct_change().autocorr()

print("Autocorrelação mensal de RS: ", Autocorr_RS)

print("")
print("")


Parah_2_Years = Parah.resample(rule='2M').last()

Autocorr_Parah_2_Years = Parah_2_Years['N_Hepat_A_por_Mes'].pct_change().autocorr()

print("Autocorrelação pct_change bimensal: ", Autocorr_Parah_2_Years)

Autocorr_Parah_2_Years = Parah_2_Years['N_Hepat_A_por_Mes'].autocorr()

print("Autocorrelação bimensal: ", Autocorr_Parah_2_Years)






Parah_Resampled = Parah.resample(rule='Y').last()

Autocorr_Parah_Year = Parah_Resampled['N_Hepat_A_por_Mes'].pct_change().autocorr()


print("Autocorrelação Anual: ", Autocorr_Parah_Year)


Parah_2_Years = Parah.resample(rule='2Y').last()

Autocorr_Parah_2_Years = Parah_2_Years['N_Hepat_A_por_Mes'].pct_change().autocorr()

print("Autocorrelação Bianual: ", Autocorr_Parah_2_Years)






Parah_4_Years = Parah.resample(rule='4Y').last()
Autocorr_Parah_4_Years = Parah_4_Years['N_Hepat_A_por_Mes'].pct_change().autocorr()


print("Autocorrelação de 4 em 4 anos: ", Autocorr_Parah_4_Years)


# In[ ]:


df_grouped = df.groupby(['Municipio_de_residencia'])

# Print dos primeiros 4 blocos do groupby (um bloco por município)

i=0
for Municipio, Municipio_df in df_grouped:
    if i<4:
        i+=1
        print('\n', Municipio)
        print(Municipio_df.head())
    else:
        break


# In[ ]:


# Soma de todos os casos para todo o período para apenas um grupo (municío ==Abadia dos Dourados):

print("Soma da serie histórica para 'Abadia dos Dourados': {0} \n\n".format(df_grouped.get_group('Abadia dos Dourados')['N_Hepat_A_por_Mes'].sum()))

## Historico:

df_grouped.get_group('Abadia dos Dourados')


# In[ ]:


df.groupby('MUN_CODE').resample('Y').sum().unstack().head()


# In[ ]:


df.groupby('MUN_CODE').resample('Y').sum().head()


# In[ ]:


df.pivot_table(values='N_Hepat_A_por_Mes', index=['MUN_CODE', df.index], margins=False,
               aggfunc={'N_Hepat_A_por_Mes':[np.sum, np.median]}).head()


# # Extraindo os máximos de cada grupo (municípios)
# 
# ## valores e respectivos anos com os máximos de hepatite-A

# In[ ]:




TF = df.pivot_table(values='N_Hepat_A_por_Mes', index=[df.index,'MUN_CODE'], aggfunc={'N_Hepat_A_por_Mes':[np.sum
                                                                                                     #,np.max,
                                                                                                     ]})


# In[ ]:


TF.head(5)


# In[ ]:


TF.rename(columns={'sum':'Hepa_A_Y_sum'}, inplace=True)

TF.head()


# In[ ]:


TF.reset_index(inplace=True)
TF.set_index('MUN_CODE', inplace=True)

TF.head()


# # Selecionando municípios apenas do Parah

# In[ ]:


TF_PARA = TF.loc[(TF.index.str.startswith('15')),:]

TF_PARA.head()


# In[ ]:


TF_PARA.reset_index(inplace=True)
TF_PARA.set_index('DATETIME', inplace=True)

TF_PARA.head()


# ## SOMA total de todos os eventos de hepa-A da série histórica para cada município:

# In[ ]:


TF_PARA.groupby('MUN_CODE').sum().unstack()


# # Setting index para adição de espacialidade:

# In[ ]:




TF_PARA.reset_index(inplace=True)
TF_PARA.set_index('MUN_CODE', inplace=True)

TF_PARA.head()


# # Adicionando espacialidade:

# In[ ]:



SHP_path = r'C:\Doutorado\Tese\SHP\2017\Municipios\MUNICIPIOS_PARA.shp'

SHP = gpd.read_file(SHP_path)

SHP.head()


# In[ ]:


SHP['CD_GEOCMU_6'] = SHP['CD_GEOCMU'].apply(lambda x: str(x[0:6]))

SHP.head()


# In[ ]:


SHP.set_index('CD_GEOCMU_6', inplace=True)
SHP.head(3)


# In[ ]:


SHP_Y_Hepat = SHP.join(TF_PARA)

SHP_Y_Hepat.head()


# # Resample de toda série Histórica:

# In[ ]:


SHP_Y_Hepat.reset_index(inplace=True)
SHP_Y_Hepat.head()

SHP_Y_Hepat.set_index(['index', 'NM_MUNICIP'] , inplace=True)

SHP_Y_Hepat.head()


# In[ ]:


df_year = pd.DataFrame(SHP_Y_Hepat.groupby(SHP_Y_Hepat.index)['Hepa_A_Y_sum'].sum())
df_year.head()


# In[ ]:


df_year['MUN_CODE6'] = df_year.index.map(lambda x: x[0])
df_year['MUNICIPIO_NAME'] = df_year.index.map(lambda x: x[1])
df_year.head()


# In[ ]:


df_year.set_index('MUN_CODE6', inplace=True)

df_year.head()


# In[ ]:


SHP_HEPAT_SUM_TIME_SERIES = SHP.join(df_year.Hepa_A_Y_sum)

SHP_HEPAT_SUM_TIME_SERIES.head()


# # Plottings:

# In[ ]:


Anos = np.unique(SHP_Y_Hepat.DATETIME.dt.year)

Anos = Anos[np.logical_not(np.isnan(Anos))]
print(Anos)
len(Anos)
print("Usar janela de plot 4x3")


# In[ ]:


### Create colormap custom:


def Custom_colorbar(n_bins=5, nome_para_a_colorbar = 'phi_colormap', Ramp_color_padrao = True):
    from matplotlib.colors import LinearSegmentedColormap

    if Ramp_color_padrao == True:

        colors = [(1,1,1), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0,0,0)]  # W -> B -> G -> Y -> R -> K

    else:

        colors = input("Insira uma lista de tuplas representando as cores em RGB para construção do RAMPCOLOR: ")




    n_bins = 50
    cmap_name = nome_para_a_colorbar
    cmap_phi = LinearSegmentedColormap.from_list(
            cmap_name, colors, N=n_bins)

    print("Nome da colorbar: {0}".format(cmap_name))
    
    return cmap_phi


cmap_phi = Custom_colorbar()


# In[ ]:


# Setting number of ticks type == 'ints' for the colorbar:


def N_ticks (Vmin, Vmax, K_ticks):
    X = np.arange(Vmin, Vmax, 1)
    
    X2 = X
    
    N=1
    while len(X2) >K_ticks:
        N+=1
        X2 = np.arange(Vmin, Vmax, N)
        step =X2[1] - X2[0]
        X2 = np.arange(Vmin, Vmax+step, N)
    
    print(X2,'\n')
    return X2

N_ticks(1, 50, 8)

    


# In[ ]:


### Um plot com todos os anos normalizados para apenas um colorbar para toda série historica:       

fig, Ax = plt.subplots(nrows=4,ncols=3, sharex='col', sharey='row',
                       )
fig.suptitle('Incidência de hepatite-A \npor município/ano', fontsize=16)

Font_size = 6

Vmin = SHP_Y_Hepat['Hepa_A_Y_sum'].min()
Vmax = SHP_Y_Hepat['Hepa_A_Y_sum'].max()
Linewidth = 0.25
Grid_line_width = 0.08
Alpha = 0.8
N_T = N_ticks(Vmin, Vmax, 8)


for i in range(len(Anos)):
    Ano = Anos[i]
    print(str(int(Ano)))
    
    Axes = Ax.ravel()[i]
    
    SHP.plot(ax=Axes, 
             edgecolor='black', 
             color='white',
             linewidth=Linewidth)
    
    SHP_Y_Hepat[SHP_Y_Hepat.DATETIME.dt.year==Ano].plot(ax=Axes,
                                                        column='Hepa_A_Y_sum', 
                                                        legend=False,
                                                        cmap=cmap_phi,
                                                        vmin=Vmin, vmax=Vmax,
                                                        label=str(int(Ano)),
                                                        edgecolor='black', linewidth=Linewidth)
    
    

    Axes.set_aspect('equal')
    Axes.set_title(str(int(Ano)), fontsize=Font_size)
    Axes.grid(color='gray', alpha = Alpha, linewidth=Grid_line_width)
    
    AXES_XTICKS = Axes.get_xticks()

    Axes.set_xticklabels(AXES_XTICKS, fontsize=Font_size, rotation=90)
    
    Axes.locator_params(nbins=3, axis='both') # quantos ticks por
    AXES_YTICKS = Axes.get_yticks()

    Axes.set_yticklabels(AXES_YTICKS, fontsize=Font_size)

    
## Plotting the last axes>


Axes11 = Ax.ravel()[11]

SHP.plot(ax=Axes11, 
             edgecolor='black', 
             color='white',
             linewidth=Linewidth)

SHP_HEPAT_SUM_TIME_SERIES.plot(ax=Axes11, 
                               column='Hepa_A_Y_sum', 
                               legend=False,
                               cmap=cmap_phi,
                               vmin=Vmin, vmax=Vmax,
                               label=str(int(Ano)),
                               edgecolor='black', linewidth=Linewidth)

AXES_XTICKS = Axes11.get_xticks()

Axes11.set_xticklabels(AXES_XTICKS, fontsize=Font_size, rotation=90)


AXES_YTICKS = Axes11.get_yticks()

Axes11.set_yticklabels(AXES_YTICKS, fontsize=Font_size)

Axes11.set_aspect('equal')
Axes11.grid(color='gray', alpha = Alpha, linewidth=Grid_line_width)
Axes11.set_title('Soma \n{0} - {1}'.format(str(int(Anos.min())), str(int(Anos.max()))), fontsize=Font_size)

Axes11.locator_params(nbins=3, axis='both')
cax = fig.add_axes([0.85, # posição xmin em X da FIG
                    0.10, # posição ymin em Y da FIG
                    0.02,  # altura da barra
                    0.65]) # largura da barra

Vmin = SHP_Y_Hepat['Hepa_A_Y_sum'].min()
Vmax = SHP_Y_Hepat['Hepa_A_Y_sum'].max()
    
sm = plt.cm.ScalarMappable(cmap=cmap_phi, norm=plt.Normalize(vmin=Vmin, vmax=Vmax))
sm._A = []
cbar = fig.colorbar(sm, cax=cax, ticks=N_T)
cbar.ax.set_ylabel('Incidência', rotation=90)
cbar.ax.set_yticklabels(labels=cbar.ax.yaxis.get_ticklabels(), fontdict= {'fontsize': Font_size})


fig.subplots_adjust(top=0.804,
                    bottom=0.09,
                    left=0.008,
                    right=0.88,
                    hspace=0.721,
                    wspace=0.0)
fig.savefig(r'C:\Doutorado\Tese\SINAN\Casos_hepatite_A_por_municipio_por_mes_ano\Incidencia por municipio por ano.png', dpi=900)


# In[ ]:


A = SHP_Y_Hepat.reset_index()
A.set_index('index', inplace=True)
A.groupby(A.index)['Hepa_A_Y_sum'].mean().plot()


# In[ ]:
# # Mesmo dado, mas com Colorbars específicos:

# In[ ]:


fig, Ax = plt.subplots(nrows=4,ncols=3, sharex='col', sharey='row',
                       )
fig.suptitle('Incidência de hepatite-A \npor município/ano', fontsize=16)


Font_size = 6
Linewidth = 0.25
Grid_line_width = 0.08
Alpha = 0.8



Vmin = SHP_Y_Hepat['Hepa_A_Y_sum'].min()
Vmax = SHP_Y_Hepat['Hepa_A_Y_sum'].max()

N_T = N_ticks(Vmin, Vmax, 8)




for i in range(len(Anos)):
    Ano = Anos[i]
    print('\n', str(int(Ano)))
    
    Axes = Ax.ravel()[i]
    
    
    
    Vmin = SHP_Y_Hepat[SHP_Y_Hepat.DATETIME.dt.year==Ano]['Hepa_A_Y_sum'].min()
    Vmax = SHP_Y_Hepat[SHP_Y_Hepat.DATETIME.dt.year==Ano]['Hepa_A_Y_sum'].max()  
    
    
    N_T = N_ticks(Vmin, Vmax, 4)

    
    SHP_Y_Hepat.plot(ax=Axes, 
             edgecolor='black', 
             color='white',
             linewidth=Linewidth)
    
    SHP_Y_Hepat[SHP_Y_Hepat.DATETIME.dt.year==Ano].plot(ax=Axes,
                                            column='Hepa_A_Y_sum', 
                                            legend=False,
                                            cmap=cmap_phi,
                                            vmin=Vmin, vmax=Vmax,
                                            label=False,
                                            edgecolor='black', 
                                            linewidth=Linewidth)
        
    Axes.set_aspect('equal')
    Axes.set_title(str(int(Ano)), fontsize=Font_size)
    Axes.grid(color='gray', alpha = Alpha, linewidth=Grid_line_width)
    
    AXES_XTICKS = Axes.get_xticks()

    Axes.set_xticklabels(AXES_XTICKS, fontsize=Font_size, rotation=90)
    
    Axes.locator_params(nbins=3, axis='both') # quantos ticks por
    AXES_YTICKS = Axes.get_yticks()

    Axes.set_yticklabels(AXES_YTICKS, fontsize=Font_size)
    
    
    # Colorbar

    sm = plt.cm.ScalarMappable(cmap=cmap_phi, norm=plt.Normalize(vmin=Vmin, vmax=Vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=Axes, ticks=N_T)
    #cbar.ax.set_ylabel('Incidência', rotation=90)
    cbar.ax.set_yticklabels(labels=cbar.ax.yaxis.get_ticklabels(), fontdict= {'fontsize': Font_size})


    
    
    
## Plotting the last axes>

Vmin = SHP_HEPAT_SUM_TIME_SERIES['Hepa_A_Y_sum'].min()
Vmax = SHP_HEPAT_SUM_TIME_SERIES['Hepa_A_Y_sum'].max()


Axes11 = Ax.ravel()[11]

SHP.plot(ax=Axes11, 
             edgecolor='black', 
             color='white',
             linewidth=Linewidth)

SHP_HEPAT_SUM_TIME_SERIES.plot(ax=Axes11, 
                               column='Hepa_A_Y_sum', 
                               legend=False,
                               cmap=cmap_phi,
                               vmin=Vmin, vmax=Vmax,
                               label=str(int(Ano)),
                               edgecolor='black', linewidth=Linewidth)

AXES_XTICKS = Axes11.get_xticks()

Axes11.set_xticklabels(AXES_XTICKS, fontsize=Font_size, rotation=90)


AXES_YTICKS = Axes11.get_yticks()

Axes11.set_yticklabels(AXES_YTICKS, fontsize=Font_size)

Axes11.set_aspect('equal')
Axes11.grid(color='gray', alpha = Alpha, linewidth=Grid_line_width)
Axes11.set_title('Soma \n{0} - {1}'.format(str(int(Anos.min())), str(int(Anos.max()))), fontsize=Font_size)

Axes11.locator_params(nbins=3, axis='both')


sm = plt.cm.ScalarMappable(cmap=cmap_phi, norm=plt.Normalize(vmin=Vmin, vmax=Vmax))
sm._A = []

N_T = N_ticks(Vmin, Vmax, 4)

cbar = fig.colorbar(sm, ax=Axes11, ticks=N_T)
#cbar.ax.set_ylabel('Incidência', rotation=90)
cbar.ax.set_yticklabels(labels=cbar.ax.yaxis.get_ticklabels(), fontdict= {'fontsize': Font_size})


fig.subplots_adjust(top=0.804,
                    bottom=0.09,
                    left=0.008,
                    right=0.88,
                    hspace=0.721,
                    wspace=0.0)

fig.savefig(r'C:\Doutorado\Tese\SINAN\Casos_hepatite_A_por_municipio_por_mes_ano\Incidencia por municipio por ano cbar unicos.png', dpi=900)


# # Uma figura por ano com respectivo colorbar:

# In[ ]:



Font_size = 6
Linewidth = 0.25
Grid_line_width = 0.08
Alpha = 0.8


for i in range(len(Anos)):
    Ano = Anos[i]
    print('\n', str(int(Ano)))
    
    


    fig, Axes = plt.subplots(nrows=1,ncols=1, sharex='col', sharey='row')
    fig.suptitle('Incidência da hepatite-A por município' + '\n' +'Ano: '+str(int(Ano)), fontsize=16)

    
       
    
    Vmin = SHP_Y_Hepat[SHP_Y_Hepat.DATETIME.dt.year==Ano]['Hepa_A_Y_sum'].min()
    Vmax = SHP_Y_Hepat[SHP_Y_Hepat.DATETIME.dt.year==Ano]['Hepa_A_Y_sum'].max()  
    
    
    N_T = N_ticks(Vmin, Vmax, 4)

    
    SHP_Y_Hepat.plot(ax=Axes, 
             edgecolor='black', 
             color='white',
             linewidth=Linewidth)
    
    SHP_Y_Hepat[SHP_Y_Hepat.DATETIME.dt.year==Ano].plot(ax=Axes,
                                            column='Hepa_A_Y_sum', 
                                            legend=False,
                                            cmap=cmap_phi,
                                            vmin=Vmin, vmax=Vmax,
                                            label=False,
                                            edgecolor='black', 
                                            linewidth=Linewidth)
        
    Axes.set_aspect('equal')
    Axes.set_title(str(int(Ano)), fontsize=Linewidth)
    Axes.grid(color='gray', alpha = Alpha, linewidth=Grid_line_width)
    
    
    # Colorbar

    sm = plt.cm.ScalarMappable(cmap=cmap_phi, norm=plt.Normalize(vmin=Vmin, vmax=Vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=Axes, ticks=N_T)
    #cbar.ax.set_ylabel('Incidência', rotation=90)
    cbar.ax.set_yticklabels(labels=cbar.ax.yaxis.get_ticklabels(), fontdict= {'fontsize': Font_size})


    

    fig.subplots_adjust(top=0.855,
                    bottom=0.065,
                    left=1.21e-17,
                    right=0.850,
                    hspace=0.5,
                    wspace=0.005)
                    
    Nome ='C:/Doutorado/Tese/SINAN/Casos_hepatite_A_por_municipio_por_mes_ano/Incidencia_por_municipio_ano/' + str(Ano) + '.png'
    
    plt.savefig(Nome, dpi=900)
    plt.show()


# # Um plot com a soma de todo o período

# In[ ]:


Init = int(Anos.min())
Fim = int(Anos.max())
fig, Axes = plt.subplots(nrows=1,ncols=1, sharex='col', sharey='row')

fig.suptitle('Incidência da hepatite-A por município'+ '\n' + 'Soma da série histórica: \n({0} - {1})'.format(Init, Fim), fontsize=15.5)



Font_size = 10
Linewidth = 0.55
Grid_line_width = 0.08
Alpha = 0.8





Vmin = SHP_HEPAT_SUM_TIME_SERIES['Hepa_A_Y_sum'].min()
Vmax = SHP_HEPAT_SUM_TIME_SERIES['Hepa_A_Y_sum'].max()

N_T = N_ticks(Vmin, Vmax, 6)

SHP.plot(ax=Axes, 
             edgecolor='black', 
             color='white',
             linewidth=Linewidth)

SHP_HEPAT_SUM_TIME_SERIES.plot(ax=Axes, 
                               column='Hepa_A_Y_sum', 
                               legend=False,
                               cmap=cmap_phi,
                               vmin=Vmin, vmax=Vmax,
                               label=str(int(Ano)),
                               edgecolor='black', linewidth=Linewidth)
AXES_XTICKS = Axes.get_xticks()

Axes11.set_xticklabels(AXES_XTICKS, fontsize=Font_size, rotation=90)


AXES_YTICKS = Axes.get_yticks()

Axes.set_yticklabels(AXES_YTICKS, fontsize=Font_size)

Axes.set_aspect('equal')
Axes.grid(color='gray', alpha = Alpha, linewidth=Grid_line_width)

Axes.locator_params(nbins=3, axis='both')


sm = plt.cm.ScalarMappable(cmap=cmap_phi, norm=plt.Normalize(vmin=Vmin, vmax=Vmax))
sm._A = []
cbar = fig.colorbar(sm, ax=Axes, ticks=N_T, ) #extend='max')
#cbar.ax.set_ylabel('Incidência', rotation=90)
cbar.ax.set_yticklabels(labels=cbar.ax.yaxis.get_ticklabels(), fontdict= {'fontsize': Font_size})


fig.subplots_adjust(top=0.804,
                    bottom=0.09,
                    left=0.008,
                    right=0.88,
                    hspace=0.721,
                    wspace=0.0)



Nome =r'C:\Doutorado\Tese\SINAN\Casos_hepatite_A_por_municipio_por_mes_ano\Incidencia_por_municipio_ano\Soma_todo_periodo.png'
    
#plt.savefig(Nome, dpi=900)
plt.show()

