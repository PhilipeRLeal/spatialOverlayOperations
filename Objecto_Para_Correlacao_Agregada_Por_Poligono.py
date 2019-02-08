



import pandas as pd
import numpy as np
pd.set_option("display.max_rows",10000)
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt
import os, sys
import geopandas as gpd
import matplotlib.patches as mpatches
from scipy.stats import pearsonr, spearmanr
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

class Correlacionador_e_Plotador_por_Poligono(object):
    def __init__(self, GDF):
        self.GDF = GDF
        self.__Corr_Pronto = None
        self.Alpha = 0.05

    def Correlacionador(self, DF, Nome_Param_1='EVI_250m_mean', Nome_Param_2='HAV'):

		
		
        return [np.round(self.Tipo_de_Correlacao(DF.loc[:,Nome_Param_1].values, DF.loc[:,Nome_Param_2].apply(lambda x: np.log(x+0.00001)).values), 10)]

    def Replacer(self, instance):
        instance['R2'] = instance['Pearson_R2'][0][0]
        instance['P_value'] = instance['Pearson_R2'][0][1]
		
		
        return instance

    def label(self, xy, text, Transform, y_delta=0.15, x_delta=0, Fontsize=12):
        y = xy[1] + y_delta  # shift y-value for label so that it's below the artist
        plt.text(xy[0]+x_delta, y, text, ha="center", family='sans-serif', size=Fontsize, transform=Transform)




    def Get_Correlation_from_Municipalities(self, 
											Groupby_parameter_name='GEOCODE_7', 
											Tipo_de_Correlacao='Pearson', 
											Alpha=0.05):
        """ 
		DF: Geodataframe contendo todos os dados já tabelados para correlação
		Groupby_parameter_name: nome do atributo para ser agrupado. Por padrão: 'GEOCODE_7'
		Tipo_de_Correlacao: Pearson ou Spearman. Por padrão, usar Pearson

        """
        
        if Tipo_de_Correlacao.lower().startswith('pe'):
            self.Tipo_de_Correlacao= pearsonr
        else:
            self.Tipo_de_Correlacao=spearmanr

        print("Eis as colunas para correlação: ", self.GDF.columns)
        Corr = self.GDF.groupby([str(Groupby_parameter_name)]).apply(func=self.Correlacionador, 
																	 Nome_Param_1=input('Insira o nome do Parametro 1 para correlacao: '),
																	 Nome_Param_2=input('Insira o nome do Parametro 2 para correlacao: '))

        Corr.name = 'Pearson_R2'
        self.Corr = pd.DataFrame(Corr)


        self.Corr = self.Corr.apply(self.Replacer, axis=1)


        self.Alpha = Alpha

        self.Corr.loc[self.Corr.P_value >self.Alpha, 'R2_masked' ] = np.nan

        try:
            
            if self.GDF.crs == None:
                print("Faltou a inserção de um CRS. Tentaremos inserir pelo padrão do geopandas.GeoDataFrame")
                self.Corr = gpd.GeoDataFrame(self.GDF.loc[:, [str(Groupby_parameter_name), 'geometry_G']].merge(self.Corr, left_on=[str(Groupby_parameter_name)], 
    					right_on=[str(Groupby_parameter_name)],
    					how='inner'), geometry='geometry_G')
            else:
                self.Corr = gpd.GeoDataFrame(self.GDF.loc[:, [str(Groupby_parameter_name), 'geometry_G']].merge(self.Corr, left_on=[str(Groupby_parameter_name)], 
    					right_on=[str(Groupby_parameter_name)],
    					how='inner'), geometry='geometry_G', crs=self.GDF.crs)
                
            print("\n\n" , "GeoDataFrame criado com sucesso\n\n")
        except:
            print("\n\n" , "Problemas na criação do GeoDataframe")

        self.__Corr_Pronto = 1

        self.Corr.fillna(0, inplace=True)

        return self.Corr

    def Print_Correlated_GDF(self, Transform=None, Projection=None,
							Suptitle='Correlação do EVI com o semi-log da incidência do HAV',
							Linewidth= 0.3,
							N_Ticks=5):
        """
		Parametros:
			GDF: geodataframe (geopandas)
			Transform: Geotransform instance from the cartopy relative to the GDF data
			Projection: projeção desejada
			Suptitle: Título da figura. Por padrão: 'Correlação do EVI com o semi-log da incidência do HAV'
			Linewidth: expessura da linha dos polígonos. Por padrão, 0.5
			
		Returna: lista com 2 objectos do plot
			[0]: Axes da figura
			[1]: A figura em si

        """
        
        if self.__Corr_Pronto is None:
            self.Get_Correlation_from_Municipalities()
        else:
            print("Continuando")
            None

        if Transform == None:

            self.Transform = ccrs.Geodetic(globe=ccrs.Globe(ellipse='GRS80')) # setting transform

        if Projection == None:
            self.projection=ccrs.PlateCarree() # setting projection


        self.fig, self.ax = plt.subplots(1,1, subplot_kw={'facecolor':'white', 
														  'projection': self.projection}) 
			
        self.fig.suptitle(Suptitle)


        Legend_KWDS ={'loc':(1.25, 0.18),'fontsize':8,
                  'bbox_transform':self.ax.transAxes,
                  'markerscale':0.75, # The relative size of legend markers compared with the originally drawn ones
                  'columnspacing':0.5, # The spacing between columns
                  'labelspacing':0.8, # The vertical space between the legend entries
                  'handletextpad':0.001} # float or None: The vertical space between the legend entries.

        self.Corr.plot(ax= self.ax, 
					   column='R2', legend=True, 
						facecolor='white', 
						edgecolor='k', 
						alpha=0.6, 
						linewidth=Linewidth,
						transform=self.Transform,
						legend_kwds=Legend_KWDS)


        xy =(0.15, 0.02)

        Nan_patch = mpatches.Patch(color='grey', label='P-valor > $alfa$: {0}'.format(self.Alpha))

        self.fig.legend(handles=[Nan_patch], loc=(xy[0], xy[1]), fontsize=7.5)

        self.Corr.loc[self.Corr['P_value']>self.Alpha].plot(ax=self.ax, 
                                                             legend=False, 
                                                             facecolor='grey', 
                                                             edgecolor='k', 
                                                             linewidth=Linewidth,
                                                             transform=self.Transform)

        self.Gridliner = self.ax.gridlines(crs=self.projection , draw_labels=True, linewidth=0.5, alpha=0.4, color='k', linestyle='--')
        
        
        self.Gridliner.top_labels = False
        self.Gridliner.right_labels = False
        self.Gridliner.xlabels_top = False
        self.Gridliner.ylabels_right = False
        if input("Quer converter os decimais '.' em ',': (S/N) ").lower() == 's':
			

            for i in range(len(self.fig.axes)):
                self.__Correcao_de_Ticks_para_Figuras_Brasileiras(axes_number=i)
        else:
            None


		### Adding North Arrow:
        self.Add_North_Arrow()
		## Adding scale bar:

        self.Add_scale_bar(self.ax, length=200, location=(0.5, -0.12))

        self.ax.xlabels_top = False
        self.ax.ylabels_right = False
        self.ax.ylabels_left = True
        self.ax.xlines = True
        self.ax.xformatter = LONGITUDE_FORMATTER
        self.ax.yformatter = LATITUDE_FORMATTER


        llx0, llx1, lly0, lly1 = self.ax.get_extent(ccrs.PlateCarree())

        self.ax.xlocator = mticker.FixedLocator(np.round(np.linspace(llx0, llx1, N_Ticks),0) , N_Ticks)
        self.ax.ylocator = mticker.FixedLocator(np.round(np.linspace(lly0, lly1, N_Ticks),0) , N_Ticks)


        self.ax.xlabel_style = {'size': 15, 'color': 'gray'}
        self.ax.xlabel_style = {'size': 15, 'color': 'gray'}

#        
#        BB = self.ax.get_position()
#        
#        Axes = self.fig.get_axes()
#
#        Colorbar = Axes[-1]
#        
#        xoc, xf = Colorbar.get_position().intervalx
#
#        Axes = self.fig.get_axes()
#        Colorbar = Axes[-1]
#        Bbox = Colorbar.get_position()
#        
#        xoc, xfc = Bbox.intervalx
#        ###########
#        
#        
#        
#        x0, xf = BB.intervalx
#        y0, yf = BB.intervaly
#        x_tail = xf
#        y_tail = y0
#        dx = xf - xoc
#        dy = yf - y0
#
#        Retangle = mpatches.Rectangle( (x_tail, y_tail), dx*2, dy,  fill=True, transform=self.fig.transFigure, facecolor='Blue', figure=self.fig)
#
#        self.fig.patches.extend([Retangle])
		
        if input("Quer salvar a figura?: (S/N)").lower() == 's':
            Path_fig_save = input("Insira o caminho do diretorio para salvar a figura: ")

            if os.path.exists(Path_fig_save) == False:
                os.mkdir(Path_fig_save)

            Nome_fig = input("Insira o nome da Figura para ser salva (ex: figura.png): ")

            self.fig.savefig(os.path.join(Path_fig_save, Nome_fig), dpi=900)
            print("\n\n", "Figura {0} salva ".format(Nome_fig), "\n\n")
        else:
            None


        return [self.fig, self.ax]

    def __Correcao_de_Ticks_para_Figuras_Brasileiras(self, axes_number=-1):
        """
		Parameter:
			axes_number: o número do axes da figura que será corrigida. 
				Por padrão é o fig.axes[-1] == (axes do colorbar).

        """
        self.Ticksbar = self.fig.axes[-1]


        self.Ticksbar_Y_legend = self.Ticksbar.get_yticklabels()

        for yi in self.Ticksbar_Y_legend:
            T = yi.get_text()
            T = T.replace('.',',')
            yi = yi.set_text(T)
			
            print(T)


        self.Ticksbar.set_yticklabels(self.Ticksbar_Y_legend)



        self.Ticksbar_X_legend = self.Ticksbar.get_xticklabels()

        for xi in self.Ticksbar_X_legend:
            T = xi.get_text()
            T = T.replace('.',',')
            xi = xi.set_text(T)
			
            print(T)


        self.Ticksbar.set_xticklabels(self.Ticksbar_X_legend)

		## Setting ticks from the Axis X and Y of the main Axes
        Xticks = self.ax.get_xticklabels()

        for i in Xticks:
            T = i.get_text()
            T = T.replace('.',',')
            i = i.set_text(T)
					
            print(T)
		    
        self.ax.set_xticklabels(Xticks)


        Yticks = self.ax.get_yticklabels()

        for i in Xticks:
            T = i.get_text()
            T = T.replace('.',',')
            i = i.set_text(T)
					
            print(T)
		    
        self.ax.set_yticklabels(Yticks)



    def Add_scale_bar(self, ax, length=None, location=(0.5, 0.01), linewidth=2.5):
        """
		ax is the axes to draw the scalebar on.
		Length is the length of the scalebar in km.
		Location is center of the scalebar in axis coordinates.
		(ie. 0.5 is the middle of the plot)
		linewidth is the thickness of the scalebar.
        """
        
        
		#Get the limits of the axis in lat long
        llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
		#Make tmc horizontally centred on the middle of the map,
		#vertically at scale bar location
        sbllx = (llx1 + llx0) / 2
        sblly = lly0 + (lly1 - lly0) * location[1]
        tmc = ccrs.TransverseMercator(sbllx, sblly)
		#Get the extent of the plotted area in coordinates in metres
        x0, x1, y0, y1 = ax.get_extent(tmc)
		#Turn the specified scalebar location into coordinates in metres
        sbx = x0 + (x1 - x0) * location[0]
        sby = y0 + (y1 - y0) * location[1]

		#Calculate a scale bar length if none has been given
		#(Theres probably a more pythonic way of rounding the number but this works)
        if not length: 
            length = (x1 - x0) / 5000 #in km
            ndim = int(np.floor(np.log10(length))) #number of digits in number
            length = round(length, -ndim) #round to 1sf
			#Returns numbers starting with the list
            def scale_number(x):
                if str(x)[0] in ['1', '2', '5']: return int(x)		
                else: return scale_number(x - 10 ** ndim)
            length = scale_number(length) 

		#Generate the x coordinate for the ends of the scalebar
        bar_xs = [sbx - length * 500, sbx + length * 500]
		
		#Plot the scalebar
        ax.plot(bar_xs, [sby, sby], 
				transform=tmc, 
				color='k',
				clip_on=False, 
				linewidth=linewidth,
				zorder=100)
		

		#Plot the scalebar label
        ax.text(sbx, sby, str(length) + ' km', 
				clip_on=False,  
				transform=tmc,
				horizontalalignment='center', 
				verticalalignment='bottom',
				zorder=100)
	
        ax.set_extent([llx0, llx1, lly0, lly1], crs=ccrs.PlateCarree())

    def Add_North_Arrow(self):
        x_tail = 0.975
        y_tail = 0.01
        x_head = 0.975
        y_head = 0.030
        dx = x_head - x_tail
        dy = y_head - y_tail

		
        arrow = mpatches.Arrow(x_tail, y_tail, dx, dy, width=0.05, transform=self.fig.transFigure, color='k', figure=self.fig)

        self.fig.patches.extend([arrow])
    
        plt.text(x_head, 
				 y_head + 0.8*(y_head - y_tail) , 
				s='N', 
				size=11, 
				ha='center', 
				va='center',
				color='K', 
				transform=self.fig.transFigure, 
				figure=self.fig)