



import pandas as pd
import numpy as np
pd.set_option("display.max_rows",10000)
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import matplotlib.patches as mpatches
from scipy.stats import pearsonr, spearmanr
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy



class Correlacionador_e_Plotador_por_Poligono(object):
    def __init__(self, gdf, figure_path_saver=None, figura_em_Ingles='N'):
        """
        Esta classe de objeto realiza a análise de correlação município específica com base na série histórica dos dados fornecidos.
        Esta classe também plota esta análise de correlação.
        
        Parâmetros de entrada:
            - GDF: geodataframe contendo os dados a serem correlacionados
            
            - figure_path_saver (por padrão==None): caminho a ser utilizado para save da figura gerada
                Se a classe for instanciada com figure_path_saver definido como uma string, ele se utilizará desta String para salvar a figura automaticamente
            
            - Figura_em_Ingles (por padrão =='N'): se deixado como padrão, a figura terá formato brasileiro (decimal == ',') e algumas legendas ('p-valor', 'alfa') em português
        
        """
        
        
        self.GDF = gdf
        self.__Corr_Pronto = None
        self.Alpha = 0.05
        self.Figura_em_Ingles = figura_em_Ingles
        
        self.definindo_locale_Matplotlib(self.Figura_em_Ingles)
        
        self.figure_path_saver(figure_path_saver)
    
    
    
    def figure_path_saver(self, path):
        """
        
        Function that sets the path into which the figure will be saved
        
        """
        
        self.__Figure_Path_saver = path        
        
    def definindo_locale_Matplotlib(self, internacional='N'):
        """
        Function that defines the locale.setlocale module
        
        """
        
        
        
        if internacional=='N':
            import locale
            try:
                locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
            except:
                locale.setlocale(locale.LC_ALL, "Portuguese_Brazil.1252")
            import matplotlib as mpl
            mpl.rcParams['axes.formatter.use_locale'] = True
        
        else:
            None
    
    
    def correlacionador(self, df, nome_param_1='EVI_250m_mean', nome_param_2='HAV', decisao=2):
        '''
        decisao: Define em qual variável será aplicada a transformação pelo log
                0: nenhuma
                1: Parâmetro 1
                2: Parâmetro 2
                3: Ambos parâmetros
                
            Por Padrão, decisao = 2
        
        '''
        
        DF = df
        
        if decisao == 0:
            return [np.round(self.Tipo_de_Correlacao(DF.loc[:,nome_param_1].values, DF.loc[:,nome_param_2].values), 10)]  
        elif decisao == 1:
            return [np.round(self.Tipo_de_Correlacao(DF.loc[:,nome_param_1].apply(lambda x: np.log(x+self.Semi_log_Adder)).values, DF.loc[:,nome_param_2].values), 10)]
        elif decisao == 2:
            return [np.round(self.Tipo_de_Correlacao(DF.loc[:,nome_param_1].values, DF.loc[:,nome_param_2].apply(lambda x: np.log(x+self.Semi_log_Adder)).values), 10)]
        else:
            return [np.round(self.Tipo_de_Correlacao(DF.loc[:,nome_param_1].apply(lambda x: np.log(x+self.Semi_log_Adder)).values, DF.loc[:,nome_param_2].apply(lambda x: np.log(x+self.Semi_log_Adder)).values), 10)]
        
    def replacer(self, instance):
        instance['R2'] = instance['Pearson_R2'][0][0]
        instance['P_value'] = instance['Pearson_R2'][0][1]
		
		
        return instance

    def label(self, xy, text, transform, y_delta=0.15, x_delta=0, Fontsize=12):
        """
        
        Function that inserts label to the map figure
        
        """
        
        
        y = xy[1] + y_delta  # shift y-value for label so that it's below the artist
        plt.text(xy[0]+x_delta, y, text, ha="center", family='sans-serif', size=Fontsize, transform=transform)




    def get_Correlation_from_Municipalities(self, 
											groupby_parameter_name='GEOCODE_7', 
											tipo_de_Correlacao='Pearson', 
											alpha=0.05,
                                            Geometry_column=None,
                                            nome_param_1=None,
                                            nome_param_2 = None,
                                            decisao=None):
        """ 
		DF: Geodataframe contendo todos os dados já tabelados para correlação
		groupby_parameter_name: nome do atributo para ser agrupado. Por padrão: 'GEOCODE_7'
		tipo_de_Correlacao: Pearson ou Spearman. Por padrão, usar Pearson
        
        decisao: define para quais variáveis dentre nome_param_1 e nome_param_2 devem ter seus dados transformados pelo semi-log
           \n [1] para: nome_param_1 ;\n [2] para: nome_param_2 ;\n [3] para Ambas: (nome_param_1, nome_param_2) ;\n [0] para Nenhuma

        """
        
        if hasattr(self.GDF, 'geometry'):
                
                self.Geometry_column = 'geometry'
                
        else:
            None
        
        if hasattr(self, 'Geometry_column') == False and Geometry_column == None:
            print("\n\n","Eis as colunas do GDF para seleção do geometry_column:", self.GDF.columns)
            
            self.Geometry_column = input("Selecione entre as opções dadas acima, qual deve ser utilizada como geometry para reconstrução do GeodataFrame: " )
        
        else:
            if hasattr(self, 'Geometry_column') == False:
                self.Geometry_column = Geometry_column
            else:
                Geometry_column = self.Geometry_column
        
        
        # Definindo o tipo de correlação que será utilizada:
        
        
        if tipo_de_Correlacao.lower().startswith('pe'):
            self.Tipo_de_Correlacao= pearsonr
        else:
            self.Tipo_de_Correlacao=spearmanr
        
        
        if nome_param_1 == nome_param_2 == None:
        
            print("Eis as colunas para correlação: ", self.GDF.columns)
        
        else:
            None
        
        
        if nome_param_1 == None or nome_param_2 == None:
            self.__nome_param_1=input('Insira o nome do Parametro 1 para correlacao: ')
            self.__nome_param_2=input('Insira o nome do Parametro 2 para correlacao: ')
        
        
        else:
            self.__nome_param_1 = nome_param_1
            self.__nome_param_2 = nome_param_2
        
        if decisao==None:
        
            self.__decisao = int(input("Quais variáveis devem ter seus dados transformados pelo semi-log? \n [1] para: {0} ;\n [2] para: {1} ;\n [3] para Ambas: ({0}, {1}) ;\n [0] para Nenhuma: \n".format(self.__nome_param_1 , self.__nome_param_2)))
        else:
            self.__decisao = decisao
            
        if self.__decisao != 0:
            print("Valor padrão do Semi-log: {0}".format(0.00001))
            if input("Usar o valor padrão para a transformação de semi-log (S/N)? ").lower() == 's':
                self.Semi_log_Adder = 0.00001
                
            
            else:
                self.Semi_log_Adder = float(input('Defina o valor a ser adicionado na funcao semi-log: '))
        
        
        Corr = self.GDF.groupby([str(groupby_parameter_name)]).apply(func=self.correlacionador, 
																	 nome_param_1 = self.__nome_param_1,
																	 nome_param_2 = self.__nome_param_2,
                                                                     decisao = self.__decisao)
        

        Corr.name = 'Pearson_R2'
        self.Corr = pd.DataFrame(Corr)


        self.Corr = self.Corr.apply(self.replacer, axis=1)


        self.Alpha = alpha

        self.Corr.loc[self.Corr.P_value >self.Alpha, 'R2_masked' ] = np.nan

        try:
            
            if hasattr(self.GDF, 'crs') == False:
                print("Faltou a inserção de um CRS. Tentaremos inserir pelo padrão do geopandas.GeoDataFrame")
                self.Corr = gpd.GeoDataFrame(self.GDF.loc[:, [str(groupby_parameter_name), self.Geometry_column]].merge(self.Corr, left_on=[str(groupby_parameter_name)], 
    					right_on=[str(groupby_parameter_name)],
    					how='inner'), geometry=self.Geometry_column)
            else:
                self.Corr = gpd.GeoDataFrame(self.GDF.loc[:, [str(groupby_parameter_name), self.Geometry_column]].merge(self.Corr, left_on=[str(groupby_parameter_name)], 
    					right_on=[str(groupby_parameter_name)],
    					how='inner'), geometry=self.Geometry_column, crs=self.GDF.crs)
                
            print("\n\n" , "GeoDataFrame criado com sucesso\n\n")
        except:
            print("\n\n" , "Problemas na criação do GeoDataframe")

        self.__Corr_Pronto = 1

        self.Corr.fillna(0, inplace=True)

        return self.Corr

    def plot_Correlated_GDF(self, 
                             transform=None, 
                             projection=None,
							 suptitle='Correlação do EVI com o semi-log da incidência do HAV',
							 linewidth= 0.2,
							 n_ticks=5,
                             groupby_parameter_name=None, 
							 tipo_de_correlacao='Pearson', 
							 alpha=0.05,
                             fig=None,
                             ax=None,
                             nome_param_1=None,
                             nome_param_2=None,
                             decisao=2,
                             colorbar_yaxis_label='$R^2$\n(Pearson)'):
        """
		Parametros:
			GDF: geodataframe (geopandas)
			transform: Geotransform instance from the cartopy relative to the GDF data
			projection: projeção desejada
			suptitle: Título da figura. Por padrão: 'Correlação do EVI com o semi-log da incidência do HAV'
			linewidth: expessura da linha dos polígonos. Por padrão, 0.5
			
		Retorna: lista com 2 objectos do plot
			[0]: Axes da figura
			[1]: A figura em si

        """
        
        
        if groupby_parameter_name == None:
            print("groupby_parameter_name is None. Selecione entre as opções dadas: \n", self.GDF.columns)
            
            groupby_parameter_name = input("Insira o nome do atributo que será utilizado para agrupar os dados: \n")
        
        else:
            None
            
        self.groupby_parameter_name = groupby_parameter_name    
        
        
        
        if self.__Corr_Pronto is None:
            self.get_Correlation_from_Municipalities(groupby_parameter_name=groupby_parameter_name, 
                        							 tipo_de_Correlacao=tipo_de_correlacao, 
                        							 alpha=alpha,
                                                     nome_param_1=nome_param_1,
                                                     nome_param_2=nome_param_2,
                                                     decisao=decisao)
        else:
            if input("Os dados já foram calculados uma vez. Gostaria de calcular novamente (S/N)?").upper() == 'S':
                self.get_Correlation_from_Municipalities(groupby_parameter_name=groupby_parameter_name, 
                        							 tipo_de_Correlacao=tipo_de_correlacao, 
                        							 alpha=alpha,
                                                     nome_param_1=nome_param_1,
                                                     nome_param_2=nome_param_2,
                                                     decisao=decisao)
            else:
                None
        
        column_param_name = 'R2'
        
        self.plot_GDF(transform = transform, 
                      projection = projection, 
                      suptitle = suptitle,
					  linewidth = linewidth,
					  n_ticks = n_ticks,
                      fig = fig,
                      ax = ax,
                      column_param_name = column_param_name,
                      colorbar_yaxis_label=colorbar_yaxis_label)
        
    def plot_GDF(self, transform=None, projection=None, suptitle=None, linewidth=0.3, 
                 n_ticks=5, fig=None, ax=None, column_param_name=None,
                 mask_non_significant_correlations=True,
                 colorbar_yaxis_label='$R^2$\n(Pearson)',
                 save_fig=True):
        
        """
        
        Function that allows one to plot a standard map based on a specified attribute of the class instance GDF
        
        
        """
        
        
        if transform == None:

            self.Transform = ccrs.Geodetic(globe=ccrs.Globe(ellipse='GRS80')) # setting transform
        else:
            self.Transform = transform
        if projection == None:
            self.Projection=ccrs.PlateCarree() # setting projection
            
        else:
            self.Projection = projection

        if ax == None or fig == None:
            self.Fig, self.Ax = plt.subplots(1,1, subplot_kw={'facecolor':'white', 
														      'projection': self.Projection}) 
    
        else:
            if ax !=None and isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot):
                self.Fig = fig
                self.Ax = ax
                
            else:
                print("O axes fornecido não é geoaxes. refaça o algoritmo fornecendo uma instância Geoaxes do tipo cartopy.mpl.geoaxes.GeoAxesSubplot \n\n")
            
        
        self.Fig.suptitle(suptitle)


        Legend_KWDS ={
                      'loc':(1.25, 0.18),'fontsize':8,
                      'bbox_transform':self.Ax.transAxes,
                      'markerscale':0.75, # The relative size of legend markers compared with the originally drawn ones
                      'columnspacing':0.5, # The spacing between columns
                      'labelspacing':0.8, # The vertical space between the legend entries
                      'handletextpad':0.001} # float or None: The vertical space between the legend entries.
        try:
            gpd.GeoDataFrame(self.GDF, geometry='geometry').plot(ax=self.Ax, 
                       legend=False, 
                       facecolor='grey', 
                       edgecolor='k', 
                       linewidth=linewidth,
                       transform=self.Transform)
        except:
            None
        
        try:
        
            self.Corr.plot(ax= self.Ax, 
							column=column_param_name, 
							legend=True, 
							facecolor='white', 
							edgecolor='k', 
							alpha=0.6, 
							linewidth=linewidth,
							transform=self.Transform,
							legend_kwds=Legend_KWDS)
        except:
            None
			

        try:
            self.Colorbar = self.Fig.axes[-1]
            self.Colorbar.axes.set_ylabel(colorbar_yaxis_label)
        except:
            print("Nao foi possivel adicionar um label ao colorbar")
        
        if mask_non_significant_correlations==True:
            
            self._mask_p_values_polygons()
            
            xy =(0.15, 0.02)
        
            if self.Figura_em_Ingles.upper() == 'N':
    
                
                Nan_patch = mpatches.Patch(color='grey', label='P-valor > $alfa$: {0}'.format(str(self.Alpha).replace('.',',')))
    
                self.Fig.legend(handles=[Nan_patch], loc=(xy[0], xy[1]), fontsize=7.5)
                
            else:
                Nan_patch = mpatches.Patch(color='grey', label='P-value > $alpha$: {0}'.format(self.Alpha))
    
                self.Fig.legend(handles=[Nan_patch], loc=(xy[0], xy[1]), fontsize=7.5)
        
        else:
            None
        
        
        self._add_additional_features_to_map(n_ticks=n_ticks)
        
        if save_fig == True:
            if mask_non_significant_correlations==False:
                
                abs_original_path = os.path.abspath(self.__Figure_Path_saver)
                
                dir_name = os.path.dirname(self.__Figure_Path_saver)
                basename = os.path.basename(self.__Figure_Path_saver).split('.')[0] + '_unmasked_unsignificant_correlations.png'
                
                new_path = os.path.join(dir_name, basename)
                
                self.figure_path_saver(new_path)
                
                self._save_fig()
            
                self.figure_path_saver(abs_original_path)
                
            else:
                self._save_fig()
            
        else:
            None
        
        return [self.Fig, self.Ax]
    
    
    def _save_fig(self):
        """
        Function that saves the figure in the pre-specified path
        
        """
        
        if self.__Figure_Path_saver is None:
            if input("Quer salvar a figura?: (S/N) \n   ").lower() == 's':
                Path_fig_save = input("Insira o caminho do diretorio para salvar a figura: ")
    
                if os.path.exists(Path_fig_save) == False:
                    os.mkdir(Path_fig_save)
    
                Nome_fig = input("Insira o nome da Figura para ser salva (ex: figura.png): ")
    
                self.Fig.savefig(os.path.join(Path_fig_save, Nome_fig), dpi=900)
                print("\n\n", "Figura {0} salva ".format(Nome_fig), "\n\n")
                
                self.figure_path_saver(os.path.join(Path_fig_save, Nome_fig))
                
            else:
                None
                
        else:
            
            if os.path.exists(os.path.dirname(self.__Figure_Path_saver)) == False:
                os.mkdir(os.path.dirname(self.__Figure_Path_saver))
            
            else:
                None
                
            if self.__Figure_Path_saver.endswith('.png')==True:
                
                self.Fig.savefig(self.__Figure_Path_saver, dpi=900)
                print("\n\n", "Figura {0} salva ".format(os.path.basename(self.__Figure_Path_saver), "\n\n"))
            
            else:
                try:
                    self.Fig.savefig(self.__Figure_Path_saver, dpi=900)
                    print("\n\n", "Figura {0} salva ".format(os.path.basename(self.__Figure_Path_saver)), "\n\n")   
                
                except:
                    print("Problemas no save da figura. Use um nome da figura terminando com .png para garantir o save do Matplotlib")
                    Saver = input("Insira um novo nome para a figura (dica: termine com .png): ")
                    self.Fig.savefig(Saver, dpi=900)
                
                
    def _mask_p_values_polygons(self, n_ticks=5):
        """
        Function that masks unsignificant correlation vectors according to the p-value analysis.
        
        """
        
        self.Corr.loc[self.Corr['P_value']>self.Alpha].plot(ax=self.Ax, 
                                                             legend=False, 
                                                             facecolor='grey', 
                                                             edgecolor='k', 
                                                             linewidth=0.2,
                                                             transform=self.Transform)
        
        
    def _add_additional_features_to_map(self, n_ticks=5):
        """
        Generation function that inserts scale bar, north arrow, and gridlines to the map Figure.
        
        This function is only applyable to this class instance."
        
        """
        self.Gridliner = self.Ax.gridlines(crs=self.Projection , draw_labels=True, linewidth=0.5, alpha=0.4, color='k', linestyle='--')
        
        
        self.Gridliner.top_labels = False
        self.Gridliner.right_labels = False
        self.Gridliner.xlabels_top = False
        self.Gridliner.ylabels_right = False
        
        
		### Adding North Arrow:
        self.add_North_Arrow()
        
		## Adding scale bar:

        self.add_scale_bar(ax=self.Ax, length=200, location=(0.5, -0.12))

        self.Ax.xlabels_top = False
        self.Ax.ylabels_right = False
        self.Ax.ylabels_left = True
        self.Ax.xlines = True
        self.Ax.xformatter = LONGITUDE_FORMATTER
        self.Ax.yformatter = LATITUDE_FORMATTER


        llx0, llx1, lly0, lly1 = self.Ax.get_extent(ccrs.PlateCarree())

        self.Ax.xlocator = mticker.FixedLocator(np.round(np.linspace(llx0, llx1, n_ticks),0) , n_ticks)
        self.Ax.ylocator = mticker.FixedLocator(np.round(np.linspace(lly0, lly1, n_ticks),0) , n_ticks)


        self.Ax.xlabel_style = {'size': 15, 'color': 'gray'}
        self.Ax.ylabel_style = {'size': 15, 'color': 'gray'}
    
                    
                    
    def add_scale_bar(self, ax, length=200, location=(0.5, 0.01), linewidth=2.5):
        """
        Function that adds scale bar:
            
        Parameters:
    		ax is the axes to draw the scalebar on.
    		length is the length of the scalebar in km.
    		location is center of the scalebar in axis coordinates.
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

    def add_North_Arrow(self):
        
        """Function that adds north arrow"""
        
        
        x_tail = 0.975
        y_tail = 0.01
        x_head = 0.975
        y_head = 0.030
        dx = x_head - x_tail
        dy = y_head - y_tail

		
        arrow = mpatches.Arrow(x_tail, y_tail, dx, dy, width=0.05, transform=self.Fig.transFigure, color='k', figure=self.Fig)

        self.Fig.patches.extend([arrow])
    
        plt.text(x_head, 
				 y_head + 0.8*(y_head - y_tail) , 
				s='N', 
				size=11, 
				ha='center', 
				va='center',
				color='K', 
				transform=self.Fig.transFigure, 
				figure=self.Fig)