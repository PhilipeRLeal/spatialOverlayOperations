



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
    def __init__(self, GDF, Figure_Path_saver=None, Figura_em_Ingles='N'):
        """
        Esta classe de objeto realiza a análise de correlação município específica com base na série histórica dos dados fornecidos.
        Esta classe também plota esta análise de correlação.
        
        Parâmetros de entrada:
            - GDF: geodataframe contendo os dados a serem correlacionados
            
            - Figure_Path_saver (por padrão==None): caminho a ser utilizado para save da figura gerada
                Se a classe for instanciada com Figure_Path_saver definido como uma string, ele se utilizará desta String para salvar a figura automaticamente
            
            - Figura_em_Ingles (por padrão =='N'): se deixado como padrão, a figura terá formato brasileiro (decimal == ',') e algumas legendas ('p-valor', 'alfa') em português
        
        """
        
        
        self.GDF = GDF
        self.__Corr_Pronto = None
        self.Alpha = 0.05
        self.Figura_em_Ingles = Figura_em_Ingles
        
        self.Definindo_locale_Matplotlib(self.Figura_em_Ingles)
        
        self.Figure_Path_saver = Figure_Path_saver
    
    
    @property
    def Figure_Path_saver(self):
        return self.__Figure_Path_saver
        
    @Figure_Path_saver.setter
    def Figure_Path_saver(self, Path):
        self.__Figure_Path_saver = Path        
        
    def Definindo_locale_Matplotlib(self, Internacional='N'):
        if Internacional=='N':
            import locale
            locale.setlocale(locale.LC_ALL, "Portuguese_Brazil.1252")
            import matplotlib as mpl
            mpl.rcParams['axes.formatter.use_locale'] = True
        
        else:
            None
    
    
    def Correlacionador(self, DF, Nome_Param_1='EVI_250m_mean', Nome_Param_2='HAV', Decisao=2):

        
        if Decisao == 0:
            return [np.round(self.Tipo_de_Correlacao(DF.loc[:,Nome_Param_1].values, DF.loc[:,Nome_Param_2].values), 10)]  
        elif Decisao == 1:
            return [np.round(self.Tipo_de_Correlacao(DF.loc[:,Nome_Param_1].apply(lambda x: np.log(x+self.Semi_log_Adder)).values, DF.loc[:,Nome_Param_2].values), 10)]
        elif Decisao == 2:
            return [np.round(self.Tipo_de_Correlacao(DF.loc[:,Nome_Param_1].values, DF.loc[:,Nome_Param_2].apply(lambda x: np.log(x+self.Semi_log_Adder)).values), 10)]
        else:
            return [np.round(self.Tipo_de_Correlacao(DF.loc[:,Nome_Param_1].apply(lambda x: np.log(x+self.Semi_log_Adder)).values, DF.loc[:,Nome_Param_2].apply(lambda x: np.log(x+self.Semi_log_Adder)).values), 10)]
        
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
											Alpha=0.05,
                                            Geometry_column=None):
        """ 
		DF: Geodataframe contendo todos os dados já tabelados para correlação
		Groupby_parameter_name: nome do atributo para ser agrupado. Por padrão: 'GEOCODE_7'
		Tipo_de_Correlacao: Pearson ou Spearman. Por padrão, usar Pearson

        """
        
        
        
        if hasattr(self, 'Geometry_column') == False and Geometry_column == None:
            print("\n\n","Eis as colunas do GDF para seleção do geometry_column:", self.GDF.columns)
            self.Geometry_column = input("Selecione entre as opções dadas acima, qual deve ser utilizada como geometry para reconstrução do GeodataFrame: " )
        else:
            if hasattr(self, 'Geometry_column') == False:
                self.Geometry_column = Geometry_column
            else:
                Geometry_column = self.Geometry_column
        
        
        # Definindo o tipo de correlação que será utilizada:
        
        
        if Tipo_de_Correlacao.lower().startswith('pe'):
            self.Tipo_de_Correlacao= pearsonr
        else:
            self.Tipo_de_Correlacao=spearmanr

        print("Eis as colunas para correlação: ", self.GDF.columns)
        
        
        
        self.__Nome_Param_1=input('Insira o nome do Parametro 1 para correlacao: ')
        self.__Nome_Param_2=input('Insira o nome do Parametro 2 para correlacao: ')
        
        self.__Decisao = int(input("Quais variáveis devem ter seus dados transformados pelo semi-log? \n [1] para: {0} ;\n [2] para: {1} ;\n [3] para Ambas: ({0}, {1}) ;\n [0] para Nenhuma: \n".format(self.__Nome_Param_1 , self.__Nome_Param_2)))
        
        if self.__Decisao != 0:
            print("Valor padrão do Semi-log: {0}".format(0.00001))
            if input("Usar o valor padrão para a transformação de semi-log (S/N)? ").lower() == 's':
                self.Semi_log_Adder = 0.00001
                
            
            else:
                self.Semi_log_Adder = float(input('Defina o valor a ser adicionado na funcao semi-log: '))
        
        
        Corr = self.GDF.groupby([str(Groupby_parameter_name)]).apply(func=self.Correlacionador, 
																	 Nome_Param_1 = self.__Nome_Param_1,
																	 Nome_Param_2 = self.__Nome_Param_2,
                                                                     Decisao = self.__Decisao)
        

        Corr.name = 'Pearson_R2'
        self.Corr = pd.DataFrame(Corr)


        self.Corr = self.Corr.apply(self.Replacer, axis=1)


        self.Alpha = Alpha

        self.Corr.loc[self.Corr.P_value >self.Alpha, 'R2_masked' ] = np.nan

        try:
            
            if hasattr(self.GDF, 'crs') == False:
                print("Faltou a inserção de um CRS. Tentaremos inserir pelo padrão do geopandas.GeoDataFrame")
                self.Corr = gpd.GeoDataFrame(self.GDF.loc[:, [str(Groupby_parameter_name), self.Geometry_column]].merge(self.Corr, left_on=[str(Groupby_parameter_name)], 
    					right_on=[str(Groupby_parameter_name)],
    					how='inner'), geometry=self.Geometry_column)
            else:
                self.Corr = gpd.GeoDataFrame(self.GDF.loc[:, [str(Groupby_parameter_name), self.Geometry_column]].merge(self.Corr, left_on=[str(Groupby_parameter_name)], 
    					right_on=[str(Groupby_parameter_name)],
    					how='inner'), geometry=self.Geometry_column, crs=self.GDF.crs)
                
            print("\n\n" , "GeoDataFrame criado com sucesso\n\n")
        except:
            print("\n\n" , "Problemas na criação do GeoDataframe")

        self.__Corr_Pronto = 1

        self.Corr.fillna(0, inplace=True)

        return self.Corr

    def Print_Correlated_GDF(self, Transform=None, Projection=None,
							Suptitle='Correlação do EVI com o semi-log da incidência do HAV',
							Linewidth= 0.3,
							N_Ticks=5,
                            groupby_parameter_name=None, 
							tipo_de_correlacao='Pearson', 
							alpha=0.05):
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
        
        
        if groupby_parameter_name == None:
            print("groupby_parameter_name is None. Selecione entre as opções dadas: \n", self.GDF.columns)
            
            groupby_parameter_name = input("Insira o nome do atributo que será utilizado para agrupar os dados: \n")
        
        else:
            None
            
        self.groupby_parameter_name = groupby_parameter_name    
        
        
        
        if self.__Corr_Pronto is None:
            self.Get_Correlation_from_Municipalities(Groupby_parameter_name=groupby_parameter_name, 
                        							 Tipo_de_Correlacao=tipo_de_correlacao, 
                        							 Alpha=alpha)
        else:
            if input("Os dados já foram calculados uma vez. Gostaria de calcular novamente (S/N)?").upper() == 'S':
                self.Get_Correlation_from_Municipalities(Groupby_parameter_name=groupby_parameter_name, 
                        							 Tipo_de_Correlacao=tipo_de_correlacao, 
                        							 Alpha=alpha)
            else:
                None

        if Transform == None:

            self.Transform = ccrs.Geodetic(globe=ccrs.Globe(ellipse='GRS80')) # setting transform

        if Projection == None:
            self.projection=ccrs.PlateCarree() # setting projection


        self.fig, self.ax = plt.subplots(1,1, subplot_kw={'facecolor':'white', 
														  'projection': self.projection}) 
			
        self.fig.suptitle(Suptitle)


        Legend_KWDS ={
                      'loc':(1.25, 0.18),'fontsize':8,
                      'bbox_transform':self.ax.transAxes,
                      'markerscale':0.75, # The relative size of legend markers compared with the originally drawn ones
                      'columnspacing':0.5, # The spacing between columns
                      'labelspacing':0.8, # The vertical space between the legend entries
                      'handletextpad':0.001} # float or None: The vertical space between the legend entries.
        try:
            gpd.GeoDataFrame(self.GDF, geometry='geometry_G').plot(ax=self.ax, 
                       legend=False, 
                       facecolor='grey', 
                       edgecolor='k', 
                       linewidth=Linewidth,
                       transform=self.Transform)
        except:
            None
        
        
        
        self.Corr.plot(ax= self.ax, 
					    column='R2', legend=True, 
						facecolor='white', 
						edgecolor='k', 
						alpha=0.6, 
						linewidth=Linewidth,
						transform=self.Transform,
						legend_kwds=Legend_KWDS,
                        label= '$R^2$')

        self.Colorbar = self.fig.axes[-1]
        
        try:
            self.Colorbar.axes.set_ylabel('$R^2$\n(Pearson)')
        except:
            print("Nao foi possivel adicionar um label ao colorbar")
        #self.Colorbar.ax.tick_params(labelsize=10) --- to set the size of the tickparameters of the Colorbar
        
        
        xy =(0.15, 0.02)

        
              
        
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
        self.ax.ylabel_style = {'size': 15, 'color': 'gray'}

        if self.Figura_em_Ingles.upper() == 'N':

            
            Nan_patch = mpatches.Patch(color='grey', label='P-valor > $alfa$: {0}'.format(str(self.Alpha).replace('.',',')))

            self.fig.legend(handles=[Nan_patch], loc=(xy[0], xy[1]), fontsize=7.5)

            
        else:
            Nan_patch = mpatches.Patch(color='grey', label='P-value > $alpha$: {0}'.format(self.Alpha))

            self.fig.legend(handles=[Nan_patch], loc=(xy[0], xy[1]), fontsize=7.5)
        
        if self.__Figure_Path_saver is None:
            if input("Quer salvar a figura?: (S/N) \n   ").lower() == 's':
                Path_fig_save = input("Insira o caminho do diretorio para salvar a figura: ")
    
                if os.path.exists(Path_fig_save) == False:
                    os.mkdir(Path_fig_save)
    
                Nome_fig = input("Insira o nome da Figura para ser salva (ex: figura.png): ")
    
                self.fig.savefig(os.path.join(Path_fig_save, Nome_fig), dpi=900)
                print("\n\n", "Figura {0} salva ".format(Nome_fig), "\n\n")
            else:
                None
                
        else:
            
    
            if os.path.exists(os.path.dirname(self.__Figure_Path_saver)) == False:
                os.mkdir(os.path.dirname(Path_fig_save))
            
            else:
                None
                
            if self.__Figure_Path_saver.endswith('.png')==True:
                
                self.fig.savefig(self.__Figure_Path_saver, dpi=900)
                print("\n\n", "Figura {0} salva ".format(os.path.basename(self.__Figure_Path_saver), "\n\n"))
            
            else:
                try:
                    self.fig.savefig(self.__Figure_Path_saver, dpi=900)
                    print("\n\n", "Figura {0} salva ".format(os.path.basename(self.__Figure_Path_saver)), "\n\n")   
                
                except:
                    print("Problemas no save da figura. Use um nome da figura terminando com .png para garantir o save do Matplotlib")
                    Saver = input("Insira um novo nome para a figura (dica: termine com .png): ")
                    self.fig.savefig(Saver, dpi=900)
                
        return [self.fig, self.ax]

    def Correcao_de_Ticks_para_Figuras_Brasileiras(self):
        """
		Parameter:
			axes_number: o número do axes da figura que será corrigida. 
				Por padrão é o fig.axes[-1] == (axes do colorbar).

        """
        


        self.Colorbar_Y_legend = self.Colorbar.get_yticklabels()

        for yi in self.Colorbar_Y_legend:
            T = yi.get_text()
            T = T.replace('.',',')
            yi = yi.set_text(T)
			
            print(T)


        self.Colorbar.set_yticklabels(self.Colorbar_Y_legend)



        self.Colorbar_X_legend = self.Colorbar.get_xticklabels()

        for xi in self.Colorbar_X_legend:
            T = xi.get_text()
            T = T.replace('.',',')
            xi = xi.set_text(T)
			
            print(T)


        self.Colorbar.set_xticklabels(self.Colorbar_X_legend)


		## Setting ticks from the Axis X and Y of the main Axes
        Xticks = self.ax.get_xticklabels()
        X_tick_corrected_list= []
        for i in Xticks:
            T = i.get_text()
            T.replace('.',',')
            i.set_text(T)
            
            print(T)
            X_tick_corrected_list.append(T)
            
        self.ax.set_xticklabels(X_tick_corrected_list)
        
        
        print("Tentando alterar os ticks do Axes")

        Yticks = self.ax.get_yticklabels()
        Y_tick_corrected_list= []
        for i in Yticks:
            T = i.get_text()
            T.replace('.',',')
            i.set_text(T)
					
            print(T)
            Y_tick_corrected_list.append(T)
            
        self.ax.set_yticklabels(Y_tick_corrected_list)

        ### Tentativa final:
        try:
            print("Penúltima Tentativa de Setar os ticks dos axis x e y")
            print("\n\n")
            
            print("Ticks do Eixo x já corrigidas", pd.Series(self.ax.get_xticks()).apply(lambda x: str(x).replace('.', ',')).values)
            self.ax.set_xticklabels(pd.Series(self.ax.get_xticks()).apply(lambda x: str(x).replace('.', ',')).values)
            
            print("Ticks do Eixo y já corrigidas", pd.Series(self.ax.get_yticks()).apply(lambda x: str(x).replace('.', ',')).values)
            
            self.ax.set_yticklabels(pd.Series(self.ax.get_yticks()).apply(lambda x: str(x).replace('.', ',')).values)
        except:
            None
            
        print("Última Tentativa de Setar os ticks dos axis x e y")
        print("\n\n")
            
        import matplotlib.ticker as tkr
        
        def func(x, pos):  # formatter function takes tick label and tick position
            s = str(x)
            ind = s.index('.')
            return s[:ind] + ',' + s[ind+1:]   # change dot to comma
        
        y_format = tkr.FuncFormatter(func)
        self.ax.yaxis.set_major_formatter(y_format)  # set formatter to needed axis
        self.ax.xaxis.set_major_formatter(y_format)
                    
                    
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