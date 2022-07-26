# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:57:13 2020

@author: Philipe_Leal
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import geopandas as gpd
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from shapely.geometry import Point

end = '\n'*2 + '-'*40 + '\n'*2

def get_area_sum_per_group(group, geometry_column,
                           verbose=True):
    
    Areas = group[geometry_column].area
    
    Summed_area = np.sum(Areas)
    
    if verbose:
        print('Group name: ', group.name, '\n', 
           'Summed area', Summed_area, end= '\n'*3)
    
    return Summed_area
    


def select_category(df, 
                    category='A', 
                    decision_rule='max', 
                    grouping_rule=dict(),
                    geometry_column='geometry',
                    verbose=True):
    
    df['group'] = df[category].map(grouping_rule)
    
    df = gpd.GeoDataFrame(df)
    
    grouped = df.groupby('group')
    
    Agg = grouped.apply(lambda group: get_area_sum_per_group(group, geometry_column, verbose=verbose))
    
    
    if 'max' in decision_rule.lower():
        search_rule = Agg.iloc[np.argmin( (Agg - np.percentile(Agg, 100) )**2) ]
        
    elif 'min' in decision_rule.lower():
        search_rule = Agg.iloc[np.argmin( (Agg - np.percentile(Agg, 0) )**2) ]
        
    else: # if user is giving a percentage:
        search_rule = Agg.iloc[np.argmin( (Agg - np.percentile(Agg, decision_rule) )**2) ]    
    
    
    
    Retrieved = Agg[Agg==search_rule].index[0]
    
    if verbose:
        print('search_rule: ', search_rule)
    
        print('\n\n\t', 'Retrieved: ', Retrieved, end=end)
    
    
    return Retrieved


def update_main_df(main_df, geofeature):

    mask = (x.intersects(geofeature) for x in main_df.geometry)
    
    
    main_df = main_df.loc[mask].copy()
    
    if main_df.empty:
        return np.nan
        
    else:
        main_df.geometry = main_df.geometry.apply(lambda x: x.intersection(geofeature))
         
         
        return main_df
    
   
def select_categories(main_df,
                      geofeature,
                      category = 'A',
                      decision_rule = 'max', 
                      grouping_rule = dict(),
                      geometry_column = 'geometry',
                      verbose=True):
    
    
    
    main_df = update_main_df(main_df, geofeature)
    
    if isinstance(main_df, gpd.GeoDataFrame):
    
        category_selected = select_category(main_df, 
                                            category=category, 
                                            decision_rule=decision_rule, 
                                            grouping_rule=grouping_rule,
                                            geometry_column=geometry_column,
                                            verbose=verbose)
        
        return category_selected
        
    else:
        return np.nan
        


def main_selector_of_categories(main_df, 
                                ref_df,
                                categorical_columns = ['A'],
                                decision_rule = 'max', 
                                grouping_rules = [dict()],
                                geometry_column = 'geometry',
                                verbose=True,
                                plot=True):
    '''
    
    Description: This function applies an overlay operation over categorical variables.
    
    Given a main_df (the one that will be updated, and later returned to the user), 
    and a ref_df (a referential gdf which contains the given categorical_columns list),
    the given function retrieves the class whose decision rules bests over all other classes
    for each feature in main_df.
    
    For example:
    
        Given a main_df (geopandas GeoDataSeries of Polygons),
        and a ref_df (geopandas GeoDataSeries of Polygons),
        
        what is the main class (the class that predominates) in each feature of the main_df?
        
        By using the classes from ref_df, a spatial search is made for each main_df's geofeature,
        and for each case, the category that best represents that polygon is selected.
    
    ------------------------------
    
    Parameters
    
    main_df (geopandas GeoDataSeries of Polygons): the gdf that will be updated with the classes of ref_df 
    
    -----------
    
    ref_df(geopandas GeoDataSeries of Polygons): the referential gdf with the categorical classes
    
    
    -----------
    
    categorical_columns = ['A'],
    
    
    decision_rule = 'max', 
    
    
    grouping_rules = [dict()]
    
    
    geometry_column = 'geometry',
    
    
    verbose (bool): standard is True
    
    plot (bool): standard is True
    
    

    '''
                                
                                
                                
         
    legend_kwds1 = {'loc':'upper right',
                       'bbox_to_anchor':(0.85, 0.9),
                       'facecolor':'white',
                       'edgecolor':'k',
                       'title':'gdfs1'}
                       
    legend_kwds2 = {'loc':'upper right',
                       'bbox_to_anchor':(0.2, 0.9),
                       'facecolor':'white',
                       'edgecolor':'k',
                       'title':'gdfs2'}
    
    
    for category, grouping_rule in zip(categorical_columns, grouping_rules):
    
        print('- -'*20,'\n','- -'*20, '\n'*2, '\t', 'Starting category: {0}'.format(category), end='\n'*2)
        main_df[category] = main_df.geometry.apply(lambda geofeature: 
                                                     select_categories(ref_df,
                                                     geofeature,
                                                     category= category,
                                                     decision_rule='max', 
                                                     grouping_rule=grouping_rule,
                                                     geometry_column='geometry',
                                                     verbose=verbose
                                                     ))
        
        if plot:
            fig, ax = plt.subplots()
            
            legend_kwds1['bbox_transform']  = fig.transFigure
            legend_kwds2['bbox_transform']  = fig.transFigure  
            
            
            main_df.plot(categorical=True,
                         #column=category,
                         color='red',
                         alpha=0.8,
                         legend_kwds=legend_kwds2,
                         legend=True, zorder=10,
                         #cmap='viridis',
                         ax=ax)
                         
            
                              
            ref_df.plot(categorical=True,
                        column=category,
                        alpha=0.7, ax=ax, zorder=-1,
                        legend_kwds=legend_kwds1,
                        legend=True, cmap='jet')
            
            
            for x, y, label in zip(main_df.geometry.centroid.x, main_df.geometry.centroid.y, main_df.index):
                ax.annotate(str(label)+'_main', xy=(x, y), xytext=(0, 0), textcoords="offset points", zorder=12)
            
            
            fig.suptitle(category)
            fig.tight_layout()
            fig.show()
    
    
        
    return main_df

if '__main__' == __name__:
    for i in range(10):
    
        points = np.random.normal(i, 12, size=(2, 8)).reshape((8,2)) 
        
        geometries_1 = [Point(p).buffer(np.random.lognormal(2,1.7)) for p in points]
        
        geometries_2 = [Point(p*np.random.normal(0, 3)).buffer(5) for p in points]
        
        ref_df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],
                              'B' : ['rosa', 'vermelho', 'lilaz', 'rosa','lilaz', 'vermelho', 'branco', 'preto'],
                               'geometry' : geometries_1})
        
        main_df =  pd.DataFrame({'A' : ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],
                           'geometry' : geometries_2})
        
        
        
        ref_df = gpd.GeoDataFrame(ref_df)
        main_df = gpd.GeoDataFrame(main_df)
        
        
        ########
        
        
        
        ##########
        d = {'one':'Start', 'two':'Start', 'three':'End'}
        d2 = {'rosa':1, 
             'vermelho':2, 
             'lilaz':3,
             'branco':4,
             'preto':5}
        
        
        main_df = main_selector_of_categories(main_df, 
                                            ref_df,
                                            categorical_columns = ['A', 'B'],
                                            decision_rule = 'max', 
                                            grouping_rules = [d, d2],
                                            geometry_column = 'geometry',
                                            verbose=False)
        print(main_df)
        
       
        
        input('Press any to next loop \n', )
        plt.close('all')
        
        # Selected_Category = select_category(ref_df, 
        #                 category='A', 
        #                 decision_rule='min', 
        #                 grouping_rule=d,
        #                 geometry_column='geometry')
        
        
        # print('Selected_Category: ', Selected_Category)
        