# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:26:35 2019

@author: lealp
"""

import numpy as np
import geopandas as gpd

import pandas as pd
pd.set_option('display.width', 50000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


class overlay_spatial_mean_class():
    
    def __init__(self,  gdf1, gdf2, common_epsg=5880, preserve_original_columns=False):
        self.gdf1 = gdf1
        
        self.gdf2 = gdf2
        
        self.common_epsg = common_epsg
        
        self.preserve_original_columns = preserve_original_columns
        
        
        # applying function during object creation
        
        
        self.result = self.spatial_statistics_using_overlay_operations(df1, df2, common_epsg, preserve_original_columns)
    

        
    def get_area_from_gdf(self, gdf, common_epsg=5880, area_col_name=None):
        
        gdf.to_crs(epsg=common_epsg, inplace=True)
        
        gdf[area_col_name] = gdf.geometry.area
        
        return gdf

    def attribute_by_area_ratio(self, column_series, column_series_area_result, column_series_original_area):
        
        return (column_series/column_series_original_area) * column_series_area_result

    def calculate_relative_spatial_statistics(self, gdf):
        
        gdf['area_result'] = gdf.geometry.area
        
        
        for c in gdf.columns:
            
            if ( np.issubdtype(gdf[c].dtype, np.number) ) and (c not in ['geometry', 'area_result', 'to_area', 'from_area'] ) :
            
                print('operating over column {0}'.format(c))
                
                gdf[str(c) + 'rec'] = self.attribute_by_area_ratio( gdf[c],
                                                              gdf['area_result'], 
                                                              gdf['from_area']
                                                              
                                                             )
                                            
            else:
                print('{0} is being discarded from the equation'.format(c))
            
            
            
        return gdf
    
   
            
    
    def spatial_statistics_using_overlay_operations(self, gdf1, gdf2, common_epsg=5880, preserve_original_columns=False):
        """
        Function description:
            
            This function applies a overlay spatial statistical operation over two geodataframes (geopandas.geodataframes).
            
            The resulted intersected geometries from gdf1 over gdf2 are used as ratios for convertion of the numerical columns of gdf2.
            
            For each numerical column in gdf1 and gdf2, its value is multiplied by the estimated ratio Index
            for estimation of a mean value for the geometry attribute in gdf1.
            
            ex:
            
            
                "geo_from_gdf1" intersects "geo_from_gdf2" by 30%.
                
                "attribute_A_from_gdf1" will be equal to "attribute_A_from_gdf2" * 0.3
                
                
                # It is noteworthy that the function does not require that gdf1 possess all the numerical attributes from gdf2. 
                # They are generated for gdf1 during the function operation.
            
        Parameters:
            gdf1 (geodataframe): the geodataframe to which the columns from gdf2 will be evaluated. 
            
            gdf2 (geodataframe): the geodataframe base, from which the numerical attributes will be multiplied by each respective spatial ratio index
            
            common_epsg (int): a common epsg number that will be used to convert both geodataframes to.
        
        
            preserve_original_columns (bool): it sets what columns to return from the function.
                If True, all original columns from both geodataframes will be preserved in the returned gdf
                
                If False (standard), only the resultant columns will be returned by the function
                
        Return:
            Geodataframe
            
        """
        
        
        gdf1 = self.get_area_from_gdf(gdf1 , common_epsg, area_col_name = 'to_area')
        
        gdf2 = self.get_area_from_gdf(gdf2 , common_epsg, area_col_name = 'from_area')
    
        
        res_union = gpd.overlay(gdf1, gdf2, how = 'intersection')
        
        original_columns = res_union.columns
        
        res_union = self.calculate_relative_spatial_statistics(res_union)
        
        
        if preserve_original_columns == False:
            for c in original_columns:
                if c not in ['geometry']:
                    res_union.drop({c}, axis=1, inplace=True)
                    
        
        return res_union
    
    
    def __call__ (self, gdf1, gdf2, common_epsg=5880, preserve_original_columns=False):
        
        
        self.__init__(gdf1, gdf2, common_epsg=5880, preserve_original_columns=False)
        
    
    def __str__(self):
        
        return 'geopandas.geodataframe.overlay_class'
    
    def __dir__(self):
        
        return ['attribute_by_area_ratio',
                 'calculate_relative_spatial_statistics',
                 'common_epsg',
                 'gdf1',
                 'gdf2',
                 'get_area_from_gdf',
                 'overlay_result',
                 'preserve_original_columns',
                 'spatial_statistics_using_overlay_operations']
    

if "__main__" == __name__:
    
    from shapely.geometry import Polygon
    
    polys1 = gpd.GeoSeries([Polygon([(0,0), (2,0), (2,2), (0,2)]),
                                   Polygon([(2,2), (4,2), (4,4), (2,4)])])
     
    
    polys2 = gpd.GeoSeries([Polygon([(1,1), (3,1), (3,3), (1,3)]),
                                  Polygon([(3,3), (5,3), (5,5), (3,5)])])
    
    df1 = gpd.GeoDataFrame({'geometry': polys1, 'df1':[1,2]})
    
    
    
    
    df2 = gpd.GeoDataFrame({'geometry': polys2, 'df2':[1,2]})
    
    df2['valor_df2'] = [15, 10]
    
    
    df1.crs = {'init' :'epsg:4326'}
    
    df2.crs = {'init' :'epsg:4326'}
    
    
    Result = overlay_spatial_mean_class(df1, df2, common_epsg=4623, preserve_original_columns=True)
    
    
    Result(df1, df2, common_epsg=4623, preserve_original_columns=True)
    
    # results
    
    print(Result.result)
    
    
    # method available
    
    print(dir(Result))
    
