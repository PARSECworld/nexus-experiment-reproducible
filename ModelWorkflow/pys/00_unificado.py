#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# !pip install --upgrade certifi
# !pip install google-colab
# pip install --use-pep517 google-colab
# pip install google-colab

# from google.colab import drive
# drive.mount("/content/drive",force_remount = True)

# %cd drive/Shareddrives/'NEXUS-PARSEC - TCCs'/ProjetoRaf-Car-Igor/data/indicadores_IDH_Nexus


# In[4]:


import geopandas


# In[5]:


from os import listdir
import glob
import pandas
import geopandas
geo = geopandas.read_file('../data/raw/setores_shapefile/ac_setores_censitarios/12SEE250GC_SIR.shp')
geo.info()


# In[6]:


MERCATOR_CRS = "EPSG:3857"  # Mercator projection CRS.
METRIC_CRS = {"proj":"cea"} # Metric CRS. Useful for calculations
WGS84_CRS = "EPSG:4326"     # World Geodetic System
SIRGAS_CRS = "EPSG:4674"    # SIRGAS 2000, a projection for LATAM

PLOT_CRS = MERCATOR_CRS


# In[7]:


shapefiles_list = glob.glob('../data/raw/setores_shapefile/*/*.shp')
geo = geopandas.read_file(shapefiles_list[0])

from tqdm import tqdm
for file in tqdm(shapefiles_list[1:]):
    geoaux = geopandas.read_file(file)
    geo = pandas.concat([geo,geoaux],ignore_index = True)


# In[8]:


geo = geo.to_crs(WGS84_CRS)


# In[9]:


geo = geo.rename(columns ={'CD_GEO CODI':'Cod_setor','CD_GEOCODM' : 'Cod_municipio'})
geo = geo.astype({'Cod_municipio' : 'int64'})
geo = geo.drop(columns = ['ID','CD_GEOCODB','NM_BAIRRO','CD_GEOCODS','NM_SUBDIST','CD_GEOCODD'])
geo = geo.drop(columns = ['NM_DISTRIT','NM_MUNICIP','NM_MICRO','NM_MESO','ID1'])
geo.info()


# In[10]:


geo['centroid'] = geo['geometry'].centroid
# #Extract lat and lon from the centerpoint
geo["lat"] = geo['centroid'].map(lambda p: p.x)
geo["lon"] = geo['centroid'].map(lambda p: p.y)
geo['area'] = geo['geometry'].area*1.25e4

#'import math
cod_municipio = geo['Cod_municipio']
cod_estado = (cod_municipio/100000)
cod_estado = cod_estado.astype('int64')
geo['Cod_estado'] = (cod_estado)

geo = geo.assign(year = 2010)


# In[11]:


geo


# # Notebook 2

# # Merge NEXUS data
# ## Description
# This notebook merges the geographic data of NEXUS census tracts with its indicator values. Then, it clears that dataset, excluding all samples that have at least one indicator as null, unknown or off the interval (0,1].
# ## Inputs
# `NEXUS_census_tracts_2010.csv`: dataset containing the geographic data of each census tract in the NEXUS area.
# 
# `IDM_NEXUS_{state acronym}.csv`: 19 datasets, one for each state acronym, containing indicator target values of each census tract in the NEXUS area.
# 
# ## Outputs
# `NEXUS_data_2010.csv`: dataset containing the geographic data and indicator target values of each census tract within the NEXUS area.

# # Configs

# ## Libraries

# In[12]:


import glob
import pandas as pd
import geopandas as gpd


# ## Projection constants

# In[13]:


MERCATOR_CRS = "EPSG:3857"  # Mercator projection CRS.
METRIC_CRS = {"proj":"cea"} # Metric CRS. Useful for calculations
WGS84_CRS = "EPSG:4326"     # World Geodetic System
SIRGAS_CRS = "EPSG:4674"    # SIRGAS 2000, a projection for LATAM

PLOT_CRS = MERCATOR_CRS


# # Execution

# ## Read input data

# ### Read geographic dataset

# In[14]:


#geodf_census_tracts = gpd.read_file("../data/interim/NEXUS_tracts_2010.csv") #, encoding = "iso-8859-1"
geodf_census_tracts = geo


# In[15]:


# from shapely.wkt import loads
# geodf_census_tracts.geometry =  geodf_census_tracts['geometry_aux'].apply(loads)


# In[16]:


geo_backup = geodf_census_tracts.copy()


# In[17]:


#!! pode inverter aqui
#geodf_census_tracts = geodf_census_tracts.set_crs(WGS84_CRS) 
geodf_census_tracts = geodf_census_tracts.to_crs(MERCATOR_CRS) 


# In[18]:


geodf_census_tracts.head()


# In[19]:


geodf_census_tracts = geodf_census_tracts.rename(columns = {
    "CD_GEOCODI": "cod_sector",
    "CD_GEOCODM": "cod_mun",
    "TYPE": "type"
  })


# In[20]:


# original
# geodf_census_tracts = geodf_census_tracts.astype({
#     "cod_sector": "int64",
#     "cod_mun": "int64"
#   })

geodf_census_tracts = geodf_census_tracts.astype({
    "cod_sector": "int64",
    "Cod_municipio": "int64"
  })


# In[21]:


geodf_census_tracts.info()


# ### Read indicator dataset

# In[22]:


indicator_data_paths = glob.glob("../data/raw/NexusIndicators/*.csv")
assert(len(indicator_data_paths) == 19)


# In[23]:


df_indicators = pd.read_csv(indicator_data_paths[0])


# In[24]:


for state_indicator_data_path in indicator_data_paths[1:]:
    df_indicators = pd.concat([
      df_indicators,
      pd.read_csv(state_indicator_data_path)
    ])


# In[25]:


df_indicators = df_indicators.rename(columns = {
    "Cod_setor": "cod_sector",
    "IDHL-Setor": "longevity",
    "AlfabIDHM": "literacy",
    "RendaIDHM": "income"
  })
df_indicators = df_indicators.astype({
    "cod_sector": "int64",
  })


# In[26]:


df_indicators = df_indicators[["cod_sector", "income", "literacy", "longevity"]]


# In[27]:


df_indicators.info()


# ## Merge datasets

# In[28]:


# from shapely.wkt import loads
# geodf_census_tracts.geometry =  geodf_census_tracts['geometry_aux'].apply(loads)
#geodf_census_tracts = geodf_census_tracts.set_geometry('geometry')


# In[29]:


geodf_nexus = geodf_census_tracts.merge(df_indicators, how = "outer", on = "cod_sector")


# In[30]:


geodf_nexus = geodf_nexus.assign(year = 2010)


# In[31]:


geodf_nexus["cod_state"] = geodf_nexus.apply(lambda x: int(x["Cod_municipio"] / 100000), axis = 1)


# In[32]:


geodf_nexus


# In[33]:


geodf_nexus_backup = geodf_nexus.copy()


# In[34]:


#original
#geodf_nexus = geodf_nexus[["cod_sector", "cod_mun", "cod_state", "type", "year", "income", "literacy", "longevity", "WKT", "geometry"]]


# In[35]:


geodf_nexus = geodf_nexus[["cod_sector", "Cod_municipio", "cod_state", "TIPO", "year", "income", "literacy", "longevity", "geometry"]]


# In[36]:


geodf_nexus.info()


# ## Clear data

# In[37]:


geodf_nexus


# In[38]:


geodf_nexus = geodf_nexus.dropna()


# In[39]:


geodf_nexus.describe()


# ### Clear based on income column
# 

# #### Values equal or smaller than 0
# There are negative values (`-inf`) in `income` column. It is important to analyze its histogram and fix data equal to or less than 0.

# In[40]:


geodf_nexus = geodf_nexus.replace(
    {"income":  -float("inf")},
    {"income":  -1}
)


# In[41]:


geodf_nexus["income"].describe()


# In[42]:


geodf_nexus[geodf_nexus["income"] <= 0]["income"].hist()


# In[43]:


100 * len(geodf_nexus[geodf_nexus["income"] <= 0]["income"]) / len(geodf_nexus["income"])


# The amount of values equal or smaller than 0 in `income` column is less than 0.1%.
# 
# These samples are removed.

# In[44]:


geodf_nexus = geodf_nexus[geodf_nexus["income"] > 0]


# In[45]:


geodf_nexus.describe()


# #### Values bigger than 1
# There are values above 1 in `income` column. It is important to analyze its histogram and fix data bigger than 1.

# In[46]:


geodf_nexus[geodf_nexus["income"] > 1]["income"].hist()


# In[47]:


100 * len(geodf_nexus[geodf_nexus["income"] > 1]["income"]) / len(geodf_nexus["income"])


# The amount of values bigger than 1 in `income` column is close to 0.1%.
# 
# These samples values are updated to `1.0`.

# In[48]:


geodf_nexus["income"] = geodf_nexus.apply(lambda row: min(1.0, row["income"]), axis = "columns")


# In[49]:


geodf_nexus.describe()


# ### Clear based on literacy column
# There are values equal to 0 in `literacy` column. It is important to analyze its histogram and fix data equal to 0.
# 

# In[50]:


geodf_nexus["literacy"].hist()


# In[51]:


100 * len(geodf_nexus[geodf_nexus["literacy"] == 0]["literacy"]) / len(geodf_nexus["literacy"])


# The amount of values equal to 0 in `literacy` column is negligible.
# 
# These samples are removed.

# In[52]:


geodf_nexus = geodf_nexus[geodf_nexus["literacy"] > 0]


# In[53]:


geodf_nexus.describe()


# ## Store dataset

# In[54]:


# geodf_nexus = geodf_nexus[[
#     "cod_sector",
#     "Cod_municipio",
#     "cod_state",
#     "TIPO",
#     "year",
#     "income",
#     "literacy",
#     "longevity",
#   ]]


# In[55]:


# gerado por mim
geodf_nexus[geodf_nexus.cod_sector.isin([110030405000001, 110030405000002, 110030405000003, 110030405000004, 110030405000005])]


# In[56]:


#antes printado no notebook
geodf_nexus.head()


# In[57]:


geodf_nexus.info()


# In[58]:


geodf_nexus.to_csv("../data/processed/NEXUS_data_2010.csv", index = False)


# In[59]:


geodf_nexus_backup = geodf_nexus.copy()


# In[60]:


geodf_nexus


# # Notebook 3

# # Select Clusters
# ## Description
# This notebook draws a grid over the NEXUS area map and selects clusters formed by multiple census tracts.
# ## Inputs
# `NEXUS_data_2010.csv`: dataset containing the geographic data and indicator target values of each census tract within the NEXUS area.
# ## Outputs

# # Configs

# ## Libraries

# In[61]:


import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon
from shapely.wkt import loads, dumps
from tqdm import tqdm


# In[62]:


import utils.plot_utils


# In[63]:


# import sys
# sys.path.append("/content/drive/Shareddrives/NEXUS paper - TCC/codes/utils/")
# import plot_utils


# ## Projection constants

# In[64]:


MERCATOR_CRS = "EPSG:3857"  # Mercator projection CRS.
METRIC_CRS = {"proj":"cea"} # Metric CRS. Useful for calculations
WGS84_CRS = "EPSG:4326"     # World Geodetic System
SIRGAS_CRS = "EPSG:4674"    # SIRGAS 2000, a projection for LATAM

PLOT_CRS = MERCATOR_CRS


# # Execution

# ## Read input data

# In[65]:


#geodf_nexus = gpd.read_file("../data/interim/NEXUS_data_2010.csv", encoding = "iso-8859-1")
#geodf_nexus = geodf_nexus.set_crs(MERCATOR_CRS)


# In[66]:


geodf_nexus = geodf_nexus.to_crs(METRIC_CRS)


# In[67]:


geodf_nexus = geodf_nexus.astype({
    "cod_sector": "int64",
    "income": "float64",
    "literacy": "float64",
    "longevity": "float64"
  })


# In[68]:


geodf_nexus = geodf_nexus[["cod_sector", "year", "income", "literacy", "longevity", "geometry"]]


# In[69]:


geodf_nexus.info()


# In[70]:


geodf_nexus.head()


# ## Create grid
# This step objective is to draw a grid over the area of interest, where the size of each square is equal to the cluster image size ($size = 6.7 km$).
# 
# 

# ### Create grid: define grid limits
# First, get the vertices coordinates that form the bounding box of the area of interest: `minx`, `maxx`, `miny` and `maxy`.
# 
# Each square in the grid is called by its upper left vertice indices as $(i,j)$.
# 
# <img src="https://drive.google.com/uc?export=view&id=1uZQxdN0N-YHUI8dI55bYdI1Pco29Q9lZ"/>
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# The vertices horizontal coordinates are $x_i, i = 0, 1, ..., n$, while their vertical coordinates are $y_j, j = 0, 1, ..., m$.
# 
# The i-th coordinate along the horizontal axis is calculated as $x_i = i \cdot size + min_x$.
# 
# The j-th coordinate along the vertical axisis calculated as $y_j = j \cdot size + min_y$.
# 
# <img src="https://drive.google.com/uc?export=view&id=11b077Douq1OBFVxdym9-xUSjutvy3vgN"/>
# 

# In[71]:


min_x, min_y, max_x, max_y = geodf_nexus["geometry"].total_bounds


# Plot the map bounding box, just for visualization.

# In[72]:


bbox_vertices = [
    Point(min_x, min_y),
    Point(min_x, max_y),
    Point(max_x, max_y),
    Point(max_x, min_y)
]
bbox_vertices = gpd.GeoSeries(bbox_vertices, crs = METRIC_CRS)


# In[73]:


# plt.figure(figsize = (10, 10))
ax = plt.subplot(1, 1, 1)
utils.plot_utils.plot_map(ax, geodf_nexus)
utils.plot_utils.plot_square(ax, bbox_vertices)
plt.show()


# ### Create grid: find census tracts clusters in grid
# To assess where an arbitrary point $(x,y)$ resides in the grid, it is possible to calculate the i-th and j-th square it is in as follows:
# 
# $i = \lfloor \frac{x - min_x}{size}  \rfloor$
# 
# $j = \lfloor \frac{y - min_y}{size} \rfloor$
# 
# For each census tract, it is necessary to calculate in which squares in the grid it resides. This is possible by obtaining the four vertices that form its bounding box and, then, calculating the four clusters squares in the grid that delimit that bounding box.
# 
# Those squares are
#  $(i_{min}, j_{min})$,
#  $(i_{min}, j_{max})$,
#  $(i_{max}, j_{max})$ and 
#  $(i_{max}, j_{min})$.
# 
# <img src="https://drive.google.com/uc?export=view&id=1KrLNsN4H59xN92tTdHdV1JJ2olTerxip"/>

# In[74]:


cluster_image_lenght = 9e3 # 6.7e3


# In[75]:


# from shapely.geometry import MultiPolygon

# # Supondo que gdf seja seu GeoDataFrame
# gdf = geodf_nexus.copy()
# gdf['geometry'] = gdf['geometry'].apply(lambda x: MultiPolygon([x]) if x.type == 'Polygon' else x)


# In[76]:


def calculate_bounding_squares(
        geometry, # shapely.geometry.multipolygon.MultiPolygon
        size: float,
        min_x: float,
        min_y: float
    ):
    """
        Take a MultiPolygon and find in which grid squares its bounding box resides. 
    """
    cluster_min_x, cluster_min_y, cluster_max_x, cluster_max_y = geometry.bounds

    i_min = np.floor((cluster_min_x - min_x) / size).astype(int)
    i_max = np.floor((cluster_max_x - min_x) / size).astype(int)
    j_min = np.floor((cluster_min_y - min_y) / size).astype(int)
    j_max = np.floor((cluster_max_y - min_y) / size).astype(int)
    
    return [i_min, j_min, i_max, j_max]


# In[77]:


geodf_nexus[["i_min", "j_min", "i_max", "j_max"]] = geodf_nexus.apply(
    lambda row: calculate_bounding_squares(
        row["geometry"],
        cluster_image_lenght,
        min_x,
        min_y
      ),
      axis = "columns",
      result_type = "expand"
    )


# ### Create grid: calculate area composition of each square

# First, calculate the portion of each census tract in every cluster square in the grid.

# In[78]:


def create_square_polygon_from_coordinate_indices(
        i: np.int32,
        j: np.int32,
        size: float,
        min_x: float, 
        min_y: float
    ):
    """
        Transform the indices of a cluster square (i,j) into a Polygon representing 
        that square, based on size, which is the side lenght of the square and min_x
        and min_y, which are coordinates of the bounding box of the area of interest.
    """
    x_left = i * size + min_x
    x_right = (i + 1) * size + min_x

    y_top = j * size + min_y
    y_bottom = (j + 1) * size + min_y

    vertices = [
            Point(x_left, y_top),
            Point(x_left, y_bottom),
            Point(x_right, y_bottom),
            Point(x_right, y_top)
    ]

    return Polygon([(point.x, point.y) for point in vertices])


# In[79]:


def calculate_tract_inner_squares_area_portion(
        census_tract: pd.core.series.Series,
        size: float,
        min_x: float, 
        min_y: float
    ):
    """
        Take a census tract and find out in which cluster squares it is inside, by
        taking the intersection of the tract polygon and the squares and calculating
        the portion of the square area that is composed by that tract. This value is
        stored in a dictionary specific for that census tract (cod_sector), for each
        cluster square (i,j).
    """
    try:
        tract_portion_in_squares = {}
        tract_polygon = census_tract["geometry"]
        for i in range(census_tract["i_min"], census_tract["i_max"] + 1):
            for j in range(census_tract["j_min"], census_tract["j_max"] + 1):
                square_polygon = create_square_polygon_from_coordinate_indices(i, j, size, min_x, min_y)
                intersection_polygon = tract_polygon.intersection(square_polygon)
                portion = intersection_polygon.area / square_polygon.area
                if portion > 0:
                    tract_portion_in_squares[Point(i, j)] = portion

        return tract_portion_in_squares
    except:
        return None


# In[ ]:


geodf_nexus["portions"] = geodf_nexus.apply(
  lambda row: calculate_tract_inner_squares_area_portion(
      row,
      cluster_image_lenght,
      min_x,
      min_y
  ),
  axis = "columns",
)


# In[ ]:


geodf_nexus["portions"].isnull().values.any()


# Then, create a single structure containing all the proportions by cluster square.
# 
# The structure is a dict `cluster_portions` where each key is a square that composes the grid, represented by the tuple `(i,j)`. Each value is another dict where keys are the `cod_sector` of a tract that composes that square and its value is the area portion of the square that is composed by the given tract.
# 
# There is an additional item for each square, `total`, which is the sum of all portions. Thus, it goes from `0.0` to `1.0`.
# 
# PS.: this can be further optimized using parallelism.

# In[ ]:


cluster_portions = {}
for i in tqdm(range(geodf_nexus.shape[0])):
    census_tract = geodf_nexus.iloc[i]
    tract_portions = census_tract["portions"]
    for square, portion in tract_portions.items():
        square_portions = cluster_portions.setdefault(square, {})
        if len(square_portions) == 0:
            square_portions["total"] = 0.0
        square_portions[census_tract["cod_sector"]] = portion
        square_portions["total"] += portion
        cluster_portions[square] = square_portions


# In[ ]:


len(cluster_portions)


# ## Find possible clusters
# With the cluster squares that represent the grid and their area composition based on each census tract, it is possible to disregard those that present very large empty areas, based on an arbitrary threshold.
# 
# Then, for the remaining clusters, the indicator target values are calculated as the weighted average of the values, with the weights being the area proportions of each sector.

# ### Filter clusters by coverage area

# Based on experimentation, there are 76365 possible clusters. The following table displays how the threshold affects that total due to filtering:
# 
# `thresold`   | clusters
# -------------|-------------
# 1.00         | 47743
# 0.99         | 66762
# 0.95         | 68289
# 0.80         | 70446

# In[ ]:


area_threshold = 1.00


# In[ ]:


cluster_portions_filtered = {}
for square, portion in cluster_portions.items():
  if portion["total"] >= area_threshold:
    cluster_portions_filtered[square] = portion


# In[ ]:


len(cluster_portions_filtered)


# ### Calculate target indicators

# In[ ]:


clusters = []

indicators = ["income", "literacy", "longevity"]
for square, portion in tqdm(cluster_portions_filtered.items()):
  indicator_values = np.zeros(len(indicators))
  total = portion["total"]
  for cod_sector, area_percentage in portion.items():
    if cod_sector != "total":
      tract_indicator_values = geodf_nexus[geodf_nexus["cod_sector"] == cod_sector][indicators].values[0]
      indicator_values += np.array(tract_indicator_values) * area_percentage / total
  cluster_sample = dict(zip(indicators, indicator_values))
  cluster_sample["indices"] = square
  clusters.append(cluster_sample)


# In[ ]:


df_clusters = pd.DataFrame.from_dict(clusters)


# In[ ]:


df_clusters.head()


# In[ ]:


df_clusters


# ## Store data

# In[ ]:


df_clusters["geometry"] = df_clusters.apply(
    lambda cluster: create_square_polygon_from_coordinate_indices(
        cluster["indices"].x,
        cluster["indices"].y,
        cluster_image_lenght,
        min_x,
        min_y
    ),
    axis = "columns"
)


# In[ ]:


df_clusters["WKT"] = df_clusters.apply(lambda cluster: dumps(cluster["geometry"]), axis = "columns")


# In[ ]:


geodf_clusters = gpd.GeoDataFrame(
    df_clusters,
    crs = METRIC_CRS,
    geometry = [loads(mpoly) for mpoly in df_clusters["WKT"]])
geodf_clusters = geodf_clusters.to_crs(WGS84_CRS)


# In[ ]:


geodf_clusters[["lon", "lat"]] = geodf_clusters.apply(
    lambda cluster: [cluster["geometry"].centroid.x, cluster["geometry"].centroid.y],
    axis = "columns",
    result_type = "expand"
)


# In[ ]:


geodf_clusters["country"] = "brazil"
geodf_clusters["year"] = 2010


# In[ ]:


geodf_clusters_clean = geodf_clusters[[
    "country",
    "year",
    "lat",
    "lon",
    "income",
    "literacy",
    "longevity"
  ]]


# In[ ]:


geodf_clusters_clean.describe()


# In[ ]:


geodf_clusters_clean.info()


# In[ ]:


geodf_clusters_clean.head()


# In[100]:


geodf_clusters_clean.to_csv(f"../data/processed/clusters_data_{cluster_image_lenght}km.csv", index = False)


# In[101]:


geodf_clusters_clean.to_csv(f"../data/processed/clusters_data_{cluster_image_lenght}km.csv", index = False)


# ## Plot map

# In[102]:


plt.figure(figsize = (20, 20))
ax = plt.subplot(1, 1, 1)
utils.plot_utils.plot_map(ax, geodf_nexus.to_crs(WGS84_CRS))
geodf_clusters.boundary.plot(ax = ax)
ax.set_xlim(-53, -36)
ax.set_ylim(-11, -7)
plt.show()


# In[103]:


plt.figure(figsize = (20, 20))
ax = plt.subplot(1, 1, 1)
plot_utils.plot_map(ax, geodf_nexus.to_crs(WGS84_CRS))
geodf_clusters.boundary.plot(ax = ax)
ax.set_xlim(-53, -36)
ax.set_ylim(-11, -7)
plt.show()


# In[ ]:


geodf_clusters_clean


# In[ ]:




