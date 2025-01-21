#!/usr/bin/env python
# coding: utf-8

# ## Google Drive

# ## Imports and Earth Engine Setup

# In[ ]:


import math
from typing import Any, Optional

import ee
import pandas as pd

from preprocessing import ee_utils


# Authentication with the Earth Engine API

# In[ ]:


ee.Authenticate()


# In[ ]:


ee.Initialize()


# ## Constants

# In[ ]:


DHS_EXPORT_FOLDER = 'nexus_reproducible'
CHUNK_SIZE = 30


# In[ ]:


# input data paths
DHS_CSV_PATH = '/home/sagemaker-user/reproducible/data/processed/clusters_data_9000.0km.csv'

# band names
MS_BANDS = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']

# Projection
WGS84_CRS = "EPSG:4326"      # World Geodetic System
# image parameters
PROJECTION = WGS84_CRS    # see Projection Section above.
SCALE = 30                # export resolution: 30m/px
EXPORT_TILE_RADIUS = 127  # image dimension = (2*EXPORT_TILE_RADIUS) + 1 = 255px


# ## Exporting Images

# In[ ]:


from typing import Dict, Tuple, Optional

def export_images(df: pd.DataFrame,
                  sector_type: str,
                  year: int,
                  export_folder: str,
                  chunk_size: Optional[int] = None
                  ) -> Dict[Tuple[str, str, int, int], ee.batch.Task]:
    '''
    Args
    - df: pd.DataFrame, contains columns ['lat', 'lon', 'sector_type', 'year']
    - sector_type: str, together with `year` determines the survey to export
    - year: int, together with `sector_type` determines the survey to export
    - export_folder: str, name of folder for export
    - chunk_size: int, optionally set a limit to the # of images exported per TFRecord file
        - set to a small number (<= 50) if Google Earth Engine reports memory errors

    Returns: dict, maps task name tuple (export_folder, sector_type, year, chunk) to ee.batch.Task
    '''
    subset_df = df[(df['country'] == sector_type) & (df['year'] == year)].reset_index(drop=True)
    if chunk_size is None:
        chunk_size = len(subset_df)
    num_chunks = int(math.ceil(len(subset_df) / chunk_size))
    tasks = {}

    for i in range(num_chunks):
        chunk_slice = slice(i * chunk_size, (i+1) * chunk_size - 1)  # df.loc[] is inclusive
        fc = ee_utils.df_to_fc(subset_df.loc[chunk_slice, :])
        start_date, end_date = ee_utils.surveyyear_to_range(year)

        # create 3-year Landsat composite image
        roi = fc.geometry()
        imgcol = ee_utils.LandsatSR(roi, start_date=start_date, end_date=end_date).merged
        imgcol = imgcol.map(ee_utils.mask_qaclear).select(MS_BANDS)
        img = imgcol.median()

        # add nightlights, latitude, and longitude bands
        img = ee_utils.add_latlon(img)
        img = img.addBands(ee_utils.composite_nl(year))

        fname = f'{sector_type}_{year}_{i:02d}'
        # tasks[(export_folder, sector_type, year, i)] = ee_utils.get_array_patches(
        #     img=img, scale=SCALE, ksize=EXPORT_TILE_RADIUS,
        #     points=fc,
        #     prefix=export_folder, fname=fname,
        #     )
        tasks[(export_folder, sector_type, year, i)] = ee_utils.get_array_patches(
            img=img, scale=SCALE, ksize=EXPORT_TILE_RADIUS,
            points=fc, export=EXPORT,
            prefix=export_folder, fname=fname,
            bucket=BUCKET)
    return tasks


# ## Verifying workflow 
# We are going to drop 'literacy' and 'longevity' and 'TIPO' and renaming 'income' to 'wealthpooled' 

# In[ ]:


dhs_df = pd.read_csv(DHS_CSV_PATH, float_precision='high')


# In[ ]:


dhs_df.shape


# In[ ]:


dhs_surveys = list(dhs_df.groupby(['country', 'year']).groups.keys())
dhs_surveys


# In[ ]:


dhs_df.describe()


# ## Downloading the Images
# Creating the tasks that export the images to Google Drive

# In[ ]:


EXPORT = 'drive'
BUCKET = None
for sector_type, year in dhs_surveys:
    new_tasks = export_images(
        df=dhs_df, sector_type=sector_type, year=year,
        export_folder=DHS_EXPORT_FOLDER, chunk_size=CHUNK_SIZE)
    tasks.update(new_tasks)


# In[ ]:


ee_utils.wait_on_tasks(tasks, poll_interval=60)

