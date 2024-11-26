#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[6]:


get_ipython().system('pip install diagrams')


# In[8]:


get_ipython().system('apt-get update')


# In[10]:


get_ipython().system('apt-get install -y graphviz')


# ## Pre-requisites

# Register a Google account at [https://code.earthengine.google.com](https://code.earthengine.google.com).

# In[2]:


get_ipython().system('pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib')


# In[2]:


import tensorflow as tf


# In[3]:


get_ipython().system('pip install gdown')


# In[4]:


# """
# Step by step to enable the Google Cloud/Google API service:

# 1. Access the Google Cloud Console (https://console.cloud.google.com/).

# 2. Create a new project or select an existing one:
#     - In the top navigation panel, click on "Select a project" and then "New Project" or choose an existing one.

# 3. Enable the Google Drive API for your project:
#     - In the navigation pane, go to "APIs and Services" > "Library".
#     - Search for "Google Drive API" and click to access it.
#     - Click "Enable" to activate the API in your project.

# 4. Configure the credentials for your application:
#     - In the same panel, go to "Credentials".
#     - Click "Create Credentials" and choose "OAuth Client ID".
#     - If prompted, configure the Consent Screen.
#     - Select "Desktop Application" as the application type.
#     - Name your credential and click "Create".

# 5. Download the credentials JSON file:
#     - After creating the OAuth client ID, click the download icon next to the created credential to download the JSON file.
#     - Save this file to your computer. It will be required for the OAuth authentication script.

# 6. Run the OAuth authentication script in your local environment:
#     - The script will open a browser window for you to log in with your Google account and grant the necessary permissions.
#     - After completing authentication, a `token.pickle` file will be generated.

# 7. Transfer the `token.pickle` file to your development environment (such as AWS SageMaker):
#     - This file contains the access credentials and will be used by your script to interact with the Google Drive API.
# """

# # If you want to automatically download files from Google Drive
# AUTOMATIC_DOWNLOAD_FROM_GOOGLEDRIVE = True
# # Path to the credentials file you uploaded
# SERVICE_ACCOUNT_FILE = 'config_files/client_secret.json'
# # Path to token.pickle file
# TOKEN_FILE_NAME = 'config_files/token.pickle'
# # Replace 'your_folder_id' with the folder ID in Google Drive
# GOOGLE_IMAGES_FOLDER_ID = '1cVFy3RQ0LgR_ZCy1tETwEwZlp89hmGCj'


# In[5]:


get_ipython().system('pip install earthengine-api')


# TQDM Library - Progression Bar

# In[6]:


get_ipython().system('pip install tqdm')


# In[ ]:





# ## Google Drive

# ## Imports and Earth Engine Setup

# In[7]:


from __future__ import annotations

import math
from typing import Any, Optional

import ee
import pandas as pd

from preprocessing import ee_utils


# Authentication with the Earth Engine API

# In[8]:


ee.Authenticate()


# In[9]:


ee.Initialize()


# ## Projection

# In[10]:


MERCATOR_CRS = "EPSG:3857"   # Mercator projection CRS.
#METRIC_CRS = "EPSG:32634"   # Metric CRS. Useful for calculations
METRIC_CRS = {"proj":"cea"}
WGS84_CRS = "EPSG:4326"      # World Geodetic System
SIRGAS_CRS = "EPSG:4674"     # SIRGAS 2000, a projection for LATAM


# ## Constants

# In[11]:


# EXPORT = 'drive'
# fname = 'earthengine_api'


# In[12]:


# ========== ADAPT THESE PARAMETERS ==========

# To export to Google Drive, uncomment the next 2 lines
# EXPORT = 'drive'
# BUCKET = None

# To export to s3, uncoment the next lines:


#BUCKET = 'parsec-brazil-images'
#prefix = 's3://parsec-brazil-images/nexus/data'

# export location parameters
# Here we have to specify the full drive path, otherwise it will save using
# the root directory.
DHS_EXPORT_FOLDER = 'nexus_paper_ajusted'
# This Drive folder was used to store all the Clusters recieved by the EE API

# Set CHUNK_SIZE to None to export a single TFRecord file per (country, year). However,
# this may fail if it exceeds Google Earth Engine memory limits. Decrease CHUNK_SIZE
# to a small number (<= 50) until Google Earth Engine stops reporting memory errors
CHUNK_SIZE = 30


# In[13]:


# input data paths
#
DHS_CSV_PATH = '/home/sagemaker-user/reproducible/data/processed/clusters_data_9000.0km.csv'
#DHS_CSV_PATH = '/root/Datasets/SelectedClusters/dataset_clean.csv'

# band names
MS_BANDS = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']

# image parameters
PROJECTION = WGS84_CRS    # see Projection Section above.
SCALE = 30                # export resolution: 30m/px
EXPORT_TILE_RADIUS = 127  # image dimension = (2*EXPORT_TILE_RADIUS) + 1 = 255px


# ## Exporting Images

# In[14]:


def export_images(df: pd.DataFrame,
                  sector_type: str,
                  year: int,
                  export_folder: str,
                  chunk_size: Optional[int] = None
                  ) -> dict[tuple[str, str, int, int], ee.batch.Task]:
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


# In[15]:


tasks: dict[tuple[str, str, int, int], ee.batch.Task] = {}


# ## Verifying workflow 
# We are going to drop 'literacy' and 'longevity' and 'TIPO' and renaming 'income' to 'wealthpooled' 

# In[16]:


dhs_df = pd.read_csv(DHS_CSV_PATH, float_precision='high')
# dhs_df = dhs_df.head(50)
# dhs_df.to_csv("dataset_test.csv", index = False)


# In[17]:


dhs_df.shape


# In[18]:


dhs_surveys = list(dhs_df.groupby(['country', 'year']).groups.keys())
dhs_surveys


# In[19]:


dhs_df.describe()


# ## Downloading the Images
# Creating the tasks that export the images to Google Drive

# In[20]:


# if export == 'gcs':
#     task = ee.batch.Export.table.toCloudStorage(
#         collection=collection,
#         description=fname,
#         bucket=bucket,
#         fileNamePrefix=f'{prefix}/{fname}',
#         fileFormat='TFRecord',
#         selectors=selectors)


# In[21]:


# EXPORT = 'gcs'
# BUCKET = 'parsec-brazil-images'
# prefix = 'nexus/data'
# fname = 'earthengine_api'


# In[22]:


# import os

# os.environ['AWS_ACCESS_KEY_ID'] = 'AKIASH3JGTGZ3DBK3KUL'
# os.environ['AWS_SECRET_ACCESS_KEY'] = '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'


# In[23]:


# dhs_df_backup = dhs_df.copy()


# In[24]:


# dhs_df = dhs_df.iloc[0:10]


# In[25]:


EXPORT = 'drive'
BUCKET = None
for sector_type, year in dhs_surveys:
    new_tasks = export_images(
        df=dhs_df, sector_type=sector_type, year=year,
        export_folder=DHS_EXPORT_FOLDER, chunk_size=CHUNK_SIZE)
    tasks.update(new_tasks)


# In[ ]:


ee_utils.wait_on_tasks(tasks, poll_interval=60)


# In[7]:


import io
import pickle
import os.path
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload 
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

destination_folder = '../Datasets/SelectedClusters/nexus_tfrecords_raw/'

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    
#FILE = 'credentials.json'
if AUTOMATIC_DOWNLOAD_FROM_GOOGLEDRIVE:
    creds = None

    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    if os.path.exists(TOKEN_FILE_NAME):
        with open(TOKEN_FILE_NAME, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                SERVICE_ACCOUNT_FILE, SCOPES)
            creds = flow.run_local_server(open_browser=False)
        # Save the credentials for the next run
        with open(TOKEN_FILE_NAME, 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)

    page_token = None
    while True:
        results = service.files().list(
                q=f"'{GOOGLE_IMAGES_FOLDER_ID}' in parents",
                pageSize=10, fields="nextPageToken, files(id, name)",
                pageToken=page_token).execute()
        items = results.get('files', [])

        if not items:
            print('No files found.')
        else:
            for item in items:
                print(u'{0} ({1})'.format(item['name'], item['id']))
                file_id = item['id']
                request = service.files().get_media(fileId=file_id)
                file_path = os.path.join(destination_folder, item['name'])
                with open(file_path, 'wb') as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
                        print("Download %d%%." % int(status.progress() * 100))

        page_token = results.get('nextPageToken', None)
        if page_token is None:
            break


# In[ ]:


import io
import pickle
import os.path
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload 
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

#FILE = 'credentials.json'
if AUTOMATIC_DOWNLOAD_FROM_GOOGLEDRIVE:
    creds = None

    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    if os.path.exists(TOKEN_FILE_NAME):
        with open(TOKEN_FILE_NAME, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                SERVICE_ACCOUNT_FILE, SCOPES)
            creds = flow.run_local_server(open_browser=False)
        # Save the credentials for the next run
        with open(TOKEN_FILE_NAME, 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)

    page_token = None
    while True:
        results = service.files().list(
                q=f"'{GOOGLE_IMAGES_FOLDER_ID}' in parents",
                pageSize=10, fields="nextPageToken, files(id, name)",
                pageToken=page_token).execute()
        items = results.get('files', [])

        if not items:
            print('No files found.')
        else:
            for item in items:
                print(u'{0} ({1})'.format(item['name'], item['id']))
                file_id = item['id']
                request = service.files().get_media(fileId=file_id)

                with open(item['name'], 'wb') as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
                        print("Download %d%%." % int(status.progress() * 100))

        page_token = results.get('nextPageToken', None)
        if page_token is None:
            break

