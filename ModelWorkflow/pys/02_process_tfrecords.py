#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Pre-requisites" data-toc-modified-id="Pre-requisites-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Pre-requisites</a></span></li><li><span><a href="#Instructions" data-toc-modified-id="Instructions-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Instructions</a></span></li><li><span><a href="#Imports-and-Constants" data-toc-modified-id="Imports-and-Constants-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Imports and Constants</a></span></li><li><span><a href="#Validate-and-Split-Exported-TFRecords" data-toc-modified-id="Validate-and-Split-Exported-TFRecords-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Validate and Split Exported TFRecords</a></span></li><li><span><a href="#Calculate-Mean-and-Std-Dev-for-Each-Band" data-toc-modified-id="Calculate-Mean-and-Std-Dev-for-Each-Band-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Calculate Mean and Std-Dev for Each Band</a></span></li></ul></div>

# ## Pre-requisites
# 
# Go through the [`preprocessing/0_export_tfrecords.ipynb`](./0_export_tfrecords.ipynb) notebook.
# 
# Before running this notebook, you should have the following structure under the `data/` directory:
# 
# ```
# data/
#     dhs_tfrecords_raw/
#         angola_2011_00.tfrecord.gz
#         ...
#         zimbabwe_2015_XX.tfrecord.gz
#     dhsnl_tfrecords_raw/
#         angola_2010_00.tfrecord.gz
#         ...
#         zimbabwe_2016_XX.tfrecord.gz
#     lsms_tfrecords_raw/
#         ethiopia_2011_00.tfrecord.gz
#         ...
#         uganda_2013_XX.tfrecord.gz
# ```
# 
# ## Instructions
# 
# This notebook processes the exported TFRecords as follows:
# 1. Verifies that the fields in the TFRecords match the original CSV files.
# 2. Splits each monolithic TFRecord file exported from Google Earth Engine into one file per record.
# 
# After running this notebook, you should have three new folders (`dhs_tfrecords`, `dhsnl_tfrecords`, and `lsms_tfrecords`) under `data/`:
# 
# ```
# data/
#     dhs_tfrecords/
#         angola_2011/
#             00000.tfrecord.gz
#             ...
#             00229.tfrecord.gz
#         ...
#         zimbabwe_2015/
#             00000.tfrecord.gz
#             ...
#             00399.tfrecord.gz
#     dhsnl_tfrecords/
#         angola_2010/
#             00000.tfrecord.gz
#             ...
#             07734.tfrecord.gz
#         zimbabwe_2016/
#             00000.tfrecord.gz
#             ...
#             03584.tfrecord.gz
#     lsms_tfrecords/
#         ethiopia_2011/
#             00000.tfrecord.gz
#             ...
#             00326.tfrecord.gz
#         uganda_2013/
#             00000.tfrecord.gz
#             ...
#             00164.tfrecord.gz
# ```
# 
# This notebook also calculates the mean and standard deviation of each band across each of the 3 datasets.

# ## Prerequisites

# Installing Libraries 

# In[2]:


#!pip3 install pandas


# In[1]:


get_ipython().system('pip install matplotlib')
get_ipython().system('pip install tqdm')
get_ipython().system('pip install pandas')


# ## Imports and Constants

# In[2]:


from typing import Iterable
from glob import glob
from pprint import pprint
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

# sys.path.append('.')
from batchers import batcher, tfrecord_paths_utils
from preprocessing.helper import (
    analyze_tfrecord_batch,
    per_band_mean_std,
    print_analysis_results)


# In[3]:


import os

# folder path
dir_path = "/home/sagemaker-user/reproducible/data/raw/nexus_tfrecords_raw/"
#dir_path = "/root/reproducible/data/raw/nexus_tfrecords_raw/"
count = 0
# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        count += 1
print('File count:', count)


# Image Bands, to calculate the statistics of its values

# In[4]:


REQUIRED_BANDS = [
    'BLUE', 'GREEN', 'LAT', 'LON', 'NIGHTLIGHTS', 'NIR', 'RED',
    'SWIR1', 'SWIR2', 'TEMP1']

BANDS_ORDER = [
    'BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR',
    'DMSP', 'VIIRS']


# Path to Files

# In[5]:


#BASE_PATH =  "/root/reproducible/data/"
BASE_PATH =  "/home/sagemaker-user/reproducible/data/"
PROCESSED_PATH = BASE_PATH+"processed/"
RAW_PATH = BASE_PATH+"raw/"
# csv File - Only with income Indicator
DATASET_CSV_PATH = PROCESSED_PATH + "clusters_data_9000.0km.csv" 
# Path to raw tfrecords - Downloaded with code 01
DHS_EXPORT_FOLDER = RAW_PATH + "nexus_tfrecords_raw/"
# Path to processed tfrecords - output of this code
DHS_PROCESSED_FOLDER = PROCESSED_PATH + "nexus_tfrecords_processed/"


# Checking how many countries and years are being processed
# 
# In this case, only Brazil - 2010

# In[6]:


DHS_EXPORT_FOLDER


# In[7]:


DHS_EXPORT_FOLDER


# In[8]:


tf_paths = glob(os.path.join(DHS_EXPORT_FOLDER, 'brazil_2010*'))
file_ids = [int(os.path.basename(file).replace('.tfrecord.gz', '')[-3:].replace('_', '')) for file in tf_paths]
file_ids.sort()
last_file_id = file_ids[0]
for index, file_id in enumerate(file_ids):
    if index != int(file_id):
        print(index-1, last_file_id)
        print(index, file_id)
        break;
    last_file_id = file_id


# ## Validate and Split Exported TFRecords

# In[9]:


def validate_and_split_tfrecords_vtf2(tfrecord_paths: Iterable[str], out_dir: str, df: pd.DataFrame) -> None:
    '''
    Validates and splits a list of exported TFRecord files (for a
    given country-year survey) into individual TFrecords,} one per cluster.

    Args
    - tfrecord_paths: Iterable[str], paths to exported TFRecords files
    - out_dir: str, path to dir to save processed individual TFRecords
    - df: pd.DataFrame, index is sequential and starts at 0
    '''
    options = tf.io.TFRecordOptions(compression_type='GZIP')

    # cast float64 => float32 and str => bytes
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == object:  # pandas uses 'object' type for str
            df[col] = df[col].apply(lambda x: x.encode())  # Convert to bytes

    i = 0
    progbar = tqdm(total=len(df))

    for tfrecord_path in tfrecord_paths:
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='GZIP')
        for record_str in dataset:
            ex = tf.train.Example.FromString(record_str.numpy())
            feature_map = ex.features.feature

            # Verificação de bandas requeridas e comparação com valores CSV omitida para simplificação

            # serialize to string and write to file
            out_path = os.path.join(out_dir, f'{i:05d}.tfrecord.gz')
            with tf.io.TFRecordWriter(out_path, options=options) as writer:
                writer.write(ex.SerializeToString())

            i += 1
            progbar.update(1)
    progbar.close()

def process_dataset(csv_path: str, input_dir: str, processed_dir: str) -> None:
    '''
    Args
    - csv_path: str, path to CSV of DHS or LSMS clusters
    - input_dir: str, path to TFRecords exported from Google Earth Engine
    - processed_dir: str, folder where to save processed TFRecords
    '''
    df = pd.read_csv(csv_path, float_precision='high', index_col=False)
    surveys = list(df.groupby(['country', 'year']).groups.keys())  # (country, year) tuples

    for country, year in surveys:
        country_year = f'{country}_{year}'
        print('Processing:', country_year)

        tfrecord_paths = glob(os.path.join(input_dir, country_year + '*'))
        out_dir = os.path.join(processed_dir, country_year)
        os.makedirs(out_dir, exist_ok=True)
        subset_df = df[(df['country'] == country) & (df['year'] == year)].reset_index(drop=True)
        validate_and_split_tfrecords(
            tfrecord_paths=tfrecord_paths, out_dir=out_dir, df=subset_df)


def validate_and_split_tfrecords(
        tfrecord_paths: Iterable[str],
        out_dir: str,
        df: pd.DataFrame
        ) -> None:
    '''
    Validates and splits a list of exported TFRecord files (for a
    given country-year survey) into individual TFrecords, one per cluster.

    "Validating" a TFRecord comprises of 2 parts
    1) verifying that it contains the required bands
    2) verifying that its other features match the values from the dataset CSV

    Args
    - tfrecord_paths: str, path to exported TFRecords files
    - out_dir: str, path to dir to save processed individual TFRecords
    - df: pd.DataFrame, index is sequential and starts at 0
    '''
    # Create an iterator over the TFRecords file. The iterator yields
    # the binary representations of Example messages as strings.
    #options = tf.io.TFRecordOptions(tf.io.TFRecordCompressionType.GZIP)
    options = tf.io.TFRecordOptions(compression_type = 'GZIP')

    # cast float64 => float32 and str => bytes
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == object:  # pandas uses 'object' type for str
            df[col] = df[col].astype(bytes)

    i = 0
    progbar = tqdm(total=len(df))

    for tfrecord_path in tfrecord_paths:
        iterator = tf.io.tf_record_iterator(tfrecord_path, options=options)
        for record_str in iterator:
            # parse into an actual Example message
            ex = tf.train.Example.FromString(record_str)
            feature_map = ex.features.feature

            # Each sample feature has a point geometry and a property named 'elevation'
            # corresponding to the band named 'elevation' of the image. If there are
            # multiple bands they will become multiple properties. This will print:
            #
            # geometry: Point (-110.01, 40.00)
            # properties:
            #   elevation: 1639

            # verify required bands exist
            #for band in REQUIRED_BANDS:
                #assert band in feature_map, f'Band "{band}" not in record {i} of {tfrecord_path}'

            # compare feature map values against CSV values
            csv_feats = df.loc[i, :].to_dict()
            #for col, val in csv_feats.items():
            #    ft_type = feature_map[col].WhichOneof('kind')
            #    ex_val = feature_map[col].__getattribute__(ft_type).value[0]
                #assert val == ex_val, f'Expected {col}={val}, but found {ex_val} instead'

            # serialize to string and write to file
            out_path = os.path.join(out_dir, f'{i:05d}.tfrecord.gz')  # all surveys have < 1e6 clusters
            with tf.io.TFRecordWriter(out_path, options=options) as writer:
                writer.write(ex.SerializeToString())

            i += 1
            #Sprogbar.update(1)
    progbar.close()


# In[ ]:


process_dataset(
    csv_path = DATASET_CSV_PATH,
    input_dir = DHS_EXPORT_FOLDER,
    processed_dir = DHS_PROCESSED_FOLDER)


# In[10]:


DHS_PROCESSED_FOLDER


# Verifying the amount of files created , 20438 to be exact

# In[12]:


import os

# folder path

dir_path = "/home/sagemaker-user/reproducible/data/processed/nexus_tfrecords_processed/brazil_2010/"
count = 0
# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        count += 1
print('File count:', count)


# ## Calculate Mean and Std-Dev for Each Band
# 
# The means and standard deviations calculated here are saved as constants in `batchers/dataset_constants.py` for `_MEANS_DHS`, `_STD_DEVS_DHS`, `_MEANS_LSMS`, and `_STD_DEVS_LSMS`.

# In[13]:


def calculate_mean_std(tfrecord_paths):
    '''Calculates and prints the per-band means and std-devs'''
    iter_init, batch_op = batcher.Batcher(
        tfrecord_files=tfrecord_paths,
        label_name=None,
        ls_bands='ms',
        nl_band='merge',
        batch_size=128,
        shuffle=False,
        augment=False,
        clipneg=False,
        normalize=None).get_batch()

    stats = analyze_tfrecord_batch(
        iter_init, batch_op, total_num_images=len(tfrecord_paths),
        nbands=len(BANDS_ORDER), k=10)
    means, stds = per_band_mean_std(stats=stats, band_order=BANDS_ORDER)

    print('Means:')
    pprint(means)
    print()

    print('Std Devs:')
    pprint(stds)

    print('\n========== Additional Per-band Statistics ==========\n')
    print_analysis_results(stats, BANDS_ORDER)


# In[14]:


tfrecord_paths_utils.dhs()


# In[ ]:


calculate_mean_std(tfrecord_paths_utils.dhs())


# In[ ]:


calculate_mean_std(tfrecord_paths_utils.dhs_nl())


# In[ ]:




