#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Imports-and-Constants" data-toc-modified-id="Imports-and-Constants-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Imports and Constants</a></span></li><li><span><a href="#Load-histograms-(or-create-if-needed)" data-toc-modified-id="Load-histograms-(or-create-if-needed)-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Load histograms (or create if needed)</a></span><ul class="toc-item"><li><span><a href="#Split-NL-band" data-toc-modified-id="Split-NL-band-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Split NL band</a></span></li><li><span><a href="#Plot-histograms" data-toc-modified-id="Plot-histograms-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Plot histograms</a></span></li></ul></li>

# Prerequisites:
# 1. Run `preprocessing/analyze_tfrecords_dhs.ipynb`.
# 2. Run `data_analysis/dhs.ipynb` to create the `dhs_loc_dict.pkl` and `dhs_incountry_folds.pkl`.

# # Imports and Constants

# Downloading Libraries

# In[2]:


get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')


# In[3]:


get_ipython().system('pip install scikit-learn')


# In[4]:


from collections import defaultdict
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.neighbors

from batchers import batcher1, dataset_constants
from models.histograms import (
    get_per_image_histograms,
    plot_band_hists,
    plot_label_hist,
    split_nl_hist)
from models.linear_model import ridge_cv
from models.knn import knn_cv_opt
from utils.analysis import calc_score, evaluate
from utils.general import load_npz
from utils.plot import scatter_preds


# # Important Constants

# In[5]:


# os.environ['CUDA_VISIBLE_DEVICES'] = ''

DATASET_NAME = 'DHS_OOC_A'
LABEL_NAME = 'wealthpooled'

FOLDS = ['A', 'B', 'C', 'D', 'E']
SPLITS = ['train', 'val', 'test']
COUNTRIES = dataset_constants.DHS_COUNTRIES
print(COUNTRIES)

MEANS = dataset_constants.MEANS_DICT[DATASET_NAME]
STD_DEVS = dataset_constants.STD_DEVS_DICT[DATASET_NAME]
s = dataset_constants.SURVEY_NAMES[DATASET_NAME]

LOGS_ROOT_DIR = "/root/reproducible/data/logs/04_dhs_baselines/" 
print(LOGS_ROOT_DIR)


# # Load histograms (or create if needed)

# Constants to Plot the Histograms

# In[6]:


# BAND_BIN_EDGES = [-10^5, -5.0, -4.9, -4.8, ..., 4.8, 4.9, 5.0, 10^5]
# BAND_BIN_CENTERS = [-5.05, -4.95, -4.85, ..., 4.85, 4.95, 5.05]
BAND_BIN_EDGES = np.concatenate([
    [-1e5],
    np.arange(-5.0, 5.1, 0.1),
    [1e5]
])
BAND_BIN_CENTERS = np.arange(-5.05, 5.15, 0.1)

# LABEL_BIN_EDGES = [-2.0, -1.9, ..., 2.8, 2.9]
# LABEL_BIN_CENTERS = [-1.95, -1.85, ..., 2.75, 2.85]
LABEL_BIN_EDGES = np.arange(-2.0, 3.0, 0.1)
LABEL_BIN_CENTERS = np.arange(-1.95, 2.95, 0.1)

# band name => color for plotting
BAND_COLORS = {
    'BLUE'  : 'blue',
    'GREEN' : 'green',
    'RED'   : 'red',
    'SWIR1' : 'purple',
    'SWIR2' : 'brown',
    'TEMP1' : 'orange',
    'NIR'   : 'pink',
    'NIGHTLIGHTS': 'black',
    'DMSP'  : 'black',
    'VIIRS' : 'gray',
}
BAND_ORDER = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']
BAND_ORDER_NLSPLIT = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'DMSP', 'VIIRS']

BANDS_DICT = {
    'rgb': [0, 1, 2],
    'rgbnl': [0, 1, 2, 7, 8],
    'ms': [0, 1, 2, 3, 4, 5, 6],
    'msnl': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'nl': [7, 8],
}


# In[7]:


def get_batcher(tfrecord_files, label):
    if label == "literacy":
        return batcher1.Batcher(
            tfrecord_files=tfrecord_files,
            dataset=DATASET_NAME,
            batch_size=128,
            ls_bands='ms',
            nl_band='merge',
            label_name="literacy",
            shuffle=False,
            augment=False,
            negatives='zero',
            normalize=True)
    
    elif label == "longevity":
        return batcher1.Batcher(
            tfrecord_files=tfrecord_files,
            dataset=DATASET_NAME,
            batch_size=128,
            ls_bands='ms',
            nl_band='merge',
            label_name="longevity",
            shuffle=False,
            augment=False,
            negatives='zero',
            normalize=True)
    else:
        return batcher1.Batcher(
            tfrecord_files=tfrecord_files,
            dataset=DATASET_NAME,
            batch_size=128,
            ls_bands='ms',
            nl_band='merge',
            label_name="income",
            shuffle=False,
            augment=False,
            negatives='zero',
            normalize=True)


# Function that Creates the .npz Files - Containing the information to plot the histograms

# In[8]:


def generate_image_histogram(label):
    #file_path = f"/home/sagemaker-user/reproducible/Datasets/SelectedClusters/dhs_co_{label}.npz"
    file_path = f"/home/sagemaker-user/reproducible/data/processed/dhs_co_{label}.npz"
    print(file_path)
    
    if label == "longevity":
        if not os.path.exists(file_path):
            # THIS REQUIRES >= 35 GB RAM
            tfrecord_paths = batcher1.get_tfrecord_paths(dataset=DATASET_NAME, split='all')
            init_iter, batch_op = get_batcher(tfrecord_paths,label).get_batch()
            results = get_per_image_histograms(init_iter, batch_op, band_bin_edges=BAND_BIN_EDGES)
            print(results)
            print('Saving image histograms to', file_path)
            np.savez_compressed(file_path, **results)
            
    elif label == "literacy":
        if not os.path.exists(file_path):
            # THIS REQUIRES >= 35 GB RAM
            tfrecord_paths = batcher1.get_tfrecord_paths(dataset=DATASET_NAME, split='all')
            init_iter, batch_op = get_batcher(tfrecord_paths,label).get_batch()
            results = get_per_image_histograms(init_iter, batch_op, band_bin_edges=BAND_BIN_EDGES)
            print(results)
            print('Saving image histograms to', file_path)
            np.savez_compressed(file_path, **results)
    else :       
        if not os.path.exists(file_path):
            # THIS REQUIRES >= 35 GB RAM
            tfrecord_paths = batcher1.get_tfrecord_paths(dataset=DATASET_NAME, split='all')
            init_iter, batch_op = get_batcher(tfrecord_paths,label).get_batch()
            results = get_per_image_histograms(init_iter, batch_op, band_bin_edges=BAND_BIN_EDGES)
            print(results)
            print('Saving image histograms to', file_path)
            np.savez_compressed(file_path, **results)
    


# In[9]:


indicators = ["income","longevity","literacy"]
for indicator in indicators:
    generate_image_histogram(indicator)


# Verifying the .npz files

# In[10]:


print(1)


# In[13]:


indicators = ["income","longevity","literacy"]
for label in indicators:
    file_path = f"/home/sagemaker-user/reproducible/data/processed/dhs_co_{label}.npz"
    print(f"para o Indicador {label}:")
    result = load_npz(file_path)
    image_hists = result['image_hists']
    labels = result['labels']

    locs = result['locs']
    years = result['years']
    nls_center = result["nls_center"]

    nls_mean = result['nls_mean']

    dmsp_mask = years < 2012
    viirs_mask = ~dmsp_mask


# ## Split NL band

# In[14]:


indicators = ["income","longevity","literacy"]
for label in indicators:
    file_path = f"/home/sagemaker-user/reproducible/data/processed/dhs_co_{label}.npz"
    print(f"para o Indicador {label}:")
    result = load_npz(file_path)
    image_hists = result['image_hists']
    labels = result['labels']
    image_hists = split_nl_hist(image_hists, years)
    print(image_hists)
    print(years)


# ## Plot histograms

# In[15]:


indicators = ["income","longevity","literacy"]
datasets = [2,3,4]
for i,label in enumerate(indicators):
    file_path = f"/home/sagemaker-user/reproducible/data/processed/dhs_co_{label}.npz"
    print(f"para o Indicador {label}:")
    result = load_npz(file_path)
    
    image_hists = result['image_hists']
    labels = result['labels']

    
    band_hists = np.sum(image_hists, axis=0, dtype=np.int64)  # shape [C, nbins]
    plot_band_hists(
        band_hists,
        BAND_ORDER_NLSPLIT,
        band_colors=BAND_COLORS,
        bin_centers=BAND_BIN_CENTERS,
        xlabel='normalized pixel value',
        ylabel='count',
        title=f'pixel distributions - {label}',
        yscale='log')
    plot_label_hist(
        labels=labels,
        bin_edges=LABEL_BIN_EDGES,
        title=f'label distributions - {label}',
        figsize=(5, 3))


# In[ ]:




