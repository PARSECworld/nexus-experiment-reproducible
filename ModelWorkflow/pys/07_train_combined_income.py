#!/usr/bin/env python
# coding: utf-8

# ## Pre-requisites
# 
# Before running this notebook, you should have already used the `extract_features.py` script to extract features from models trained on DHS data. You should have the following structure under the `final_ex/income/` directory:
# 
# ```
# dhs_incountry/
#     DHS_Incountry_A_ms_samescaled_b64_fc01_conv01_lr001/
#         features.npz
#     ...
#     DHS_Incountry_C_ms_samescaled_250_b64_fc1.0_conv1/
#         features.npz
#     ...
#     DHS_Incountry_E_nl_random_b64_fc01_conv01_lr001/
#         features.npz
# ```
# 
# ## Instructions
# 
# This notebook essentially performs fine-tuning of the final-layer of the Resnet-18 models. However, instead of directly fine-tuning the Resnet-18 models in TensorFlow, we train ridge-regression models using the extracted features. We take this approach for two reasons:
# 
# 1. It is easier to perform leave-one-group-out ("logo") cross-validated ridge regression using scikit-learn, as opposed to TensorFlow. For out-of-country (OOC) experiments, the left-out group is the test country. For in-country experiments, the left-out group is the test split.
# 2. We can concatenate the 512-dim features from the RGB/MS CNN models with the 512-dim features from the NL CNN models to form a larger 1024-dim feature vector capturing RGB/MS + NL imagery information. We do this instead of training a CNN with the MS+NL imagery stacked together as an input because we found it to result in better performance.
# 

# ## Imports and Constants

# ### Install Libraries

# ### Imports

# In[2]:


#from __future__ import annotations

from collections.abc import Iterable
import os
import pickle

import numpy as np
import pandas as pd

from batchers import dataset_constants
from models.linear_model import ridge_cv
from utils.general import load_npz


# ### Important constants and Paths

# In[3]:


FOLDS = ['A', 'B', 'C', 'D', 'E']
SPLITS = ['train', 'val', 'test']
label = 'income'
OUTPUTS_ROOT_DIR = f'/home/sagemaker-user/reproducible/logs/{label}'
INPUTS_ROOT_DIR = f'/home/sagemaker-user/reproducible/final_ex/{label}'
INPUTS_ROOT_DIR_RANDOM = f'/home/sagemaker-user/reproducible/final_ex'
COUNTRIES = dataset_constants.DHS_COUNTRIES

KEEPS = [0.05, 0.1, 0.25, 0.5]
SEEDS = [123, 456, 789]


# ## Load data

# In[4]:


MODEL_DIRS = {
    #Incountry models - NightLight
    
    'incountry_resnet_nl_A':'DHS_Incountry_A_nl_random_b64_fc1.0_conv1.0_lr0001',
    'incountry_resnet_nl_B':'DHS_Incountry_B_nl_random_b64_fc1.0_conv1.0_lr0001',
    'incountry_resnet_nl_C':'DHS_Incountry_C_nl_random_b64_fc1.0_conv1.0_lr0001',
    'incountry_resnet_nl_D':'DHS_Incountry_D_nl_random_b64_fc1.0_conv1.0_lr0001',
    'incountry_resnet_nl_E':'DHS_Incountry_E_nl_random_b64_fc1_conv1.0_lr001',
    
    # Incountry models - MultiSpectral
    
    'incountry_resnet_ms_A':'DHS_Incountry_A_ms_samescaled_200_b64_fc01_conv01_lr001',
    'incountry_resnet_ms_B':'DHS_Incountry_B_ms_samescaled_200_b64_fc1_conv1_lr001',
    'incountry_resnet_ms_C':'DHS_Incountry_C_ms_samescaled_200_b64_fc1.0_conv1.0_lr0001',
    'incountry_resnet_ms_D':'DHS_Incountry_D_ms_samescaled_200_b64_fc001_conv1.0_lr0001',
    'incountry_resnet_ms_E':'DHS_Incountry_E_ms_samescaled_200_b64_fc001_conv001_lr0001',
    
    # Incountry models - MultiSpectral with 250 epochs
    
    #'incountry_resnet_ms_250_A':'DHS_Incountry_A_ms_samescaled_250_b64_fc01_conv01_lr001',
    #'incountry_resnet_ms_250_B':'DHS_Incountry_B_ms_samescaled_250_b64_fc1_conv1_lr001',
    #'incountry_resnet_ms_250_C':'DHS_Incountry_C_ms_samescaled_250_b64_fc1.0_conv1.0_lr0001',
    #'incountry_resnet_ms_250_D':'DHS_Incountry_D_ms_samescaled_250_b64_fc001_conv1.0_lr0001',
    #'incountry_resnet_ms_250_E':'DHS_Incountry_E_ms_samescaled_250_b64_fc001_conv001_lr0001',
}


# `country_labels` is a `np.ndarray` that shows which country each cluster belongs to. Countries are indexed by their position in `dataset_constants.DHS_COUNTRIES`.
# ```python
# array([ 0,  0,  0, ..., 22, 22, 22])
# ```
# 
# `incountry_group_labels` is a `np.ndarray` that shows which "test" fold each cluster belongs to. The first cluster belongs to the "test" split of fold "B" (folds here are 0-indexed).
# ```python
# array([1, 1, 4, ..., 1, 0, 3])
# ```

# ### Check Number of Countries

# In[5]:


countries=["brazil"]

print(len(countries))


# ### Check Indicators

# In[6]:


FOLDS = ["A", "B", "C", "D", "E"]
indicators = ['income','longevity','literacy']
for label in indicators:
    df = pd.read_csv('/home/sagemaker-user/reproducible/data/processed/clusters_data_9000.0km.csv', float_precision='high', index_col=False)
    labels = df[label].to_numpy(dtype=np.float32)
    print(labels)


# ### Extract Indicators, location and country names in the correct order

# In[7]:


label = 'income'

print(f'for label {label}:')
print(f'dhs_co_{label}.npz Dictionary Key, Value Type and Shape')
df = pd.read_csv('/home/sagemaker-user/reproducible/data/processed/clusters_data_9000.0km.csv', float_precision='high', index_col=False)
npz = load_npz('/home/sagemaker-user/reproducible/data/processed/dhs_co_income.npz')
labels = npz['labels']
print(labels)
print(len(labels))
locs = npz['locs']
country_labels = df['country'].map(countries.index).to_numpy()
print(country_labels)


# ### Get the test split indices for each fold.

# In[8]:


with open('/home/sagemaker-user/reproducible/data/processed/dhs_incountry_co.pkl', 'rb') as f:
    incountry_folds = pickle.load(f)
incountry_group_labels = np.zeros(len(df), dtype=np.int64)
print('Indices for each Fold')
print()
for i, fold in enumerate(FOLDS):
    test_indices = incountry_folds[fold]['test']
    print(f'For Fold {FOLDS[i]}')
    print(test_indices)
    incountry_group_labels[test_indices] = i
    


# Figure - Incountry Clusters split by Folds
# ![incountry_folds.png](attachment:b8b8361e-752e-432f-aef6-36195cd38af5.png)

# ## Incountry

# ### Ridge Regression using only the best set

# In[9]:


f = 'E'
num_examples = len(labels)
model_name = 'incountry_resnet_ms'
model_fold_name = f'{model_name}_{f}'
print(model_fold_name)
model_dir = MODEL_DIRS[model_fold_name]
npz_path = os.path.join(INPUTS_ROOT_DIR, 'dhsincountry', model_dir, 'features.npz')
print(npz_path)
print(labels)
npz = load_npz(npz_path, check={'labels': labels})
features = npz['features']


# In[10]:


#!!
# Tamanho máximo permitido com base no shape de features
max_size = features.shape[0]

# Para cada fold, removendo índices que são maiores que o permitido
for fold in incountry_folds:
    # Filtrar índices de treino
    train_indices = incountry_folds[fold]['train']
    filtered_train_indices = [idx for idx in train_indices if idx < max_size]
    incountry_folds[fold]['train'] = filtered_train_indices

    # Filtrar índices de validação
    val_indices = incountry_folds[fold]['val']
    filtered_val_indices = [idx for idx in val_indices if idx < max_size]
    incountry_folds[fold]['val'] = filtered_val_indices

    # Filtrar índices de teste
    test_indices = incountry_folds[fold]['test']
    filtered_test_indices = [idx for idx in test_indices if idx < max_size]
    incountry_folds[fold]['test'] = filtered_test_indices

    print(f"Fold {fold} - Índices de treino, validação e teste atualizados")


# In[11]:


from models.linear_model import ridge_cv, train_linear_model
import sklearn.linear_model


# In[12]:



X = features[np.arange(num_examples)]
y = labels[np.arange(num_examples)]
X_train = features[incountry_folds[f]['train']]
y_train = labels[incountry_folds[f]['train']]
X_val = features[incountry_folds[f]['val']]
y_val = labels[incountry_folds[f]['val']]
X_test = features[incountry_folds[f]['test']]
y_test = labels[incountry_folds[f]['test']]

best_model, best_train_preds, best_val_preds = train_linear_model(X_train, y_train, X_val, y_val,
                       linear_model=sklearn.linear_model.Ridge,
                       plot_alphas=False, optimize='r2')

test_preds = best_model.predict(X)

savedir = os.path.join(OUTPUTS_ROOT_DIR, 'resnet_ms/only_e')
filename = 'test_preds.npz'
npz_path = os.path.join(savedir, filename)
save_dict = {}
if savedir is not None:
        os.makedirs(savedir, exist_ok=True)

        # build up save_dict
        if 'labels' in save_dict:
            assert np.array_equal(labels, save_dict['labels'])
        save_dict['labels'] = labels
        save_dict['test_preds'] = test_preds

        print('saving test preds to:', npz_path)
        np.savez_compressed(npz_path, **save_dict)


# In[13]:


# !!# for fold in incountry_folds:
#     train_indices = incountry_folds[fold]['train']
#     val_indices = incountry_folds[fold]['val']
#     if max(train_indices) >= features.shape[0] or max(val_indices) >= features.shape[0]:
#         print(f"Índice fora do limite no fold {fold}")

# print(features.shape)


# ### Ridge Regression with k-Fold Cross Validation for ResNet-NL and ResNet-MS models

# In[14]:


def ridgecv_incountry_wrapper(model_name: str, savedir: str) -> None:
    '''
    Args
    - model_name: str, corresponds to keys in MODEL_DIRS (without the fold suffix)
    - savedir: str, path to directory for saving ridge regression weights and predictions
    '''
    print(INPUTS_ROOT_DIR)
    features_dict = {}
    for f in FOLDS:
        model_fold_name = f'{model_name}_{f}'
        
        model_dir = MODEL_DIRS[model_fold_name]
        if 'random' in model_dir:
            npz_path = os.path.join(INPUTS_ROOT_DIR_RANDOM, 'dhsincountry', model_dir, 'features.npz')
        else:
            npz_path = os.path.join(INPUTS_ROOT_DIR, 'dhsincountry', model_dir, 'features.npz')
        print(labels)
        npz = load_npz(npz_path, check={'labels': labels})
        features_dict[f] = npz['features']

    ridge_cv(
        features=features_dict,
        labels=labels,
        group_labels=incountry_group_labels,
        group_names=FOLDS,
        do_plot=True,
        savedir=savedir,
        save_weights=True,
        verbose=True)


# ### Ridge Regression with k-Fold Cross Validation for ResNet-MS

# In[15]:


import os

npz_path = os.path.join(savedir, 'ridge_weights.npz')
if os.path.exists(npz_path):
    new_npz_path = npz_path.replace('.npz', '_backup.npz')
    os.rename(npz_path, new_npz_path)
    print(f"Arquivo {npz_path} renomeado para {new_npz_path}")

import glob
import os

# Caminho para os arquivos .npz no diretório de saída
npz_files = glob.glob(os.path.join(savedir, '*.npz'))

# Remover ou renomear todos os arquivos .npz
for npz_file in npz_files:
    os.remove(npz_file)
    print(f"Arquivo {npz_file} removido")


# In[16]:


model_name = 'incountry_resnet_nl'
savedir = os.path.join(OUTPUTS_ROOT_DIR, 'resnet_nl')
ridgecv_incountry_wrapper(model_name, savedir)


# In[17]:


model_name = 'incountry_resnet_ms'
savedir = os.path.join(OUTPUTS_ROOT_DIR, 'resnet_ms')
ridgecv_incountry_wrapper(model_name, savedir)


# # Concatenated MS + NL Features

# In[34]:


def ridgecv_incountry_concat_wrapper(model_names, savedir) -> None:
    '''
    Args
    - model_names: list of str, correspond to keys in MODEL_DIRS (without the fold suffix)
    - savedir: str, path to directory for saving ridge regression weights and predictions
    '''
    features_dict = {}
    for i, f in enumerate(FOLDS):
        concat_features = []  # list of np.array, each shape [N, D_i]
        for model_name in model_names:
            print(model_name)
            model_dir = MODEL_DIRS[f'{model_name}_{f}']
            if 'random' in model_dir:
                npz_path = os.path.join(INPUTS_ROOT_DIR_RANDOM, 'dhsincountry', model_dir, 'features.npz')
            else:
                npz_path = os.path.join(INPUTS_ROOT_DIR, 'dhsincountry', model_dir, 'features.npz')
            npz = load_npz(npz_path, check={'labels': labels})
            concat_features.append(npz['features'])
        concat_features = np.concatenate(concat_features, axis=1)  # shape [N, D_1 + ... + D_m]
        features_dict[f] = concat_features

    ridge_cv(
        features=features_dict,
        labels=labels,
        group_labels=incountry_group_labels,
        group_names=FOLDS,
        do_plot=True,
        savedir=savedir,
        save_weights=True,
        verbose=True)


# In[36]:


model_names = ['incountry_resnet_ms', 'incountry_resnet_nl']
savedir = os.path.join(OUTPUTS_ROOT_DIR, 'resnet_msnl_concat')

# Assuming the training function is executed here
ridgecv_incountry_concat_wrapper(model_names, savedir)

