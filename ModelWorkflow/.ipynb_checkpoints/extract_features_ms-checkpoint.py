from collections import defaultdict
from collections.abc import Callable
import json
import os
from typing import Optional, List, Dict, Tuple

import numpy as np
import tensorflow as tf

from batchers import batcher
from batchers import tfrecord_paths_utils as tfrecord_paths_utils1
from models.resnet_model import Hyperspectral_Resnet
from utils.run import check_existing, run_extraction_on_models

#label = 'literacy'
label = 'income'
OUTPUTS_ROOT_DIR = f'/home/sagemaker-user/reproducible/final_ex/income/dhsincountry'

BATCH_SIZE = 128
KEEP_FRAC = 1.0
IS_TRAINING = True
CACHE = False

from typing import List

DHS_MODELS = [
    # put paths to DHS models here (relative to OUTPUTS_ROOT_DIR)
    
    # 'DHS_Incountry_A_nl_random_b64_fc1.0_conv1.0_lr0001',
    # 'DHS_Incountry_B_nl_random_b64_fc1.0_conv1.0_lr0001',
    # 'DHS_Incountry_C_nl_random_b64_fc1.0_conv1.0_lr0001',
    # 'DHS_Incountry_D_nl_random_b64_fc1.0_conv1.0_lr0001',
    # 'DHS_Incountry_E_nl_random_b64_fc1_conv1.0_lr001',
    
    
    #'DHS_Incountry_A_ms_samescaled_b64_fc01_conv01_lr001',
    #'DHS_Incountry_B_ms_samescaled_b64_fc1_conv1_lr001',
    #'DHS_Incountry_C_ms_samescaled_b64_fc1.0_conv1.0_lr0001',
    #'DHS_Incountry_D_ms_samescaled_b64_fc001_conv1.0_lr0001',
    #'DHS_Incountry_E_ms_samescaled_b64_fc001_conv001_lr0001',

    'DHS_Incountry_A_ms_samescaled_200_b64_fc01_conv01_lr001',
    'DHS_Incountry_B_ms_samescaled_200_b64_fc1_conv1_lr001',
    'DHS_Incountry_C_ms_samescaled_200_b64_fc1.0_conv1.0_lr0001',
    'DHS_Incountry_D_ms_samescaled_200_b64_fc001_conv1.0_lr0001',
    'DHS_Incountry_E_ms_samescaled_200_b64_fc001_conv001_lr0001',  
    
    
    #'DHS_Incountry_A_ms_samescaled_250_b64_fc01_conv01_lr001',
    #'DHS_Incountry_B_ms_samescaled_250_b64_fc1_conv1_lr001',
    #'DHS_Incountry_C_ms_samescaled_250_b64_fc1.0_conv1.0_lr0001',
    #'DHS_Incountry_D_ms_samescaled_250_b64_fc001_conv1.0_lr0001',
    #'DHS_Incountry_E_ms_samescaled_250_b64_fc001_conv001_lr0001',
    
#     'DHS_Incountry_A_ms_samescaled_RN50_200_b64_fc01_conv01_lr001',
#     'DHS_Incountry_B_ms_samescaled_RN50_200_b64_fc1_conv1_lr001',
#     'DHS_Incountry_C_ms_samescaled_RN50_200_b64_fc1.0_conv1.0_lr0001',
#     'DHS_Incountry_D_ms_samescaled_RN50_200_b64_fc001_conv1.0_lr0001',
#     'DHS_Incountry_E_ms_samescaled_RN50_200_b64_fc001_conv001_lr0001',  
   
    # put paths to DHSNL models here (for transfer learning)
    # - NOTE: when extracting features for transfer learning models,
    #   set MODEL_PARAMS['num_outputs'] = 2. The transfer learning models output
    #   predictions for both DMSP and VIIRS nightlight intensities.
    #'transfer/transfer_nlcenter_ms_b64_fc001_conv001_lr0001',
    #'transfer/transfer_nlcenter_rgb_b64_fc001_conv001_lr0001',

    # get paths for DHS OOC keep-frac models
    # TODO
]

LSMS_MODELS = [
    # put paths to LSMS models here (relative to OUTPUTS_ROOT_DIR)
    # TODO
]

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

MODEL_PARAMS = {
    'fc_reg': 5e-3,
    'conv_reg': 5e-3,
    'num_layers': 18,
    'num_outputs': 1,
    'is_training': IS_TRAINING,
}

def get_model_class(model_arch: str) -> Callable:
    if model_arch == 'resnet':
        model_class = Hyperspectral_Resnet
    else:
        raise ValueError('Unknown model_arch. Currently only "resnet" is supported.')
    return model_class

def get_batcher(dataset: str, ls_bands: str, nl_band: str, num_epochs: int,
                cache: bool) -> Tuple[batcher.Batcher, int, dict]:
    if dataset == 'dhs':
        tfrecord_paths = tfrecord_paths_utils1.dhs()
    elif dataset == 'lsms':
        tfrecord_paths = tfrecord_paths_utils.lsms()
    else:
        raise ValueError(f'dataset={dataset} is unsupported')

    size = len(tfrecord_paths)
    tfrecord_paths_ph = tf.placeholder(tf.string, shape=[size])
    feed_dict = {tfrecord_paths_ph: tfrecord_paths}

    if dataset == 'dhs':
        b = batcher.Batcher(
            tfrecord_files=tfrecord_paths_ph,
            label_name=label,
            ls_bands=ls_bands,
            nl_band=nl_band,
            nl_label=None,
            batch_size=BATCH_SIZE,
            epochs=num_epochs,
            normalize='DHS',
            shuffle=False,
            augment=False,
            clipneg=True,
            cache=(num_epochs > 1) and cache,
            num_threads=5)
    else:
        raise NotImplementedError

    return b, size, feed_dict

def read_params_json(model_dir: str, keys: List[str]) -> Tuple:
    json_path = os.path.join(model_dir, 'params.json')
    with open(json_path, 'r') as f:
        params = json.load(f)
    for k in keys:
        if k not in params:
            print(f'Did not find key "{k}" in {model_dir}/params.json. Setting to None.')
    result = tuple(params.get(k, None) for k in keys)
    return result

def main() -> None:
    for model_dirs in [DHS_MODELS, LSMS_MODELS]:
        print(OUTPUTS_ROOT_DIR)
        if not check_existing(model_dirs,
                              outputs_root_dir=OUTPUTS_ROOT_DIR,
                              test_filename='features.npz'):
            print('Stopping')
            return

    all_models = {'dhs': DHS_MODELS, 'lsms': LSMS_MODELS}
    models_by_config = defaultdict(list)  # Dict adaptado

    for dataset, model_dirs in all_models.items():
        for model_dir in model_dirs:
            ls_bands, nl_band, model_arch = read_params_json(
                model_dir=os.path.join(OUTPUTS_ROOT_DIR, model_dir),
                keys=['ls_bands', 'nl_band', 'model_name'])
            config = (dataset, ls_bands, nl_band, model_arch)
            models_by_config[config].append(model_dir)

    for config, model_dirs in models_by_config.items():
        dataset, ls_bands, nl_band, model_arch = config
        print('====== Current Config: ======')
        print('- dataset:', dataset)
        print('- ls_bands:', ls_bands)
        print('- nl_band:', nl_band)
        print('- model_arch:', model_arch)
        print('- number of models:', len(model_dirs))
        print()

        b, size, feed_dict = get_batcher(
            dataset=dataset, ls_bands=ls_bands, nl_band=nl_band,
            num_epochs=len(model_dirs), cache=CACHE)
        batches_per_epoch = int(np.ceil(size / BATCH_SIZE))

        run_extraction_on_models(
            model_dirs,
            ModelClass=get_model_class(model_arch),
            model_params=MODEL_PARAMS,
            batcher=b,
            batches_per_epoch=batches_per_epoch,
            out_root_dir=OUTPUTS_ROOT_DIR,
            save_filename='features.npz',
            batch_keys=['labels', 'locs', 'years'],
            feed_dict=feed_dict)

if __name__ == '__main__':
    main()
