'''
This script trains ResNet CNN models to estimate wealth for DHS and LSMS
locations. Model checkpoints and TensorBoard training logs are saved to
`out_dir`.

Usage:
    python train_direct.py \
        --label_name wealthpooled \
        --model_name resnet --num_layers 18 \
        --lr_decay 0.96 --batch_size 64 \
        --gpu 0 --num_threads 5 \
        --cache train train_eval val \
        --augment True --eval_every 1 --print_every 40 \
        --ooc {ooc} --max_epochs {max_epochs} \
        --out_dir {out_dir} \
        --keep_frac {keep_frac} --seed {seed} \
        --experiment_name {experiment_name} \
        --dataset {dataset} \
        --ls_bands {ls_bands} --nl_band {nl_band} \
        --lr {lr} --fc_reg {reg} --conv_reg {reg} \
        --imagenet_weights_path {imagenet_weights_path} \
        --hs_weight_init {hs_weight_init}

Prerequisites: download TFRecords, process them, and create incountry folds. See
    `preprocessing/1_process_tfrecords.ipynb` and
    `preprocessing/2_create_incountry_folds.ipynb`.
'''

import argparse
import json
import os
import shutil
import time
from pprint import pprint
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  # Disable TensorFlow 2.x behaviors

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://3.235.42.129:5000")

# Import custom modules (ensure these are accessible in your environment)
from batchers import batcher
from batchers import tfrecord_paths_utils
from models.resnet_model import Hyperspectral_Resnet
from utils.run import get_full_experiment_name
from utils.trainer import RegressionTrainer

ROOT_DIR = os.path.dirname(__file__)  # Folder containing this file


def run_training(sess: tf.Session,
                 ooc: bool,
                 dataset: str,
                 keep_frac: float,
                 model_name: str,
                 model_params: Dict[str, Any],
                 batch_size: int,
                 ls_bands: Optional[str],
                 nl_band: Optional[str],
                 label_name: str,
                 augment: bool,
                 learning_rate: float,
                 lr_decay: float,
                 max_epochs: int,
                 print_every: int,
                 eval_every: int,
                 num_threads: int,
                 cache: List[str],
                 out_dir: str,
                 init_ckpt_dir: Optional[str],
                 imagenet_weights_path: Optional[str],
                 hs_weight_init: Optional[str],
                 exclude_final_layer: bool
                 ) -> None:
    '''
    Trains the model and logs metrics and artifacts to MLflow.
    '''
    # ====================
    #    ERROR CHECKING
    # ====================
    assert os.path.exists(out_dir)

    if model_name == 'resnet':
        model_class = Hyperspectral_Resnet
    else:
        raise ValueError('Unknown model_name. Only "resnet" model currently supported.')

    # ====================
    #       BATCHERS
    # ====================
    if ooc:  # Out-of-country split
        if 'dhs' in dataset.lower():
            train_tfrecord_paths = tfrecord_paths_utils.dhs_ooc(dataset, split='train')
            val_tfrecord_paths = tfrecord_paths_utils.dhs_ooc(dataset, split='val')
        else:
            raise ValueError('Out-of-country with LSMS is not currently supported.')
    else:  # In-country split
        if 'dhs' in dataset.lower():
            paths = tfrecord_paths_utils.dhs_incountry(dataset, splits=['train', 'val'])
        elif 'lsms' in dataset.lower():
            paths = tfrecord_paths_utils.lsms_incountry(dataset, splits=['train', 'val'])
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        train_tfrecord_paths = paths['train']
        val_tfrecord_paths = paths['val']

    num_train = len(train_tfrecord_paths)
    num_val = len(val_tfrecord_paths)

    # Adjust dataset sizes based on keep_frac
    if keep_frac < 1.0:
        num_train = int(num_train * keep_frac)
        num_val = int(num_val * keep_frac)

        train_tfrecord_paths = np.random.choice(
            train_tfrecord_paths, size=num_train, replace=False)
        val_tfrecord_paths = np.random.choice(
            val_tfrecord_paths, size=num_val, replace=False)

    print('num_train:', num_train)
    print('num_val:', num_val)

    train_steps_per_epoch = int(np.ceil(num_train / batch_size))
    val_steps_per_epoch = int(np.ceil(num_val / batch_size))

    def get_batcher(tfrecord_paths: tf.Tensor, shuffle: bool, augment: bool,
                    epochs: int, cache: bool) -> batcher.Batcher:
        return batcher.Batcher(
            tfrecord_files=tfrecord_paths,
            label_name=label_name,
            ls_bands=ls_bands,
            nl_band=nl_band,
            batch_size=batch_size,
            epochs=epochs,
            normalize='DHS',  # Adjust if needed
            shuffle=shuffle,
            augment=augment,
            clipneg=True,
            cache=cache,
            num_threads=num_threads)

    train_tfrecord_paths_ph = tf.placeholder(tf.string, shape=[None])
    val_tfrecord_paths_ph = tf.placeholder(tf.string, shape=[None])

    with tf.name_scope('train_batcher'):
        train_batcher = get_batcher(
            train_tfrecord_paths_ph,
            shuffle=True,
            augment=augment,
            epochs=max_epochs,
            cache='train' in cache)
        train_init_iter, train_batch = train_batcher.get_batch()

    with tf.name_scope('train_eval_batcher'):
        train_eval_batcher = get_batcher(
            train_tfrecord_paths_ph,
            shuffle=False,
            augment=False,
            epochs=max_epochs + 1,
            cache='train_eval' in cache)
        train_eval_init_iter, train_eval_batch = train_eval_batcher.get_batch()

    with tf.name_scope('val_batcher'):
        val_batcher = get_batcher(
            val_tfrecord_paths_ph,
            shuffle=False,
            augment=False,
            epochs=max_epochs + 1,
            cache='val' in cache)
        val_init_iter, val_batch = val_batcher.get_batch()

    # ====================
    #        MODEL
    # ====================
    print('Building model...', flush=True)
    model_params['num_outputs'] = 1

    with tf.variable_scope(tf.get_variable_scope()) as model_scope:
        train_model = model_class(train_batch['images'], is_training=True, **model_params)
        train_preds = tf.reshape(train_model.outputs, shape=[-1], name='train_preds')

    with tf.variable_scope(model_scope, reuse=True):
        train_eval_model = model_class(train_eval_batch['images'], is_training=False, **model_params)
        train_eval_preds = tf.reshape(train_eval_model.outputs, shape=[-1], name='train_eval_preds')

    with tf.variable_scope(model_scope, reuse=True):
        val_model = model_class(val_batch['images'], is_training=False, **model_params)
        val_preds = tf.reshape(val_model.outputs, shape=[-1], name='val_preds')

    trainer = RegressionTrainer(
        train_batch, train_eval_batch, val_batch,
        train_model, train_eval_model, val_model,
        train_preds, train_eval_preds, val_preds,
        sess, train_steps_per_epoch, ls_bands, nl_band, learning_rate, lr_decay,
        out_dir, init_ckpt_dir, imagenet_weights_path,
        hs_weight_init, exclude_final_layer, image_summaries=False)

    # Initialize the dataset iterators
    sess.run([train_init_iter, train_eval_init_iter, val_init_iter], feed_dict={
        train_tfrecord_paths_ph: train_tfrecord_paths,
        val_tfrecord_paths_ph: val_tfrecord_paths
    })

    for epoch in range(max_epochs):
        if epoch % eval_every == 0:
            train_metrics = trainer.eval_train(max_nbatches=train_steps_per_epoch)
            val_metrics = trainer.eval_val(max_nbatches=val_steps_per_epoch)

            # Log metrics to MLflow
            mlflow.log_metrics({
                f"train_r2_epoch_{epoch}": train_metrics['r2'],
                f"val_r2_epoch_{epoch}": val_metrics['r2'],
                f"train_mse_epoch_{epoch}": train_metrics['mse'],
                f"val_mse_epoch_{epoch}": val_metrics['mse'],
                # Add other metrics if needed
            }, step=epoch)

        trainer.train_epoch(print_every)

    # Final evaluation
    final_train_metrics = trainer.eval_train(max_nbatches=train_steps_per_epoch)
    final_val_metrics = trainer.eval_val(max_nbatches=val_steps_per_epoch)

    # Log final metrics to MLflow
    mlflow.log_metrics({
        "final_train_r2": final_train_metrics['r2'],
        "final_train_R2": final_train_metrics['R2'],
        "final_train_mse": final_train_metrics['mse'],
        "final_train_rank": final_train_metrics['rank'],
        "final_val_r2": final_val_metrics['r2'],
        "final_val_R2": final_val_metrics['R2'],
        "final_val_mse": final_val_metrics['mse'],
        "final_val_rank": final_val_metrics['rank'],
    })

    # Log final model results
    trainer.log_results()

    # Save the model
    model_save_dir = os.path.join(out_dir, "saved_model")

    if os.path.exists(model_save_dir):
        print(f"Removing existing export directory: {model_save_dir}")
        shutil.rmtree(model_save_dir)

    tf.saved_model.simple_save(
        sess,
        model_save_dir,
        inputs={"input": train_batch['images']},
        outputs={"output": train_preds}
    )

    # Log the directory where the model was saved
    mlflow.log_artifacts(model_save_dir, artifact_path="models")

    # Close the session
    sess.close()


def run_training_wrapper(**params: Any) -> None:
    '''
    Wrapper function to set up the training environment and start the MLflow run.
    '''
    start = time.time()
    print('Current time:', start)

    # Start MLflow run
    with mlflow.start_run(run_name=params['experiment_name']):
        # Log the experiment parameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Reset any existing graph
        tf.reset_default_graph()

        # Set the random seeds
        seed = params['seed']
        np.random.seed(seed)
        tf.set_random_seed(seed)

        # Create the output directory if needed
        full_experiment_name = get_full_experiment_name(
            params['experiment_name'], params['batch_size'],
            params['fc_reg'], params['conv_reg'], params['lr']
        )
        out_dir = os.path.join(params['out_dir'], full_experiment_name)
        params_filepath = os.path.join(out_dir, 'params.json')
        if os.path.exists(params_filepath):
            print(f'File exists. Removing: {params_filepath}')
            os.remove(params_filepath)

        print(f'Outputs directory: {out_dir}')
        os.makedirs(out_dir, exist_ok=True)
        with open(params_filepath, 'w') as config_file:
            json.dump(params, config_file, indent=4)

        # Create session
        if params['gpu'] is None:  # Restrict to CPU only
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(params['gpu'])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        model_params = {
            'fc_reg': params['fc_reg'],
            'conv_reg': params['conv_reg'],
            'use_dilated_conv_in_first_layer': False,
        }

        if params['model_name'] == 'resnet':
            model_params['num_layers'] = params['num_layers']

        print(f"Started MLflow run for experiment: {params['experiment_name']}")

        try:
            run_training(
                sess=sess,
                ooc=params['ooc'],
                dataset=params['dataset'],
                keep_frac=params['keep_frac'],
                model_name=params['model_name'],
                model_params=model_params,
                batch_size=params['batch_size'],
                ls_bands=params['ls_bands'],
                nl_band=params['nl_band'],
                label_name=params['label_name'],
                augment=params['augment'],
                learning_rate=params['lr'],
                lr_decay=params['lr_decay'],
                max_epochs=params['max_epochs'],
                print_every=params['print_every'],
                eval_every=params['eval_every'],
                num_threads=params['num_threads'],
                cache=params['cache'],
                out_dir=out_dir,
                init_ckpt_dir=params['init_ckpt_dir'],
                imagenet_weights_path=params['imagenet_weights_path'],
                hs_weight_init=params['hs_weight_init'],
                exclude_final_layer=params['exclude_final_layer']
            )
        except Exception as e:
            print(f"An error occurred during training: {e}")
            raise e
        finally:
            # Close the session
            sess.close()
            print("Finished MLflow run.")

    # Finalize the time outside the MLflow block
    end = time.time()
    print('End time:', end)
    print('Time elapsed (sec.):', end - start)


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run end-to-end training.')

    # Paths
    parser.add_argument(
        '--experiment_name', default='new_experiment',
        help='Name of experiment being run')
    parser.add_argument(
        '--out_dir', default=os.path.join(ROOT_DIR, 'outputs/'),
        help='Path to output directory for saving checkpoints and TensorBoard logs')

    # Initialization
    parser.add_argument(
        '--init_ckpt_dir',
        help='Path to checkpoint directory from which to initialize weights')
    parser.add_argument(
        '--imagenet_weights_path',
        help='Path to ImageNet weights for initialization')
    parser.add_argument(
        '--hs_weight_init', choices=[None, 'random', 'same', 'samescaled'],
        help='Method for initializing weights of non-RGB bands in 1st conv layer')
    parser.add_argument(
        '--exclude_final_layer', action='store_true',
        help='Whether to exclude the final layer when loading from a checkpoint')

    # Learning parameters
    parser.add_argument(
        '--label_name', default='wealthpooled',
        help='Name of label to use from the TFRecord files')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size')
    parser.add_argument(
        '--augment', action='store_true',
        help='Whether to use data augmentation')
    parser.add_argument(
        '--fc_reg', type=float, default=1e-3,
        help='Regularization penalty factor for fully connected layers')
    parser.add_argument(
        '--conv_reg', type=float, default=1e-3,
        help='Regularization penalty factor for convolution layers')
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate for optimizer')
    parser.add_argument(
        '--lr_decay', type=float, default=1.0,
        help='Decay rate of the learning rate')

    # High-level model control
    parser.add_argument(
        '--model_name', default='resnet', choices=['resnet'],
        help='Name of model architecture')

    # ResNet-only params
    parser.add_argument(
        '--num_layers', type=int, default=18, choices=[18, 34, 50],
        help='Number of ResNet layers')

    # Data params
    parser.add_argument(
        '--dataset', default='DHS_OOC_A',
        help='Dataset to use')
    parser.add_argument(
        '--ooc', action='store_true',
        help='Whether to use out-of-country split')
    parser.add_argument(
        '--keep_frac', type=float, default=1.0,
        help='Fraction of training data to use')
    parser.add_argument(
        '--ls_bands', choices=[None, 'rgb', 'ms'],
        help='Landsat bands to use')
    parser.add_argument(
        '--nl_band', choices=[None, 'merge', 'split'],
        help='Nightlights band')

    # System
    parser.add_argument(
        '--gpu', type=int,
        help='Which GPU to use')
    parser.add_argument(
        '--num_threads', type=int, default=1,
        help='Number of threads for batcher')
    parser.add_argument(
        '--cache', nargs='*', default=[], choices=['train', 'train_eval', 'val'],
        help='List of datasets to cache in memory')

    # Miscellaneous
    parser.add_argument(
        '--max_epochs', type=int, default=150,
        help='Maximum number of epochs for training')
    parser.add_argument(
        '--eval_every', type=int, default=1,
        help='Evaluate the model on the validation set after every so many epochs')
    parser.add_argument(
        '--print_every', type=int, default=40,
        help='Print training statistics after every so many steps')
    parser.add_argument(
        '--seed', type=int, default=123,
        help='Seed for random initialization and shuffling')

    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_training_wrapper(**vars(args))
