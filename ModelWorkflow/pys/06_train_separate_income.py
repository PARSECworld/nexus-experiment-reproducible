#!/usr/bin/env python
# coding: utf-8

# In[2]:


# # to DHS incountry MS B (1996-2019) fait /terminal 

# !/home/sagemaker-user/reproducible/nexus_experiment/bin/python train_direct.py \
#         --label_name income \
#         --hs_weight_init samescaled\
#         --model_name resnet --num_layers 18 \
#         --lr_decay 0.96 --batch_size 64 \
#         --gpu 0 --num_threads 5 \
#         --cache train train_eval val \
#         --augment --eval_every 1 --print_every 40 \
#         --max_epochs 20 \
#         --out_dir /home/sagemaker-user/reproducible/final_ex/income/dhsincountry/ \
#         --keep_frac 1.0 --seed 123 \
#         --experiment_name DHS_Incountry_B_ms_samescaled_20 \
#         --dataset DHS_incountry_B \
#         --ls_band ms \
#         --lr 0.001 --fc_reg 0.1 --conv_reg 0.1 \
#         --imagenet_weights_path /home/sagemaker-user/reproducible/resnet/ImageNet-ResNet18.npz \
#         > /home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_B_ms_samescaled_20.txt 2>&1


# In[ ]:





# # Training for Income

# #### 

# In[ ]:





# In[2]:


# DHS incountry MS A (2000-2019)
#!rm /home/sagemaker-user/reproducible/final_ex/income/dhsincountry/DHS_Incountry_A_ms_samescaled_200_b64_fc01_conv01_lr001/params.json

# Verifica se o arquivo existe e o remove apenas se ele estiver presente
#!if [ -f /home/sagemaker-user/reproducible/final_ex/income/dhsincountry/DHS_Incountry_A_ms_samescaled_200_b64_fc01_conv01_lr001/params.json ]; then rm /home/sagemaker-user/reproducible/final_ex/income/dhsincountry/DHS_Incountry_A_ms_samescaled_200_b64_fc01_conv01_lr001/params.json; fi

# max_epochs 200
import psutil
import time
import threading

# Caminho do arquivo de log
log_file_path = '/home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_A_ms_samescaled_200.txt'

# Monitorar o uso de memória e escrever no arquivo de log
def monitor_memory():
    with open(log_file_path, 'a') as log_file:
        while True:
            mem_info = psutil.virtual_memory()
            log_file.write(f"Uso de memória: {mem_info.percent}%\n")
            log_file.flush()  # Certifique-se de que os dados são gravados no arquivo
            time.sleep(1)

# Executar o comando em segundo plano
def execute_command():
    get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python train_direct.py             --label_name income             --hs_weight_init samescaled            --model_name resnet --num_layers 18             --lr_decay 0.96 --batch_size 64             --gpu 0 --num_threads 5             --cache train train_eval val             --augment --eval_every 1 --print_every 40             --max_epochs 200             --out_dir /home/sagemaker-user/reproducible/final_ex/income/dhsincountry/             --keep_frac 1.0 --seed 123             --experiment_name DHS_Incountry_A_ms_samescaled_200             --dataset DHS_incountry_A             --ls_band ms             --lr 0.001 --fc_reg 0.01 --conv_reg 0.01             --imagenet_weights_path /home/sagemaker-user/reproducible/resnet/ImageNet-ResNet18.npz             > /home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_A_ms_samescaled_200.txt 2>&1')
# Start monitoring memory usage in a separate thread
memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

# Start executing the command
execute_command()


# In[ ]:


import psutil
import time
import threading

# Caminho do arquivo de log
log_file_path = '/home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_B_ms_samescaled_200.txt'

# Monitorar o uso de memória e escrever no arquivo de log
def monitor_memory():
    with open(log_file_path, 'a') as log_file:
        while True:
            mem_info = psutil.virtual_memory()
            log_file.write(f"Uso de memória: {mem_info.percent}%\n")
            log_file.flush()  # Certifique-se de que os dados são gravados no arquivo
            time.sleep(1)

# Executar o comando em segundo plano
def execute_command():
    get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python train_direct.py         --label_name income         --hs_weight_init samescaled        --model_name resnet --num_layers 18         --lr_decay 0.96 --batch_size 64         --gpu 0 --num_threads 5         --cache train train_eval val         --augment --eval_every 1 --print_every 40         --max_epochs 200         --out_dir /home/sagemaker-user/reproducible/final_ex/income/dhsincountry/         --keep_frac 1.0 --seed 123         --experiment_name DHS_Incountry_B_ms_samescaled_200         --dataset DHS_incountry_B         --ls_band ms         --lr 0.001 --fc_reg 0.1 --conv_reg 0.1         --imagenet_weights_path /home/sagemaker-user/reproducible/resnet/ImageNet-ResNet18.npz         >> /home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_B_ms_samescaled_200.txt 2>&1')

# Start monitoring memory usage in a separate thread
memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

# Start executing the command
execute_command()


# In[ ]:


#DHS incountry MS C (1996-2016)

import psutil
import time
import threading

# Caminho do arquivo de log
log_file_path = '/home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_C_ms_samescaled_200.txt'

# Monitorar o uso de memória e escrever no arquivo de log
def monitor_memory():
    with open(log_file_path, 'a') as log_file:
        while True:
            mem_info = psutil.virtual_memory()
            log_file.write(f"Uso de memória: {mem_info.percent}%\n")
            log_file.flush()  # Certifique-se de que os dados são gravados no arquivo
            time.sleep(1)

# Executar o comando em segundo plano
def execute_command():
    get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python train_direct.py             --label_name income             --hs_weight_init samescaled            --model_name resnet --num_layers 18             --lr_decay 0.96 --batch_size 64             --gpu 0 --num_threads 5             --cache train train_eval val             --augment --eval_every 1 --print_every 40             --max_epochs 200             --out_dir /home/sagemaker-user/reproducible/final_ex/income/dhsincountry/             --keep_frac 1.0 --seed 123             --experiment_name DHS_Incountry_C_ms_samescaled_200             --dataset DHS_incountry_C             --ls_band ms             --lr 0.0001 --fc_reg 1.0 --conv_reg 1.0             --imagenet_weights_path /home/sagemaker-user/reproducible/resnet/ImageNet-ResNet18.npz             > /home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_C_ms_samescaled_200.txt 2>&1')
# Start monitoring memory usage in a separate thread
memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

# Start executing the command
execute_command()


# In[ ]:


#DHS incountry MS D (1996-2016)

import psutil
import time
import threading

# Caminho do arquivo de log
log_file_path = '/home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_D_ms_samescaled_200.txt'

# Monitorar o uso de memória e escrever no arquivo de log
def monitor_memory():
    with open(log_file_path, 'a') as log_file:
        while True:
            mem_info = psutil.virtual_memory()
            log_file.write(f"Uso de memória: {mem_info.percent}%\n")
            log_file.flush()  # Certifique-se de que os dados são gravados no arquivo
            time.sleep(1)|

# Executar o comando em segundo plano
def execute_command():
    get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python train_direct.py             --label_name income             --hs_weight_init samescaled            --model_name resnet --num_layers 18             --lr_decay 0.96 --batch_size 64             --gpu 0 --num_threads 5             --cache train train_eval val             --augment --eval_every 1 --print_every 40             --max_epochs 200             --out_dir /home/sagemaker-user/reproducible/final_ex/income/dhsincountry/             --keep_frac 1.0 --seed 123             --experiment_name DHS_Incountry_D_ms_samescaled_200             --dataset DHS_incountry_D             --ls_band ms             --lr 0.0001 --fc_reg 0.001 --conv_reg 1.0             --imagenet_weights_path /home/sagemaker-user/reproducible/resnet/ImageNet-ResNet18.npz             > /home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_D_ms_samescaled_200.txt 2>&1')
# Start monitoring memory usage in a separate thread
memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

# Start executing the command
execute_command()


# In[ ]:


#DHS incountry MS E (1996-2019)

import psutil
import time
import threading

# Caminho do arquivo de log
log_file_path = '/home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_E_ms_samescaled_200.txt'

# Monitorar o uso de memória e escrever no arquivo de log
def monitor_memory():
    with open(log_file_path, 'a') as log_file:
        while True:
            mem_info = psutil.virtual_memory()
            log_file.write(f"Uso de memória: {mem_info.percent}%\n")
            log_file.flush()  # Certifique-se de que os dados são gravados no arquivo
            time.sleep(1)

# Executar o comando em segundo plano
def execute_command():
    get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python train_direct.py             --label_name income             --hs_weight_init samescaled            --model_name resnet --num_layers 18             --lr_decay 0.96 --batch_size 64             --gpu 0 --num_threads 5             --cache train train_eval val             --augment --eval_every 1 --print_every 40             --max_epochs 200             --out_dir /home/sagemaker-user/reproducible/final_ex/income/dhsincountry/             --keep_frac 1.0 --seed 123             --experiment_name DHS_Incountry_E_ms_samescaled_200             --dataset DHS_incountry_E             --ls_band ms             --lr 0.0001 --fc_reg 0.001 --conv_reg 0.001             --imagenet_weights_path /home/sagemaker-user/reproducible/resnet/ImageNet-ResNet18.npz             > /home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_E_ms_samescaled_200.txt 2>&1 ')
# Start monitoring memory usage in a separate thread
memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

# Start executing the command
execute_command()


# ## Extracting Features of Current Training

# In[ ]:


# !/home/sagemaker-user/reproducible/nexus_experiment/bin/python extract_features.py \
# > /home/sagemaker-user/reproducible/logs/extract_features.txt 2>&1 


# # RESNET 50 Test (with Income)

# In[ ]:


1


# In[ ]:


#DHS incountry MS A (2000-2019)
import psutil
import time
import threading

# Caminho do arquivo de log
log_file_path = '/home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_A_ms_samescaled_RN50_200.txt'

# Monitorar o uso de memória e escrever no arquivo de log
def monitor_memory():
    with open(log_file_path, 'a') as log_file:
        while True:
            mem_info = psutil.virtual_memory()
            log_file.write(f"Uso de memória: {mem_info.percent}%\n")
            log_file.flush()  # Certifique-se de que os dados são gravados no arquivo
            time.sleep(1)

# Executar o comando em segundo plano
def execute_command():
    get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python train_direct.py             --label_name income             --hs_weight_init samescaled            --model_name resnet --num_layers 50             --lr_decay 0.96 --batch_size 64             --gpu 0 --num_threads 5             --cache train train_eval val             --augment --eval_every 1 --print_every 40             --max_epochs 2             --out_dir /home/sagemaker-user/reproducible/final_ex/income/dhsincountry/             --keep_frac 1.0 --seed 123             --experiment_name DHS_Incountry_A_ms_samescaled_RN50_200             --dataset DHS_incountry_A             --ls_band ms             --lr 0.001 --fc_reg 0.01 --conv_reg 0.01             --imagenet_weights_path /home/sagemaker-user/reproducible/resnet/ImageNet-ResNet50.npz             > /home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_A_ms_samescaled_RN50_200.txt 2>&1 ')
# Start monitoring memory usage in a separate thread
memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

# Start executing the command
execute_command()


# In[ ]:


# to DHS incountry MS B (1996-2019) fait /terminal 

import psutil
import time
import threading

# Caminho do arquivo de log
log_file_path = '/home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_B_ms_samescaled_RN50_200.txt'

# Monitorar o uso de memória e escrever no arquivo de log
def monitor_memory():
    with open(log_file_path, 'a') as log_file:
        while True:
            mem_info = psutil.virtual_memory()
            log_file.write(f"Uso de memória: {mem_info.percent}%\n")
            log_file.flush()  # Certifique-se de que os dados são gravados no arquivo
            time.sleep(1)

# Executar o comando em segundo plano
def execute_command():
    get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python train_direct.py             --label_name income             --hs_weight_init samescaled            --model_name resnet --num_layers 50             --lr_decay 0.96 --batch_size 64             --gpu 0 --num_threads 5             --cache train train_eval val             --augment --eval_every 1 --print_every 40             --max_epochs 2             --out_dir /home/sagemaker-user/reproducible/final_ex/income/dhsincountry/             --keep_frac 1.0 --seed 123             --experiment_name DHS_Incountry_B_ms_samescaled_RN50_200             --dataset DHS_incountry_B             --ls_band ms             --lr 0.001 --fc_reg 0.1 --conv_reg 0.1             --imagenet_weights_path /home/sagemaker-user/reproducible/resnet/ImageNet-ResNet50.npz             > /home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_B_ms_samescaled_RN50_200.txt 2>&1 ')
# Start monitoring memory usage in a separate thread
memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

# Start executing the command
execute_command()


# In[ ]:


#DHS incountry MS C (1996-2016)

import psutil
import time
import threading

# Caminho do arquivo de log
log_file_path = '/home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_C_ms_samescaled_RN50_200.txt'

# Monitorar o uso de memória e escrever no arquivo de log
def monitor_memory():
    with open(log_file_path, 'a') as log_file:
        while True:
            mem_info = psutil.virtual_memory()
            log_file.write(f"Uso de memória: {mem_info.percent}%\n")
            log_file.flush()  # Certifique-se de que os dados são gravados no arquivo
            time.sleep(1)

# Executar o comando em segundo plano
def execute_command():
    get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python train_direct.py             --label_name income             --hs_weight_init samescaled            --model_name resnet --num_layers 50             --lr_decay 0.96 --batch_size 64             --gpu 0 --num_threads 5             --cache train train_eval val             --augment --eval_every 1 --print_every 40             --max_epochs 2             --out_dir /home/sagemaker-user/reproducible/final_ex/income/dhsincountry/             --keep_frac 1.0 --seed 123             --experiment_name DHS_Incountry_C_ms_samescaled_RN50_200             --dataset DHS_incountry_C             --ls_band ms             --lr 0.0001 --fc_reg 1.0 --conv_reg 1.0             --imagenet_weights_path /home/sagemaker-user/reproducible/resnet/ImageNet-ResNet50.npz             > /home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_C_ms_samescaled_RN50_200.txt 2>&1 ')
# Start monitoring memory usage in a separate thread
memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

# Start executing the command
execute_command()


# In[ ]:


#DHS incountry MS D (1996-2016)

import psutil
import time
import threading

# Caminho do arquivo de log
log_file_path = '/home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_D_ms_samescaled_RN50_200.txt'

# Monitorar o uso de memória e escrever no arquivo de log
def monitor_memory():
    with open(log_file_path, 'a') as log_file:
        while True:
            mem_info = psutil.virtual_memory()
            log_file.write(f"Uso de memória: {mem_info.percent}%\n")
            log_file.flush()  # Certifique-se de que os dados são gravados no arquivo
            time.sleep(1)

# Executar o comando em segundo plano
def execute_command():
    get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python train_direct.py             --label_name income             --hs_weight_init samescaled            --model_name resnet --num_layers 50             --lr_decay 0.96 --batch_size 64             --gpu 0 --num_threads 5             --cache train train_eval val             --augment --eval_every 1 --print_every 40             --max_epochs 2             --out_dir /home/sagemaker-user/reproducible/final_ex/income/dhsincountry/             --keep_frac 1.0 --seed 123             --experiment_name DHS_Incountry_D_ms_samescaled_RN50_200             --dataset DHS_incountry_D             --ls_band ms             --lr 0.0001 --fc_reg 0.001 --conv_reg 1.0             --imagenet_weights_path /home/sagemaker-user/reproducible/resnet/ImageNet-ResNet50.npz             > /home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_D_ms_samescaled_RN50_200.txt 2>&1 ')
# Start monitoring memory usage in a separate thread
memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

# Start executing the command
execute_command()


# In[ ]:


#DHS incountry MS E (1996-2019)

import psutil
import time
import threading

# Caminho do arquivo de log
log_file_path = '/home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_E_ms_samescaled_RN50_200.txt'

# Monitorar o uso de memória e escrever no arquivo de log
def monitor_memory():
    with open(log_file_path, 'a') as log_file:
        while True:
            mem_info = psutil.virtual_memory()
            log_file.write(f"Uso de memória: {mem_info.percent}%\n")
            log_file.flush()  # Certifique-se de que os dados são gravados no arquivo
            time.sleep(1)

# Executar o comando em segundo plano
def execute_command():
    get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python train_direct.py             --label_name income             --hs_weight_init samescaled            --model_name resnet --num_layers 50             --lr_decay 0.96 --batch_size 64             --gpu 0 --num_threads 5             --cache train train_eval val             --augment --eval_every 1 --print_every 40             --max_epochs 2             --out_dir /home/sagemaker-user/reproducible/final_ex/income/dhsincountry/             --keep_frac 1.0 --seed 123             --experiment_name DHS_Incountry_E_ms_samescaled_RN50_200             --dataset DHS_incountry_E             --ls_band ms             --lr 0.0001 --fc_reg 0.001 --conv_reg 0.001             --imagenet_weights_path /home/sagemaker-user/reproducible/resnet/ImageNet-ResNet50.npz             > /home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_E_ms_samescaled_RN50_200.txt 2>&1 ')
# Start monitoring memory usage in a separate thread
memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

# Start executing the command
execute_command()


# # NightLights Training 150

# In[ ]:


#DHS_Incountry_A_nl
import psutil
import time
import threading

# Caminho do arquivo de log
log_file_path = '/home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_A_nl_random.txt'

# Monitorar o uso de memória e escrever no arquivo de log
def monitor_memory():
    with open(log_file_path, 'a') as log_file:
        while True:
            mem_info = psutil.virtual_memory()
            log_file.write(f"Uso de memória: {mem_info.percent}%\n")
            log_file.flush()  # Certifique-se de que os dados são gravados no arquivo
            time.sleep(1)

# Executar o comando em segundo plano
def execute_command():
    get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python train_direct.py             --label_name income             --hs_weight_init random             --model_name resnet --num_layers 18             --lr_decay 0.96 --batch_size 64             --gpu 0 --num_threads 5             --cache train train_eval val             --augment --eval_every 1 --print_every 40             --max_epochs 2             --out_dir /home/sagemaker-user/reproducible/final_ex/dhsincountry/             --keep_frac 1.0 --seed 123             --experiment_name DHS_Incountry_A_nl_random             --dataset DHS_incountry_A             --nl_band split             --lr 0.0001 --fc_reg 1.0 --conv_reg 1.0             > /home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_A_nl_random.txt 2>&1 ')
# Start monitoring memory usage in a separate thread
memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

# Start executing the command
execute_command()


# In[ ]:


#DHS_Incountry_B_nl
import psutil
import time
import threading

# Caminho do arquivo de log
log_file_path = '/home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_B_nl_random.txt'

# Monitorar o uso de memória e escrever no arquivo de log
def monitor_memory():
    with open(log_file_path, 'a') as log_file:
        while True:
            mem_info = psutil.virtual_memory()
            log_file.write(f"Uso de memória: {mem_info.percent}%\n")
            log_file.flush()  # Certifique-se de que os dados são gravados no arquivo
            time.sleep(1)

# Executar o comando em segundo plano
def execute_command():
    get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python train_direct.py             --label_name income             --hs_weight_init random             --model_name resnet --num_layers 18             --lr_decay 0.96 --batch_size 64             --gpu 0 --num_threads 5             --cache train train_eval val             --augment --eval_every 1 --print_every 40             --max_epochs 2             --out_dir /home/sagemaker-user/reproducible/final_ex/dhsincountry/             --keep_frac 1.0 --seed 123             --experiment_name DHS_Incountry_B_nl_random             --dataset DHS_incountry_B             --nl_band split             --lr 0.0001 --fc_reg 1.0 --conv_reg 1.0             > /home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_B_nl_random.txt 2>&1 ')
# Start monitoring memory usage in a separate thread
memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

# Start executing the command
execute_command()


# In[ ]:


#DHS_Incountry_C_nl
import psutil
import time
import threading

# Caminho do arquivo de log
log_file_path = '/home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_C_nl_random.txt'

# Monitorar o uso de memória e escrever no arquivo de log
def monitor_memory():
    with open(log_file_path, 'a') as log_file:
        while True:
            mem_info = psutil.virtual_memory()
            log_file.write(f"Uso de memória: {mem_info.percent}%\n")
            log_file.flush()  # Certifique-se de que os dados são gravados no arquivo
            time.sleep(1)

# Executar o comando em segundo plano
def execute_command():
    get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python train_direct.py             --label_name income             --hs_weight_init random             --model_name resnet --num_layers 18             --lr_decay 0.96 --batch_size 64             --gpu 0 --num_threads 5             --cache train train_eval val             --augment --eval_every 1 --print_every 40             --max_epochs 2             --out_dir /home/sagemaker-user/reproducible/final_ex/dhsincountry/             --keep_frac 1.0 --seed 123             --experiment_name DHS_Incountry_C_nl_random             --dataset DHS_incountry_C             --nl_band split             --lr 0.0001 --fc_reg 1.0 --conv_reg 1.0             > /home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_C_nl_random.txt 2>&1 ')
# Start monitoring memory usage in a separate thread
memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

# Start executing the command
execute_command()


# In[ ]:


#DHS_Incountry_D_nl
import psutil
import time
import threading

# Caminho do arquivo de log
log_file_path = '/home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_D_nl_random.txt'

# Monitorar o uso de memória e escrever no arquivo de log
def monitor_memory():
    with open(log_file_path, 'a') as log_file:
        while True:
            mem_info = psutil.virtual_memory()
            log_file.write(f"Uso de memória: {mem_info.percent}%\n")
            log_file.flush()  # Certifique-se de que os dados são gravados no arquivo
            time.sleep(1)

# Executar o comando em segundo plano
def execute_command():
    get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python train_direct.py             --label_name income             --hs_weight_init random             --model_name resnet --num_layers 18             --lr_decay 0.96 --batch_size 64             --gpu 0 --num_threads 5             --cache train train_eval val             --augment --eval_every 1 --print_every 40             --max_epochs 2             --out_dir /home/sagemaker-user/reproducible/final_ex/dhsincountry/             --keep_frac 1.0 --seed 123             --experiment_name DHS_Incountry_D_nl_random             --dataset DHS_incountry_D             --nl_band split             --lr 0.0001 --fc_reg 1.0 --conv_reg 1.0             > /home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_D_nl_random.txt 2>&1 ')
# Start monitoring memory usage in a separate thread
memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

# Start executing the command
execute_command()


# In[ ]:


#DHS_Incountry_E_nl
import psutil
import time
import threading

# Caminho do arquivo de log
log_file_path = '/home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_E_nl_random.txt'

# Monitorar o uso de memória e escrever no arquivo de log
def monitor_memory():
    with open(log_file_path, 'a') as log_file:
        while True:
            mem_info = psutil.virtual_memory()
            log_file.write(f"Uso de memória: {mem_info.percent}%\n")
            log_file.flush()  # Certifique-se de que os dados são gravados no arquivo
            time.sleep(1)

# Executar o comando em segundo plano
def execute_command():
    get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python train_direct.py             --label_name income             --hs_weight_init random             --model_name resnet --num_layers 18             --lr_decay 0.96 --batch_size 64             --gpu 0 --num_threads 5             --cache train train_eval val             --augment --eval_every 1 --print_every 40             --max_epochs 2             --out_dir /home/sagemaker-user/reproducible/final_ex/dhsincountry/             --keep_frac 1.0 --seed 123             --experiment_name DHS_Incountry_E_nl_random             --dataset DHS_incountry_E             --nl_band split             --lr 0.001 --fc_reg 0.1 --conv_reg 1.0             > /home/sagemaker-user/reproducible/logs/training/train_log_DHS_Incountry_E_nl_random.txt 2>&1 ')
# Start monitoring memory usage in a separate thread
memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

# Start executing the command
execute_command()


# In[ ]:


# !/home/sagemaker-user/reproducible/nexus_experiment/bin/python extract_features_nl.py \
#         > /home/sagemaker-user/reproducible/logs/training/extract_features_nl.txt 2>&1 


# In[ ]:


#DHS_Incountry_E_nl
import psutil
import time
import threading

# Caminho do arquivo de log
log_file_path = '/home/sagemaker-user/reproducible/logs/training/extract_features_ms.txt'

# Monitorar o uso de memória e escrever no arquivo de log
def monitor_memory():
    with open(log_file_path, 'a') as log_file:
        while True:
            mem_info = psutil.virtual_memory()
            log_file.write(f"Uso de memória: {mem_info.percent}%\n")
            log_file.flush()  # Certifique-se de que os dados são gravados no arquivo
            time.sleep(60)

# Executar o comando em segundo plano
def execute_command():
    get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python extract_features_ms.py         > /home/sagemaker-user/reproducible/logs/training/extract_features_ms.txt 2>&1 ')
# Start monitoring memory usage in a separate thread
memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()

# Start executing the command
execute_command()


# In[ ]:


get_ipython().system('/home/sagemaker-user/reproducible/nexus_experiment/bin/python extract_features_ms.py         > /home/sagemaker-user/reproducible/logs/training/extract_features_ms.txt 2>&1 ')


# In[ ]:


# !/home/sagemaker-user/reproducible/nexus_experiment/bin/python extract_features_ms.py


# In[ ]:


home(/sagemaker-user/reproducible/final_ex/income/dhsincountry/DHS_Incountry_C_ms_samescaled_200_b64_fc1.0_conv1.0_lr0001)


# In[ ]:


1


# In[ ]:




