#!/bin/bash

# Extrai o nome base do script (sem extensão)
script_name=$(basename "$0")
script_base_name="${script_name%.*}"
log_dir="/home/sagemaker-user/reproducible/logs"
process_log_file="${log_dir}/process_log_${script_base_name}.txt"

# Cria o diretório de logs, se não existir
mkdir -p "$log_dir"

# Remove o arquivo de log anterior, se existir
rm -f "$process_log_file"

# Função para processar um comando e registrar erros
execute_and_log() {
  local cmd="$1"
  echo "Executando: $cmd" >> "$process_log_file"
  eval "$cmd" >> "$process_log_file" 2>&1
  if [ $? -ne 0 ]; then
    echo "Erro ao executar: $cmd" >> "$process_log_file"
  fi
}

# Script para adicionar arquivos ao DVC e empurrá-los para o S3
{
  echo "Início: $(date)"

  . nexus_experiment/bin/activate

  # Verificar e configurar credenciais Git se necessário
  if [ -z "$(git config --global user.email)" ]; then
    git config --global user.email "leoczzi@yahoo.com"
  fi

  if [ -z "$(git config --global user.name)" ]; then
    git config --global user.name "leoczzi"
  fi

  # Configurar o remote do DVC
  echo "Setting 'myremote' as a default remote."

  # Remover arquivos de lock corrompidos
  rm -f .dvc/tmp/rwlock .dvc/tmp/lock

  # Adicionar arquivos ao DVC
  execute_and_log "time dvc add final_ex/income/dhsincountry/*.0_lr0001"
  execute_and_log "time dvc add final_ex/literacy/dhsincountry/*.0_lr0001"
  execute_and_log "time dvc add logs/income/*.csv"
  execute_and_log "time dvc add logs/literacy/*.csv"
  execute_and_log "time dvc add data/interim/*.csv"
  execute_and_log "time dvc add data/processed/*.csv"
  execute_and_log "time dvc add data/raw/nexus_tfrecords_raw/*.gz"
  execute_and_log "time dvc add data/raw/NexusIndicators/*.csv"
  execute_and_log "time dvc add data/raw/setores_shapefile/ac_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/al_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/am_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/ap_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/ba_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/ce_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/df_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/es_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/go_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/ma_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/mg_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/ms_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/mt_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/pa_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/pb_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/pe_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/pi_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/pr_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/rj_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/rn_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/ro_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/rr_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/rs_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/sc_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/se_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/sp_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/raw/setores_shapefile/to_setores_censitarios/*.{dbf,prj,shp,shx}"
  execute_and_log "time dvc add data/processed/nexus_tfrecords_processed/brazil_2010/*.gz"
    
  # Executando o script Python para gerar metadados  
  execute_and_log "time python generate_metadata.py"
  
  # Empurrar para armazenamento remoto
  execute_and_log "time dvc push"

  # Adicionar arquivos ao Git e fazer commit
  execute_and_log "git add ."
  execute_and_log "git commit -m 'Adding and pushing files to DVC and Git with metadata'"
  execute_and_log "git push"

  echo "Término: $(date)"
} > "$process_log_file" 2>&1

echo "Fim: $(date)" >> "$process_log_file"