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

# Script principal
{
  echo "Início: $(date)"

  . nexus_experiment/bin/activate

  # Configurar credenciais do Git
  if [ -z "$(git config --global user.email)" ]; then
    git config --global user.email "leoczzi@yahoo.com"
  fi

  if [ -z "$(git config --global user.name)" ]; then
    git config --global user.name "leoczzi"
  fi

  # Adicionar regras ao .gitignore
  echo -e "\n# Arquivos grandes ignorados\n*.gz\n*.tfrecord" >> .gitignore
  execute_and_log "git add .gitignore"
  execute_and_log "git commit -m 'Atualizando .gitignore para ignorar arquivos grandes'"

  # Configurar o remote do DVC e remover locks
  echo "Setting 'myremote' as a default remote."
  rm -f .dvc/tmp/rwlock .dvc/tmp/lock

  # Adicionar arquivos ao DVC
  for dir in \
    "final_ex/income/dhsincountry/*.0_lr0001" \
    "final_ex/literacy/dhsincountry/*.0_lr0001" \
    "logs/income/*.csv" \
    "logs/literacy/*.csv" \
    "data/interim/*.csv" \
    "data/processed/*.csv" \
    "data/raw/nexus_tfrecords_raw/*.gz" \
    "data/raw/NexusIndicators/*.csv" \
    "data/raw/setores_shapefile/**/*.dbf" \
    "data/raw/setores_shapefile/**/*.prj" \
    "data/raw/setores_shapefile/**/*.shp" \
    "data/raw/setores_shapefile/**/*.shx" \
    "data/processed/nexus_tfrecords_processed/brazil_2010/*.gz"
  do
    execute_and_log "time dvc add $dir"
  done

  # Executar o script Python para gerar metadados
  execute_and_log "time python generate_metadata.py"

  # Empurrar arquivos gerenciados pelo DVC para o armazenamento remoto
  execute_and_log "time dvc push"

  # Adicionar arquivos DVC ao Git
  execute_and_log "git add ."
  execute_and_log "git commit -m 'Adicionando arquivos DVC e metadados gerados'"
  execute_and_log "git push"

  echo "Término: $(date)"
} > "$process_log_file" 2>&1

echo "Fim: $(date)" >> "$process_log_file"