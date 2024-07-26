#!/bin/bash

# Remover o arquivo de log anterior
rm -f output.log nohup.out

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
  dvc remote add -d myremote s3://mestrado-leandro/nexus_reproducible_dvc/ -F
  echo "Setting 'myremote' as a default remote."

  # Remover arquivos de lock corrompidos
  rm -f .dvc/tmp/rwlock .dvc/tmp/lock

  # Adicionar arquivos ao DVC
  time dvc add final_ex/income/dhsincountry/*.0_lr0001
  time dvc add final_ex/literacy/dhsincountry/*.0_lr0001
  time dvc add logs/income/*.csv
  time dvc add logs/literacy/*.csv
  time dvc add data/interim/*.csv
  time dvc add data/processed/*.csv
  time dvc add data/raw/nexus_tfrecords_raw/*.gz
  time dvc add data/raw/NexusIndicators/*.csv
  time dvc add data/raw/setores_shapefile/ac_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/al_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/am_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/ap_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/ba_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/ce_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/df_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/es_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/go_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/ma_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/mg_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/ms_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/mt_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/pa_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/pb_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/pe_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/pi_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/pr_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/rj_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/rn_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/ro_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/rr_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/rs_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/sc_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/se_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/sp_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/raw/setores_shapefile/to_setores_censitarios/*.{dbf,prj,shp,shx}
  time dvc add data/processed/nexus_tfrecords_processed/brazil_2010/*.gz

  # Empurrar para armazenamento remoto
  time dvc push

  # Adicionar arquivos ao Git e fazer commit
  git add .
  git commit -m "Adicionando e empurrando arquivos para DVC e Git"
  git push

  echo "Término: $(date)"
} > output.log 2>&1

echo "Fim: $(date)" >> output.log