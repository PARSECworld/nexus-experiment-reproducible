#!/bin/bash

# Remove previous log files
rm -f nohup.out output.log

# Remove DVC lock file if exists
rm -f .dvc/tmp/rwlock
rm -f .dvc/tmp/lock


# Registrar hora de início
echo "Início: $(date)" >> output.log

# Configurar o remote do DVC
time dvc remote add -d myremote s3://mestrado-leandro/nexus_reproducible_dvc/ -f >> output.log 2>&1

# Adicionar arquivos ao DVC
time dvc add final_ex/income/dhsincountry/*.0_lr0001 >> output.log 2>&1
time dvc add final_ex/literacy/dhsincountry/*.0_lr0001 >> output.log 2>&1
time dvc add logs/income/*.csv >> output.log 2>&1
time dvc add logs/literacy/*.csv >> output.log 2>&1
time dvc add data/interim/*.csv >> output.log 2>&1
time dvc add data/processed/*.csv >> output.log 2>&1
time dvc add data/raw/nexus_tfrecords_raw/*.gz >> output.log 2>&1
time dvc add data/raw/NexusIndicators/*.csv >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ac_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ac_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ac_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ac_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/al_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/al_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/al_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/al_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/am_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/am_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/am_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/am_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/ap_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ap_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ap_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ap_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/ba_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ba_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ba_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ba_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/ce_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ce_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ce_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ce_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/df_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/df_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/df_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/df_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/es_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/es_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/es_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/es_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/go_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/go_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/go_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/go_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/ma_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ma_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ma_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ma_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/mg_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/mg_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/mg_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/mg_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/ms_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ms_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ms_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ms_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/mt_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/mt_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/mt_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/mt_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/pa_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/pa_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/pa_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/pa_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/pb_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/pb_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/pb_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/pb_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/pe_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/pe_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/pe_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/pe_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/pi_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/pi_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/pi_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/pi_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/pr_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/pr_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/pr_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/pr_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/rj_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/rj_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/rj_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/rj_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/rn_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/rn_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/rn_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/rn_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/ro_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ro_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ro_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/ro_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/rr_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/rr_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/rr_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/rr_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/rs_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/rs_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/rs_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/rs_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/sc_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/sc_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/sc_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/sc_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/se_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/se_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/se_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/se_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/sp_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/sp_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/sp_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/sp_setores_censitarios/*.shx >> output.log 2>&1

time dvc add data/raw/setores_shapefile/to_setores_censitarios/*.dbf >> output.log 2>&1
time dvc add data/raw/setores_shapefile/to_setores_censitarios/*.prj >> output.log 2>&1
time dvc add data/raw/setores_shapefile/to_setores_censitarios/*.shp >> output.log 2>&1
time dvc add data/raw/setores_shapefile/to_setores_censitarios/*.shx >> output.log 2>&1
time dvc add data/processed/nexus_tfrecords_processed/brazil_2010/*.gz >> output.log 2>&1

# Adicionar arquivos ao Git
#time git add final_ex/income/*.0_lr0001.dvc final_ex/literacy/*.0_lr0001.dvc logs/income/*.csv.dvc logs/literacy/*.csv.dvc data/interim/*.csv.dvc data/processed/*.csv.dvc data/raw/nexus_tfrecords_raw/*.gz.dvc data/raw/NexusIndicators/*.csv.dvc data/raw/setores_shapefile/*.dbf.dvc data/raw/setores_shapefile/*.prj.dvc data/raw/setores_shapefile/*.shp.dvc data/raw/setores_shapefile/*.shx.dvc data/processed/nexus_tfrecords_processed/brazil_2010/*.gz.dvc .gitignore >> output.log 2>&1

time git commit -m "Add DVC tracked data files" >> output.log 2>&1

# Enviar arquivos ao S3
time dvc push >> output.log 2>&1

# Enviar alterações ao GitHub
time git push origin develop >> output.log 2>&1

# Registrar hora de término
echo "Término: $(date)" >> output.log