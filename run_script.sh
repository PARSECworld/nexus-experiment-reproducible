#!/bin/bash

# Extrai o nome base do script (sem extensão)
script_name=$(basename "$1")
script_base_name="${script_name%.*}"
extension="${script_name##*.}"

# Define o caminho para o arquivo nohup.out com base no nome do script
log_file="/home/sagemaker-user/reproducible/logs/nohup_${script_base_name}.out"

# Determina o comando a ser usado com base na extensão do arquivo
if [ "$extension" == "py" ]; then
    cmd="python"
elif [ "$extension" == "sh" ]; then
    cmd="bash"
else
    echo "Unsupported script type: .$extension"
    exit 1
fi

# Executa o script com o comando apropriado e redireciona a saída para o arquivo de log
nohup $cmd "$1" > "$log_file" 2>&1 &