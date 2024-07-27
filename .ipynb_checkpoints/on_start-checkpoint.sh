#!/bin/bash
apt-get update
apt-get install -y graphviz libgraphviz-dev pkg-config libffi-dev
. /home/sagemaker-user/reproducible/nexus_experiment/bin/activate
python -m ipykernel install --user --name nexus_experiment --display-name "Nexus Experiment"
apt install git
# Verificar e configurar credenciais Git se necessário
git remote set-url origin https://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@github.com/PARSECworld/nexus-reproducible.git

#git remote set-url origin https://leoczzi:dfxJPiLkqOpg3IFv3MJOWSK44Uik9m1GikLL@github.com/leoczzi/nexus-reproducible.git

# Se não estiverem configurados, configurar o user e email
git config --global user.name "leoczzi"
git config --global user.email "leoczzi@yahoo.com"

git push --set-upstream origin develop

deactivate