#!/bin/bash
apt-get update
apt-get install -y graphviz libgraphviz-dev pkg-config libffi-dev
. /home/sagemaker-user/reproducible/nexus_experiment/bin/activate
python -m ipykernel install --user --name nexus_experiment --display-name "Nexus Experiment"
apt install git
deactivate