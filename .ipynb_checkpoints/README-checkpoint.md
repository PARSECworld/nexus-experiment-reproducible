# Nexus Experiment - Reproducibility in Deep Learning

This repository contains the implementation of the **Nexus Experiment**, which aims to estimate socio-economic indicators (such as income, literacy, and longevity) from satellite imagery in the Caatinga and Cerrado regions of Brazil. The proposed architecture integrates several tools to ensure **reproducibility** in large-scale data science projects, including:

- **Amazon S3** (data storage)  
- **DVC (Data Version Control)** (versioning of data and models)  
- **Git** (source code version control)  
- **Docker** (consistent runtime environments)  
- **MLflow** (experiment tracking)  
- **Apache Airflow** (pipeline orchestration)  
- **Amazon SageMaker** (optional, for managed execution of experiments)

This experiment is part of the **NEXUS-PARSEC** project and was developed as a practical case study for your master thesis on **computational architectures for reproducible deep learning experiments**.

## Experiment Overview

This experiment includes

1. **Preparing census data** and creating `clusters' (square regions of ~6.72 km² each) to serve as samples for the machine learning model.  
2. **Downloading and processing satellite imagery** (multispectral and night light) using Google Earth Engine (GEE).  
3. **Train deep learning models** (using an adapted ResNet-18) to predict socio-economic indicators (focusing on income).  
4. **Evaluate the models** using spatially independent cross-validation (no image overlap between training and testing).  
5. **Compare different approaches** (multispectral bands only, night light only, or a combination of both) using linear regression (ridge) applied to features extracted from the deep learning models.  
6. **Record and analyze results** in MLflow and Airflow to ensure reproducibility.

---

## Prerequisites

To reproduce the experiment in an environment similar to the one described in the thesis, please ensure that you have (or can configure) the following components:

1. **Git**.  
2. **DVC (Data Version Control)**.  
3. **Docker**
4. **MLflow**
5. **Apache Airflow**
6. **Amazon S3** (or other scalable storage system)  
7. **Google Cloud Account** with access to **Google Earth Engine (GEE)** (to download images)  
8. **Amazon SageMaker** (optional if you want a managed environment)  
9. **Minimum 120 GB** of available space for intermediate data (~6.72 km² per cluster and satellite composites).  
10. A **compute instance** equivalent to `ml.m5.4xlarge` from AWS for the most intensive steps (especially notebook `04_loc_dicts_and_exploratory_analysis.ipynb`), or any instance with similar resources (16 vCPUs, 64 GB RAM).

### Note on pre-trained weights (ResNet-18)

- If the code is configured to use pretrained ImageNet weights (e.g. `ImageNet-ResNet18.npz`), make sure the file is included in the repository or downloaded from an external source.  
- You can also train the network from scratch (which typically requires more computational resources and parameter tuning) by disabling the load pre-trained weights step in the notebook.

---

## Input Data Acquisition

### 1. Census Tract Shapefiles (IBGE)
- Download census tract shapefiles for each state in the study area (Caatinga and Cerrado) from the [IBGE website](https://www.ibge.gov.br/).  
- Save them in `../data/raw/setores_shapefile/*/*.shp`.

### 2. Socio-economic indicators (Nexus indicators)
- There are 19 CSV files (one per state) containing income, literacy, and longevity data for each census tract.  
- Save them in `../data/raw/NexusIndicators/*.csv'.  
- Each CSV must include the census tract code and the socioeconomic indicator values.

### 3. Google Earth Engine (GEE) Configuration
1. **GEE Account**: [Sign up here](https://earthengine.google.com/signup/).  
2. **Service Account and Credentials**:  
   - Create a `client_secret.json` file in the Google Cloud Console with permissions for Earth Engine and Google Drive.  
   - Customize the paths in the `01_download_satellite_images.ipynb` notebook (`SERVICE_ACCOUNT_FILE`, `TOKEN_FILE_NAME`, etc.).  
3. **Drive or GCS**: Check the permissions to export and save the satellite images.

---

## Notebooks and run order

We have renamed the notebooks to better reflect each stage. Run them in the following order:

1. **00_data_preparation_and_clustering.ipynb`  
   - Reads and merges census shapefiles.  
   - Reads and merges socio-economic indicators (CSV).  
   - Generates *clusters* (grid) and calculates associated indicators.  
   - Saves merged data to CSV.

2. `01_download_satellite_images.ipynb`  
   - Uses Google Earth Engine to compile and export Landsat and Night Light imagery.  
   - Exports to Google Drive or GCS in TFRecord format.  
   - (Optional) Download these TFRecords locally.

3. `02_preprocess_tfrecords.ipynb`.  
   - Reads raw TFRecord files.  
   - Validates, groups and splits each TFRecord by cluster.  
   - Computes band statistics (means, standard deviations).

4. `03_create_spatial_folds.ipynb` Creates spatial folds.  
   - Creates folds (A, B, C, D, E) ensuring spatial independence (minimum distance between training/test clusters).  
   - Generates a `.pkl` file describing the assignment of clusters to each fold.

5. `04_loc_dicts_and_exploratory_analysis.ipynb`  
   - Loads `.npz` files containing bands and indicators.  
   - Creates location dictionaries (`loc_dict`) mapping clusters to attributes (country, urban/rural, etc.).  
   - Performs exploratory analysis (income distribution, night lights, etc.).  
   - **Note**: This step was run on an AWS `ml.m5.4xlarge` instance due to the large amount of data.

6. `05_train_deep_learning_income.ipynb`  
   - Trains deep learning models (adapted ResNet-18) for **income**, using multispectral bands (and/or night light, if configured).  
   - Tunes hyperparameters, performs cross-validation, and stores model checkpoints.  
   - (Optional) Uses pre-trained ImageNet weights, if available.

7. `06_train_ridge_combination_income.ipynb`  
   - Loads features extracted from the DL models (MS, NL).  
   - Trains a ridge regression model combining (or not) these features.  
   - Evaluates whether merging bands (MS + NL) improves performance.

8. `07_evaluate_models_income.ipynb`  
   - Evaluates all models (pure DL and Ridge on features).  
   - Compares metrics (R², MSE, correlation) and performs urban/rural analysis.  
   - Generates reports, scatterplots, histograms, etc.

---

## Important Configurations

- **Data paths**:  
  - `DHS_CSV_PATH` or `DATASET_CSV_PATH`: path to the *clusters* CSV.  
  - `BASE_PATH`, `RAW_PATH`, `PROCESSED_PATH`: point to raw and processed data directories.

- GEE export parameters:  
  - `GOOGLE_IMAGES_FOLDER_ID`, `EXPORT_TILE_RADIUS`, `SCALE`: set image resolution and size.

- Minimum distance for folds:  
  - `MIN_DIST` (in degrees) controls the spatial distance between training and test clusters.

---

## Execution and Docker Usage (Airflow + MLflow)

The following is a general workflow for running the notebooks, as well as instructions for spinning up the Airflow + MLflow Docker container with Supervisor.

### 1. Clone the repository and initialize DVC

```bash
git clone https://github.com/your-username/nexus-experiment.git
cd nexus-experiment
dvc init
# If there are .dvc files for datasets, run
dvc pull
```

### 2. (Optional) Build the docker image

Inside the repository, locate the `Dockerfile' that sets up:
- Python 3.9-slim.
- Airflow 2.5.1 (demo using SQLite)
- MLflow 2.3.2.
- Supervisor to manage the processes (Airflow Webserver, Airflow Scheduler, MLflow Server)

To create the image:

```bash
docker build -t nexus-experiment -f dockerfile .
```

### 3. Run the container (Airflow + MLflow)

Run the container, exposing the Airflow (8080) and MLflow (5000) ports:

```bash
docker run -it --rm \
    -p 8080:8080 \
    -p 5000:5000 \
    --name nexus-container \
    nexus-experiment
```

- **Airflow Web UI** at http://localhost:8080  
  - Default credentials (from the docker file):  
    - Username: `admin  
    - Password: `admin
- **MLflow** at http://localhost:5000  
  - Backend is SQLite, artifact root is `/opt/airflow/mlruns` inside the container.

Use `-e` to override environment variables if necessary:

```bash
docker run -it --rm \
    -e AIRFLOW_ADMIN_USER=myuser \
    -e AIRFLOW_ADMIN_PWD=mypassword \
    -p 8080:8080 \
    -p 5000:5000 \
    nexus-experiment
```

### 4. Add your code and data

Several strategies can be used:

- Bind mount a local directory:
  ```bash
  docker run -it --rm \
      -v $(pwd):/app \
      -p 8080:8080 -p 5000:5000 \
      nexus-experiment
  ```
  This will make your local repository accessible in `/app` inside the container.

- Manually **clone the repo inside the container**.  
- Modify the docker file to automatically copy your code.

### 5. Configure GEE and run notebooks

- **GEE credentials**: Place your `client_secret.json` in the container or mount it with `-v` and set the `SERVICE_ACCOUNT_FILE` and `TOKEN_FILE_NAME` variables in the notebooks.  
- **Execute the notebooks**: Launch a `Jupyter notebook' or rely on **Airflow** within the container to orchestrate steps (depending on your workflow).

### 6. Track results

- **Airflow**: Dashboard at http://localhost:8080.  
- **MLflow**: UI at http://localhost:5000 that stores parameters, metrics, and artifacts (models, plots, etc.).

### 7. Internal workflow without Docker

If you prefer not to use Docker, you can:

1. Set up a Python environment with the necessary libraries (Airflow, MLflow, etc.).  
2. Run `airflow db init`, then `airflow webserver & airflow scheduler` manually.  
3. Run the notebooks in the order described.  
4. (Optional) Configure MLflow locally:
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
   ```

---

## Experiment output

In the end, you should have various files in `../data/processed/` (or any other configured path). Some examples:

- Clusters and indicators.  
  * `../data/processed/NEXUS_data_2010.csv`.  
  * `../data/processed/clusters_data_{cluster_image_length}km.csv`.  

- Satellite Images in TFRecord.  
  * `../data/raw/nexus_tfrecords_raw/brazil_2010_XX.tfrecord.gz` (raw exports from GEE)  
  * `../data/processed/nexus_tfrecords_processed/brazil_2010/00XXX.tfrecord.gz` (split by cluster)  

- Folders (Spatial Independence).  
  * `../data/processed/dhs_incountry_co.pkl` (training/validation/testing indices for each fold)  

- Deep Learning Models (Checkpoints).  
  * `../models/checkpoints/<experiment_name>/ckpt-XX.data-00000-of-00001`.  
  * `../models/checkpoints/<experiment_name>/ckpt-XX.index`.  
  *(or other configured directories, e.g. `./final_ex/income/dhs_incountry/`)*.  

- Extracted Features (NPZ).  
  * `../data/processed/features/<model>_features.npz`.  
  *(e.g., `resnet_ms_features.npz`, `resnet_nl_features.npz`, etc.)*.  

- Ridge Regression Results  
  * `../results/ridge_weights.npz` (model coefficients)  
  * `../results/test_preds.npz` (predictions on the test set)  

- Reports, Metrics and Analysis  
  * `../results/plots/` (scatter plots, histograms, etc.)  
  * `../results/performance.csv` (R², MSE, Correlations)  
  * `../results/performance_without_nl.csv` (analysis in areas without night light)  

*(Note: Depending on how you have structured your workflow and path variables, these directories may vary. Adapt to your repository organization.)*.