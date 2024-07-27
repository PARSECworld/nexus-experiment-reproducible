import os
import json
import tensorflow as tf
import yaml
import hashlib
import fiona
import pandas as pd
from datetime import datetime

def count_tfrecords(file_path):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    return sum(1 for _ in raw_dataset)

def get_file_size(file_path):
    return os.path.getsize(file_path)

def get_creation_date(file_path):
    return datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()

def calculate_checksum(file_path, algorithm='md5'):
    hash_alg = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hash_alg.update(chunk)
    return hash_alg.hexdigest()

def extract_tfrecord_schema(file_path):
    schema = {}
    raw_dataset = tf.data.TFRecordDataset(file_path)
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        for feature_name, feature in example.features.feature.items():
            kind = feature.WhichOneof('kind')
            schema[feature_name] = kind
    return schema

def count_shapefile_records(file_path):
    with fiona.open(file_path, 'r') as shapefile:
        return len(shapefile)

def extract_shapefile_bbox(file_path):
    with fiona.open(file_path, 'r') as shapefile:
        return shapefile.bounds

def extract_shapefile_crs(file_path):
    with fiona.open(file_path, 'r') as shapefile:
        return shapefile.crs

def process_csv(file_path):
    df = pd.read_csv(file_path)
    num_records = len(df)
    num_columns = len(df.columns)
    column_names = list(df.columns)
    file_size = get_file_size(file_path)
    creation_date = get_creation_date(file_path)
    checksum = calculate_checksum(file_path)

    return {
        'description': f'CSV file {os.path.basename(file_path)}',
        'format': 'csv',
        'number_of_records': num_records,
        'number_of_columns': num_columns,
        'column_names': column_names,
        'file_size': file_size,
        'creation_date': creation_date,
        'checksum': checksum
    }

def update_dvc_file(dvc_file, metadata):
    with open(dvc_file, 'r') as file:
        dvc_content = yaml.safe_load(file)
    
    dvc_content['meta'] = metadata

    with open(dvc_file, 'w') as file:
        yaml.dump(dvc_content, file)

def process_files(directory, file_extension, process_function):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                dvc_file = os.path.join(root, f"{os.path.basename(file)}.dvc")
                
                if os.path.exists(dvc_file):
                    metadata = process_function(file_path)
                    update_dvc_file(dvc_file, metadata)
                else:
                    print(f"Warning: .dvc file for {file_path} does not exist")

if __name__ == "__main__":
    tfrecords_directory = 'data/raw/nexus_tfrecords_raw'
    shapefiles_directory = 'data/raw/setores_shapefile'
    csv_directories = [
        'final_ex/income',
        'final_ex/literacy',
        'logs/income',
        'logs/literacy',
        'data/interim',
        'data/processed',
        'data/raw/NexusIndicators'
    ]
    
    process_files(tfrecords_directory, ".tfrecords.gz", lambda file_path: {
        'description': f'TFRecord file {os.path.basename(file_path)}',
        'format': 'tfrecords.gz',
        'number_of_records': count_tfrecords(file_path),
        'file_size': get_file_size(file_path),
        'creation_date': get_creation_date(file_path),
        'checksum': calculate_checksum(file_path),
        'schema': extract_tfrecord_schema(file_path),
        'number_of_features': len(extract_tfrecord_schema(file_path))
    })
    
    process_files(shapefiles_directory, ".shp", lambda file_path: {
        'description': f'Shapefile {os.path.basename(file_path)}',
        'format': 'shp',
        'number_of_records': count_shapefile_records(file_path),
        'file_size': get_file_size(file_path),
        'creation_date': get_creation_date(file_path),
        'checksum': calculate_checksum(file_path),
        'bounding_box': extract_shapefile_bbox(file_path),
        'crs': extract_shapefile_crs(file_path)
    })
    
    for directory in csv_directories:
        process_files(directory, ".csv", process_csv)