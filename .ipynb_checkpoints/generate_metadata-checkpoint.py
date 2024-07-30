import os
import tensorflow as tf
import yaml
import hashlib
import fiona
import pandas as pd
from datetime import datetime

tf.compat.v1.enable_eager_execution()

LOG_DIR = '/home/sagemaker-user/reproducible/logs/'
LOG_FILE = os.path.join(LOG_DIR, 'log_generate_metadata.txt')

def count_tfrecords(file_path):
    try:
        raw_dataset = tf.data.TFRecordDataset(file_path)
        return sum(1 for _ in raw_dataset)
    except Exception as e:
        log_error(file_path, f"Error counting records: {e}")
        return None

def get_file_size(file_path):
    return os.path.getsize(file_path)

def get_creation_date(file_path):
    return datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()

def get_modification_date(file_path):
    return datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()

def calculate_checksum(file_path, algorithm='md5'):
    hash_alg = hashlib.new(algorithm)
    try:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hash_alg.update(chunk)
    except Exception as e:
        log_error(file_path, f"Error calculating checksum: {e}")
        return None
    return hash_alg.hexdigest()

def extract_tfrecord_schema(file_path):
    schema = {}
    try:
        raw_dataset = tf.data.TFRecordDataset(file_path)
        for raw_record in raw_dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            for feature_name, feature in example.features.feature.items():
                kind = feature.WhichOneof('kind')
                schema[feature_name] = kind
    except Exception as e:
        log_error(file_path, f"Error extracting schema: {e}")
        return None
    return schema

def count_shapefile_records(file_path):
    try:
        with fiona.open(file_path, 'r') as shapefile:
            return len(shapefile)
    except Exception as e:
        log_error(file_path, f"Error counting records: {e}")
        return 0

def extract_shapefile_bbox(file_path):
    try:
        with fiona.open(file_path, 'r') as shapefile:
            return shapefile.bounds
    except Exception as e:
        log_error(file_path, f"Error extracting bounding box: {e}")
        return None

def extract_shapefile_crs(file_path):
    try:
        with fiona.open(file_path, 'r') as shapefile:
            return shapefile.crs
    except Exception as e:
        log_error(file_path, f"Error extracting CRS: {e}")
        return None

def process_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        num_records = len(df)
        num_columns = len(df.columns)
        column_names = list(df.columns)
        file_size = get_file_size(file_path)
        creation_date = get_creation_date(file_path)
        modification_date = get_modification_date(file_path)
        checksum = calculate_checksum(file_path)

        return {
            'description': f'CSV file {os.path.basename(file_path)}',
            'format': 'csv',
            'number_of_records': num_records,
            'number_of_columns': num_columns,
            'column_names': column_names,
            'file_size': file_size,
            'creation_date': creation_date,
            'modification_date': modification_date,
            'checksum': checksum
        }
    except pd.errors.EmptyDataError:
        log_error(file_path, "CSV file is empty")
        return None
    except Exception as e:
        log_error(file_path, f"Error processing CSV file: {e}")
        return None

def update_dvc_file(dvc_file, metadata):
    def tuple_constructor(loader, node):
        return tuple(loader.construct_sequence(node))

    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

    try:
        with open(dvc_file, 'r') as file:
            dvc_content = yaml.safe_load(file)

        dvc_content['meta'] = metadata

        with open(dvc_file, 'w') as file:
            yaml.safe_dump(dvc_content, file)
    except yaml.YAMLError as exc:
        log_error(dvc_file, f"Error reading or updating .dvc file: {exc}")

def should_ignore(file_path):
    return 'checkpoint' in file_path.lower()

def log_error(file_path, message):
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"ERROR: {file_path} - {message}\n")

def log_success(file_path):
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"SUCCESS: {file_path}\n")

def process_files(directory, file_extension, process_function):
    files_processed = False
    success_count = 0
    error_count = 0
    missing_dvc_count = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension) and not should_ignore(file):
                file_path = os.path.join(root, file)
                dvc_file = file_path + '.dvc'

                if os.path.exists(dvc_file):
                    print(f"Processing file: {file_path}")
                    metadata = process_function(file_path)
                    if metadata:
                        update_dvc_file(dvc_file, metadata)
                        print(f"Updating existing metadata in {dvc_file}")
                        log_success(file_path)
                        success_count += 1
                    else:
                        log_error(file_path, "Failed to process file")
                        error_count += 1
                    files_processed = True
                else:
                    log_error(file_path, ".dvc file does not exist")
                    error_count += 1
                    missing_dvc_count += 1

    if not files_processed:
        print(f"No files with extension {file_extension} found in {directory}")

    return success_count, error_count, missing_dvc_count

if __name__ == "__main__":
    # Clear or create the log file at the start
    with open(LOG_FILE, 'w') as log_file:
        log_file.write("Processing Log\n")
        log_file.write("="*30 + "\n\n")

    tfrecords_directories = [
        'data/raw/nexus_tfrecords_raw',
        'data/processed/nexus_tfrecords_processed/brazil_2010'
    ]
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

    total_success = 0
    total_errors = 0
    total_missing_dvc = 0
    summary = {}

    for directory in tfrecords_directories:
        print(f"Starting TFRecord files processing in directory {directory}...")
        success, errors, missing_dvc = process_files(directory, ".gz", lambda file_path: {
            'description': f'TFRecord file {os.path.basename(file_path)}',
            'format': 'tfrecords.gz',
            'number_of_records': count_tfrecords(file_path) or 0,
            'file_size': get_file_size(file_path),
            'creation_date': get_creation_date(file_path),
            'modification_date': get_modification_date(file_path),
            'checksum': calculate_checksum(file_path),
            'schema': extract_tfrecord_schema(file_path) or {},
            'number_of_features': len(extract_tfrecord_schema(file_path) or {})
        })
        total_success += success
        total_errors += errors
        total_missing_dvc += missing_dvc
        summary[directory] = {'success': success, 'errors': errors, 'missing_dvc': missing_dvc}

    print("Starting Shapefiles processing...")
    success, errors, missing_dvc = process_files(shapefiles_directory, ".shp", lambda file_path: {
        'description': f'Shapefile {os.path.basename(file_path)}',
        'format': 'shp',
        'number_of_records': count_shapefile_records(file_path),
        'file_size': get_file_size(file_path),
        'creation_date': get_creation_date(file_path),
        'modification_date': get_modification_date(file_path),
        'checksum': calculate_checksum(file_path),
        'bounding_box': extract_shapefile_bbox(file_path),
        'crs': extract_shapefile_crs(file_path)
    })
    total_success += success
    total_errors += errors
    total_missing_dvc += missing_dvc
    summary[shapefiles_directory] = {'success': success, 'errors': errors, 'missing_dvc': missing_dvc}

    for directory in csv_directories:
        print(f"Starting CSV files processing in directory {directory}...")
        success, errors, missing_dvc = process_files(directory, ".csv", process_csv)
        total_success += success
        total_errors += errors
        total_missing_dvc += missing_dvc
        summary[directory] = {'success': success, 'errors': errors, 'missing_dvc': missing_dvc}

    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"\nSUMMARY:\n")
        for directory, counts in summary.items():
            log_file.write(f"{directory}:\n")
            log_file.write(f"  Success: {counts['success']}\n")
            log_file.write(f"  Errors: {counts['errors']}\n")
            log_file.write(f"  Missing .dvc: {counts['missing_dvc']}\n")
        log_file.write(f"\nTotal Success: {total_success}\n")
        log_file.write(f"Total Errors: {total_errors}\n")
        log_file.write(f"Total Missing .dvc: {total_missing_dvc}\n")
