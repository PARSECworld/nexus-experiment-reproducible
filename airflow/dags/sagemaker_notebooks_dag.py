
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
from execute_notebook import ExecuteNotebookOperator

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'retries': 1,
}

dag = DAG(
    'sagemaker_notebooks_dag',
    default_args=default_args,
    description='DAG to execute SageMaker Studio notebooks',
    schedule_interval=None,
)

start = DummyOperator(
    task_id='start',
    dag=dag,
)

notebook_1 = ExecuteNotebookOperator(
    task_id='execute_notebook_1',
    notebook_path='/home/sagemaker-user/SageMaker/01_download_images.ipynb',
    output_path='/home/sagemaker-user/SageMaker/output/01_download_images_output.ipynb',
    dag=dag,
)

notebook_2 = ExecuteNotebookOperator(
    task_id='execute_notebook_2',
    notebook_path='/home/sagemaker-user/SageMaker/02_process_tfrecords.ipynb',
    output_path='/home/sagemaker-user/SageMaker/output/02_process_tfrecords_output.ipynb',
    dag=dag,
)

# Add other notebooks here in the same way

end = DummyOperator(
    task_id='end',
    dag=dag,
)

start >> notebook_1 >> notebook_2 >> end
