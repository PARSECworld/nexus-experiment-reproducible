import mlflow

# Configuração da URI do backend com MySQL no RDS
#DB_URI = "mysql+mysqlconnector://admin:Tlp043b1*@mlflow-test-db.clrg6ngtj8ke.us-east-1.rds.amazonaws.com:3306/mlflow_db"
DB_URI = "mysql+pymysql://admin:Tlp043b1*@mlflow-test-db.clrg6ngtj8ke.us-east-1.rds.amazonaws.com:3306/mlflow_db"

# Definir o tracking server para o MLflow
mlflow.set_tracking_uri(DB_URI)

# Definir o experimento
mlflow.set_experiment("nexus_experiment")

# Iniciar uma execução de teste
with mlflow.start_run() as run:
    mlflow.log_param("parametro1", 10)
    mlflow.log_metric("accuracy", 0.95)
    # Armazenar artefatos no S3
    mlflow.log_artifacts("artifacts", artifact_path="s3://mestrado-leandro/mlflow/")
