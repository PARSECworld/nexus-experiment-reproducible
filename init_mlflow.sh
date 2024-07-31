# echo '#!/bin/bash' > mlflow-app.sh
# echo 'mlflow ui --host 0.0.0.0 --port 5000' >> mlflow-app.sh
# chmod +x mlflow-app.sh

mlflow server \
    --backend-store-uri "mysql+mysqlconnector://admin:Tlp043b1*@mlflow-test-db.clrg6ngtj8ke.us-east-1.rds.amazonaws.com:3306/mlflow_db" \
    --default-artifact-root "s3://mestrado-leandro/mlflow/" \
    --host 0.0.0.0 \
    --port 5000