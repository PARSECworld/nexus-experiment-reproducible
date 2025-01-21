# Use a Python base image (feel free to choose 3.7, 3.8, etc. as needed)
FROM python:3.9-slim

###############################################################################
# Environment variables
###############################################################################
# Airflow
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW_VERSION=2.5.1
ENV AIRFLOW_ADMIN_USER=admin
ENV AIRFLOW_ADMIN_PWD=admin
ENV AIRFLOW_ADMIN_EMAIL=admin@example.com

# MLflow
ENV MLFLOW_VERSION=2.3.2
ENV MLFLOW_HOST=0.0.0.0
ENV MLFLOW_PORT=5000
ENV MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
ENV MLFLOW_ARTIFACT_ROOT=/opt/airflow/mlruns

###############################################################################
# Install dependencies
###############################################################################
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    supervisor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Airflow and MLflow
RUN pip install --no-cache-dir \
    apache-airflow==${AIRFLOW_VERSION} \
    mlflow==${MLFLOW_VERSION}

# Create airflow user and directory
RUN useradd -ms /bin/bash airflow \
    && mkdir -p $AIRFLOW_HOME \
    && chown -R airflow:airflow $AIRFLOW_HOME

USER airflow
WORKDIR $AIRFLOW_HOME

###############################################################################
# Airflow initialization
###############################################################################
# Use SQLite for demonstration/development only
RUN airflow db init

# Create airflow admin user with environment variables
RUN airflow users create \
    --username "${AIRFLOW_ADMIN_USER}" \
    --firstname "Admin" \
    --lastname "User" \
    --role "Admin" \
    --email "${AIRFLOW_ADMIN_EMAIL}" \
    --password "${AIRFLOW_ADMIN_PWD}"

###############################################################################
# Supervisor Configuration (Airflow + MLflow)
###############################################################################
USER root

# Create a folder for supervisor configs
RUN mkdir -p /etc/supervisor/conf.d

# Create supervisor configuration inline
RUN echo "[supervisord]\nnodaemon=true\n\n" \
    "[program:airflow-webserver]\n" \
    "command=airflow webserver --port 8080\n" \
    "directory=/opt/airflow\n" \
    "user=airflow\n" \
    "autostart=true\n" \
    "autorestart=true\n" \
    "\n" \
    "[program:airflow-scheduler]\n" \
    "command=airflow scheduler\n" \
    "directory=/opt/airflow\n" \
    "user=airflow\n" \
    "autostart=true\n" \
    "autorestart=true\n" \
    "\n" \
    "[program:mlflow-server]\n" \
    "command=mlflow server --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} --default-artifact-root ${MLFLOW_ARTIFACT_ROOT} --host ${MLFLOW_HOST} --port ${MLFLOW_PORT}\n" \
    "directory=/opt/airflow\n" \
    "user=airflow\n" \
    "autostart=true\n" \
    "autorestart=true\n" \
    > /etc/supervisor/conf.d/supervisord.conf

###############################################################################
# Expose ports: Airflow (8080) + MLflow (${MLFLOW_PORT})
###############################################################################
# Airflow web server on 8080
EXPOSE 8080
# MLflow default port
EXPOSE ${MLFLOW_PORT}

###############################################################################
# Default command: start supervisor
###############################################################################
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]