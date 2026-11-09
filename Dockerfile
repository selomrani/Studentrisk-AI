FROM apache/airflow:2.10.5-python3.11

# uv binary, copied from the official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ARG AIRFLOW_VERSION=2.10.5
ARG PYTHON_VERSION=3.11

# Install into the system interpreter (no venv) so the Airflow entrypoint,
# which uses the system Python, can import these packages. Airflow's own
# constraints file keeps dependency resolution from breaking the base image.
USER root
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache \
        --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
        -r /tmp/requirements.txt
USER airflow
