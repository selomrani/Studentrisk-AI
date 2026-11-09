FROM apache/airflow:2.10.5-python3.11

# Install project‑specific Python dependencies.
# ------------------------------------------------------------------
# The base Airflow image already contains a functional entry‑point and
# the required constraints file, so we install our extra packages with
# plain pip. This avoids the broken `airflow` script that `uv` writes.
# ------------------------------------------------------------------
USER airflow

COPY requirements.txt /tmp/requirements.txt

# Install the Python requirements as the airflow user (per official docs).
RUN pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.11.txt" \
    -r /tmp/requirements.txt

# Return to the airflow user for the runtime (already airflow).
USER airflow
