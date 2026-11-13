## Streamlit champion inference

Start the MLflow service, then run the app from the repository root:

```bash
docker compose up -d mlflow
MLFLOW_TRACKING_URI=http://localhost:5000 uv run streamlit run streamlit/app.py
```

Do not use `uv run python streamlit/app.py`; that executes the script outside
Streamlit's server runtime and produces the `ScriptRunContext` warnings.

The app loads `models:/StudentRisk@champion` from MLflow. Override
`MLFLOW_TRACKING_URI`, `MLFLOW_MODEL_NAME`, or `MLFLOW_MODEL_ALIAS` with
environment variables (or matching Streamlit secrets) when deploying.
