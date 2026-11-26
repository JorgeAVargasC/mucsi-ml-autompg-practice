#!/usr/bin/env bash
MLFLOW_HOST="${MLFLOW_HOST:-${CODESPACE_NAME}-5000.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}}"
mkdir -p tracking/artifacts

mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --workers 1 \
  --backend-store-uri sqlite:///tracking/mlflow.db \
  --artifacts-destination ./tracking/artifacts \
  --allowed-hosts "localhost:*,127.0.0.1:*,${MLFLOW_HOST},${MLFLOW_HOST}:*" \
  --cors-allowed-origins "http://localhost:*,http://127.0.0.1:*,https://${MLFLOW_HOST}"