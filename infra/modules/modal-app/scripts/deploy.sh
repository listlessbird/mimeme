#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${MODAL_TOKEN_ID:-}" ]]; then
  echo "Error: MODAL_TOKEN_ID is required"
  exit 1
fi

if [[ -z "${MODAL_TOKEN_SECRET:-}" ]]; then
  echo "Error: MODAL_TOKEN_SECRET is required"
  exit 1
fi

if [[ -z "${DEPLOY_PATH:-}" || -z "${DEPLOY_TARGET:-}" ]]; then
  echo "Error: DEPLOY_PATH and DEPLOY_TARGET are required"
  exit 1
fi

echo "Deploying Modal app: ${APP_NAME}"
cd "${DEPLOY_PATH}"
uv run modal deploy "${DEPLOY_TARGET}"
echo "Modal app ${APP_NAME} deployed successfully"
