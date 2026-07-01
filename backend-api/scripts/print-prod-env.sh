#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
terraform_dir="${repo_root}/terraform/infra"

tf_output() {
  terraform -chdir="${terraform_dir}" output -raw "$1"
}

cat <<EOF
APP_ENV=production
DEBUG=false
LOG_LEVEL=INFO

DB_URL=$(tf_output db_url)

TEMPORAL_HOST=localhost:7233
TEMPORAL_NAMESPACE=default

GPU_BACKEND=modal
MODAL_APP_NAME=$(tf_output modal_app_name)

S3_ENDPOINT_URL=$(tf_output s3_endpoint_url)
S3_REGION=$(tf_output s3_region)
S3_ACCESS_KEY_ID=$(tf_output s3_access_key_id)
S3_SECRET_ACCESS_KEY=$(tf_output s3_secret_access_key)
S3_BUCKET=$(tf_output s3_bucket_name)
S3_FORCE_PATH_STYLE=$(tf_output s3_force_path_style)
S3_PRESIGNED_URL_EXPIRY=3600

AXIOM_DATASET=$(tf_output axiom_dataset_name)
AXIOM_API_TOKEN=$(tf_output axiom_ingest_token)

API_BASE_URL=$(tf_output api_url)
EOF
