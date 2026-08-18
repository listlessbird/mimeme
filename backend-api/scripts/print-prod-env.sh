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

DATABASE_URL=$(tf_output db_url)

TEMPORAL_HOST=localhost:7233
TEMPORAL_NAMESPACE=default

COMPUTE_GPU_BACKEND=modal
COMPUTE_MODAL_APP_NAME=$(tf_output modal_app_name)

MEDIA_S3_ENDPOINT_URL=$(tf_output media_s3_endpoint_url)
MEDIA_S3_REGION=$(tf_output media_s3_region)
MEDIA_S3_ACCESS_KEY_ID=$(tf_output media_s3_access_key_id)
MEDIA_S3_SECRET_ACCESS_KEY=$(tf_output media_s3_secret_access_key)
MEDIA_S3_BUCKET=$(tf_output media_s3_bucket_name)
MEDIA_S3_FORCE_PATH_STYLE=$(tf_output media_s3_force_path_style)
MEDIA_PUBLIC_BASE_URL=$(tf_output media_public_base_url)

ARTIFACT_S3_ENDPOINT_URL=$(tf_output artifact_s3_endpoint_url)
ARTIFACT_S3_REGION=$(tf_output artifact_s3_region)
ARTIFACT_S3_ACCESS_KEY_ID=$(tf_output artifact_s3_access_key_id)
ARTIFACT_S3_SECRET_ACCESS_KEY=$(tf_output artifact_s3_secret_access_key)
ARTIFACT_S3_BUCKET=$(tf_output artifact_s3_bucket_name)
ARTIFACT_S3_FORCE_PATH_STYLE=$(tf_output artifact_s3_force_path_style)

LOG_AXIOM_DATASET=$(tf_output axiom_dataset_name)
LOG_AXIOM_API_TOKEN=$(tf_output axiom_ingest_token)
LOG_AXIOM_QUERY_TOKEN=$(tf_output axiom_query_token)
LOG_AXIOM_REGION=$(tf_output axiom_region)

API_BASE_URL=$(tf_output api_url)
EOF
