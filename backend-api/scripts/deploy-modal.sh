#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
backend_dir="${repo_root}/backend-api"
terraform_dir="${repo_root}/terraform/infra"

tf_output() {
  terraform -chdir="${terraform_dir}" output -raw "$1"
}

modal_app_name="$(tf_output modal_app_name)"
modal_secret_name="$(tf_output modal_s3_secret_name)"
modal_volume_name="$(tf_output modal_hf_cache_volume_name)"

s3_endpoint_url="$(tf_output s3_endpoint_url)"
s3_region="$(tf_output s3_region)"
s3_access_key_id="$(tf_output s3_access_key_id)"
s3_secret_access_key="$(tf_output s3_secret_access_key)"
s3_bucket="$(tf_output s3_bucket_name)"
s3_force_path_style="$(tf_output s3_force_path_style)"

uv run modal secret create --force "${modal_secret_name}" \
  "S3_ENDPOINT_URL=${s3_endpoint_url}" \
  "S3_REGION=${s3_region}" \
  "S3_ACCESS_KEY_ID=${s3_access_key_id}" \
  "S3_SECRET_ACCESS_KEY=${s3_secret_access_key}" \
  "S3_BUCKET=${s3_bucket}" \
  "S3_FORCE_PATH_STYLE=${s3_force_path_style}"

(
  cd "${backend_dir}"
  MODAL_APP_NAME="${modal_app_name}" \
  MODAL_HF_CACHE_VOLUME_NAME="${modal_volume_name}" \
  MODAL_S3_SECRET_NAME="${modal_secret_name}" \
    uv run modal deploy -m modal_app.app --name "${modal_app_name}"
)
