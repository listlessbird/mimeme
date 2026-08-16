#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
backend_dir="${repo_root}/backend-api"
terraform_dir="${repo_root}/terraform/infra"
release_id="${RELEASE_ID:-$(git -C "${repo_root}" rev-parse HEAD)}"

if [[ "$(git -C "${repo_root}" rev-parse HEAD)" != "${release_id}" ]]; then
  echo "refusing to deploy Modal from a checkout that does not match RELEASE_ID=${release_id}" >&2
  exit 1
fi

tf_output() {
  terraform -chdir="${terraform_dir}" output -raw "$1"
}

modal_app_name="$(tf_output modal_app_name)"
modal_secret_name="$(tf_output modal_s3_secret_name)"
modal_volume_name="$(tf_output modal_hf_cache_volume_name)"

media_s3_endpoint_url="$(tf_output media_s3_endpoint_url)"
media_s3_region="$(tf_output media_s3_region)"
media_s3_access_key_id="$(tf_output media_s3_access_key_id)"
media_s3_secret_access_key="$(tf_output media_s3_secret_access_key)"
media_s3_bucket="$(tf_output media_s3_bucket_name)"
media_s3_force_path_style="$(tf_output media_s3_force_path_style)"

artifact_s3_endpoint_url="$(tf_output artifact_s3_endpoint_url)"
artifact_s3_region="$(tf_output artifact_s3_region)"
artifact_s3_access_key_id="$(tf_output artifact_s3_access_key_id)"
artifact_s3_secret_access_key="$(tf_output artifact_s3_secret_access_key)"
artifact_s3_bucket="$(tf_output artifact_s3_bucket_name)"
artifact_s3_force_path_style="$(tf_output artifact_s3_force_path_style)"

uv run modal secret create --force "${modal_secret_name}" \
  "MEDIA_S3_ENDPOINT_URL=${media_s3_endpoint_url}" \
  "MEDIA_S3_REGION=${media_s3_region}" \
  "MEDIA_S3_ACCESS_KEY_ID=${media_s3_access_key_id}" \
  "MEDIA_S3_SECRET_ACCESS_KEY=${media_s3_secret_access_key}" \
  "MEDIA_S3_BUCKET=${media_s3_bucket}" \
  "MEDIA_S3_FORCE_PATH_STYLE=${media_s3_force_path_style}" \
  "ARTIFACT_S3_ENDPOINT_URL=${artifact_s3_endpoint_url}" \
  "ARTIFACT_S3_REGION=${artifact_s3_region}" \
  "ARTIFACT_S3_ACCESS_KEY_ID=${artifact_s3_access_key_id}" \
  "ARTIFACT_S3_SECRET_ACCESS_KEY=${artifact_s3_secret_access_key}" \
  "ARTIFACT_S3_BUCKET=${artifact_s3_bucket}" \
  "ARTIFACT_S3_FORCE_PATH_STYLE=${artifact_s3_force_path_style}"

(
  cd "${backend_dir}"
  COMPUTE_MODAL_APP_NAME="${modal_app_name}" \
  COMPUTE_MODAL_HF_CACHE_VOLUME_NAME="${modal_volume_name}" \
  COMPUTE_MODAL_S3_SECRET_NAME="${modal_secret_name}" \
  MIMEME_RELEASE_ID="${release_id}" \
    uv run modal deploy -m mimeme.modal_app.app --name "${modal_app_name}"
)
