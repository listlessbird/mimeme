module "modal_app" {
  count  = var.enable_modal_deploy ? 1 : 0
  source = "./modules/modal-app"

  modal_token_id     = var.modal_token_id
  modal_token_secret = var.modal_token_secret

  app_name      = var.modal_app_name
  deploy_path   = local.backend_project_root
  deploy_target = "src/modal_app/app.py"
  deploy_env = {
    MODAL_APP_NAME             = var.modal_app_name
    MODAL_HF_CACHE_VOLUME_NAME = var.modal_hf_cache_volume_name
    MODAL_S3_SECRET_NAME       = var.modal_s3_secret_name
  }
  source_hash  = local.modal_source_hash
  volume_names = local.modal_volume_names

  secrets = [
    {
      name = var.modal_s3_secret_name
      values = {
        S3_ENDPOINT_URL      = "https://${var.cloudflare_account_id}.r2.cloudflarestorage.com"
        S3_REGION            = var.s3_region
        S3_ACCESS_KEY_ID     = var.r2_access_id
        S3_SECRET_ACCESS_KEY = var.r2_secret_key
        S3_BUCKET            = cloudflare_r2_bucket.storage.name
        S3_FORCE_PATH_STYLE  = tostring(var.s3_force_path_style)
      }
    }
  ]
}
