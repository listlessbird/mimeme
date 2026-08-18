locals {
  project_slug = "mimeme"
  stack        = "prod"
  name_prefix  = "${local.project_slug}-${local.stack}"

  api_hostname           = var.api_hostname
  app_r2_token_name      = var.app_r2_token_name
  artifact_r2_token_name = var.artifact_r2_token_name
  media_hostname         = var.media_hostname
  axiom_dataset_name     = var.axiom_dataset_name
  axiom_token_name       = var.axiom_token_name
  axiom_query_token_name = var.axiom_query_token_name
  database_name          = var.database_name
  database_role_name     = var.database_role_name
  modal_app_name         = var.modal_app_name
  modal_hf_volume        = var.modal_hf_cache_volume_name
  modal_s3_secret        = var.modal_s3_secret_name
  neon_project_name      = var.neon_project_name
  r2_bucket_name         = var.r2_bucket_name

  r2_bucket_item_read_pg  = "6a018a9f2fc74eb6b293b0c548f38b39"
  r2_bucket_item_write_pg = "2efd5506f9c8494dacb1fa10a3e7d5b6"
}
