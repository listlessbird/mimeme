

output "r2_bucket_id" {
  value = cloudflare_r2_bucket.storage.id
}

output "r2_bucket_name" {
  value = cloudflare_r2_bucket.storage.name
}

output "cloudflare_account_id" {
  value = var.cloudflare_account_id
}


output "neon_connection_uri" {
  value     = neon_project.findmeme-db.connection_uri
  sensitive = true
}

output "neon_connection_uri_pooler" {
  value     = neon_project.findmeme-db.connection_uri_pooler
  sensitive = true
}

output "neon_project_id" {
  value = neon_project.findmeme-db.id
}

output "neon_database_name" {
  value = neon_database.app.name
}

output "axiom_dataset_name" {
  value = axiom_dataset.api_logs.name
}

output "modal_app_name" {
  value = var.enable_modal_deploy ? module.modal_app[0].app_name : null
}

output "modal_hf_cache_volume_name" {
  value = var.enable_modal_deploy && length(module.modal_app[0].volume_names) > 0 ? module.modal_app[0].volume_names[0] : null
}

output "webui_worker_name" {
  value = var.enable_webui_deploy ? module.webui_worker[0].worker_name : null
}

output "webui_worker_url" {
  value = var.enable_webui_deploy ? local.webui_server_url : null
}
