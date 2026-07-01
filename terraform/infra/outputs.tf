output "api_url" {
  value = "https://${local.api_hostname}"
}

output "axiom_dataset_name" {
  value = axiom_dataset.api_logs.name
}

output "axiom_ingest_token" {
  value     = axiom_token.api_ingest.token
  sensitive = true
}

output "cloudflare_account_id" {
  value = var.cloudflare_account_id
}

output "cloudflare_tunnel_id" {
  value = cloudflare_zero_trust_tunnel_cloudflared.api.id
}

output "cloudflare_tunnel_token" {
  value     = data.cloudflare_zero_trust_tunnel_cloudflared_token.api.token
  sensitive = true
}

output "db_url" {
  value     = neon_project.mimeme.connection_uri
  sensitive = true
}

output "db_url_pooler" {
  value     = neon_project.mimeme.connection_uri_pooler
  sensitive = true
}

output "modal_app_name" {
  value = local.modal_app_name
}

output "modal_hf_cache_volume_name" {
  value = local.modal_hf_volume
}

output "modal_s3_secret_name" {
  value = local.modal_s3_secret
}

output "neon_database_name" {
  value = neon_database.app.name
}

output "neon_project_id" {
  value = neon_project.mimeme.id
}

output "s3_access_key_id" {
  value     = cloudflare_account_token.app_r2.id
  sensitive = true
}

output "s3_bucket_name" {
  value = cloudflare_r2_bucket.storage.name
}

output "s3_endpoint_url" {
  value = "https://${var.cloudflare_account_id}.r2.cloudflarestorage.com"
}

output "s3_force_path_style" {
  value = true
}

output "s3_region" {
  value = "auto"
}

output "s3_secret_access_key" {
  value     = sha256(cloudflare_account_token.app_r2.value)
  sensitive = true
}
