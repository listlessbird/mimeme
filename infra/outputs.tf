

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
