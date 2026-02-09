

output "r2_bucket_id" {
  value = cloudflare_r2_bucket.storage.id
}

output "r2_bucket_name" {
  value = cloudflare_r2_bucket.storage.name
}

output "cloudflare_account_id" {
  value = var.cloudflare_account_id
}
