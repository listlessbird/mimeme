output "state_bucket_name" {
  value = cloudflare_r2_bucket.terraform_state.name
}

output "state_bucket_backend_key" {
  value = "prod/terraform.tfstate"
}

output "state_s3_endpoint_url" {
  value = "https://${var.cloudflare_account_id}.r2.cloudflarestorage.com"
}

output "state_s3_access_key_id" {
  value = cloudflare_account_token.terraform_state.id
}

output "state_s3_secret_access_key" {
  value     = sha256(cloudflare_account_token.terraform_state.value)
  sensitive = true
}
