variable "cloudflare_api_token" {
  type        = string
  sensitive   = true
  description = "Cloudflare API token allowed to create the Terraform state bucket and backend token."
}

variable "cloudflare_account_id" {
  type        = string
  description = "Cloudflare account ID that owns R2."
}

variable "state_bucket_name" {
  type        = string
  description = "R2 bucket used for Terraform state."
}

variable "state_token_name" {
  type        = string
  description = "Cloudflare account token name for Terraform state access."
}
