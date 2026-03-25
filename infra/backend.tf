# Terraform State Backend Configuration
# Uses Cloudflare R2 (S3-compatible storage)
#
# Prerequisites:
# 1. Create the state bucket once outside this root module:
#      wrangler r2 bucket create findmeme-terraform-state
# 2. Generate an R2 API token with read/write permissions
# 3. Initialize Terraform with backend config:
#      terraform init -backend-config=backend.tfvars
#
# This root module cannot create the same bucket that stores its own state,
# so the backend bucket remains a one-time bootstrap step.

terraform {
  backend "s3" {
    bucket = "findmeme-terraform-state"
    key    = "dev/terraform.tfstate"
    region = "auto"

    # All sensitive/account-specific values should be passed via -backend-config.
    # You can also override "key" there per environment, for example:
    #   key = "prod/terraform.tfstate"

    # Required for Cloudflare R2 compatibility
    skip_credentials_validation = true
    skip_metadata_api_check     = true
    skip_region_validation      = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true
    use_path_style              = true
  }
}
