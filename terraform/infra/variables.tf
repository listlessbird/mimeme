variable "axiom_api_token" {
  type        = string
  sensitive   = true
  description = "Axiom token allowed to create datasets and API tokens."
}

variable "cloudflare_api_token" {
  type        = string
  sensitive   = true
  description = "Cloudflare token allowed to manage R2, DNS, and Cloudflare Tunnel resources."
}

variable "cloudflare_account_id" {
  type        = string
  description = "Cloudflare account ID that owns R2 and Cloudflare Tunnel."
}

variable "cloudflare_zone_name" {
  type        = string
  description = "Cloudflare DNS zone for public hostnames."
  default     = "mimeme.dev"
}

variable "api_hostname" {
  type        = string
  description = "Public API hostname routed through Cloudflare Tunnel."
}

variable "neon_api_key" {
  type        = string
  sensitive   = true
  description = "Neon API key allowed to manage the production database project."
}

variable "neon_org_id" {
  type        = string
  sensitive   = true
  description = "Neon organization ID."
}

variable "neon_region" {
  type        = string
  description = "Neon region for the production project."
  default     = "aws-ap-southeast-1"
}

variable "api_origin_service" {
  type        = string
  description = "Local service URL that cloudflared on the Raspberry Pi forwards API traffic to."
  default     = "http://localhost:8000"
}

variable "app_r2_token_name" {
  type        = string
  description = "Cloudflare token name for backend and Modal access to the app R2 bucket."
}

variable "artifact_r2_token_name" {
  type        = string
  description = "Cloudflare token name scoped to the private artifact bucket."
  default     = "mimeme-prod-r2-artifacts"
}

variable "media_hostname" {
  type        = string
  description = "Canonical public hostname for media objects."
  default     = "assets.mimeme.dev"
}

variable "axiom_dataset_name" {
  type        = string
  description = "Axiom dataset name for production API logs."
}

variable "axiom_token_name" {
  type        = string
  description = "Axiom ingest token name for production API logs."
}

variable "database_name" {
  type        = string
  description = "Application database name."
}

variable "database_role_name" {
  type        = string
  description = "Application database role name."
}

variable "modal_app_name" {
  type        = string
  description = "Modal app name used by backend-api."
}

variable "modal_hf_cache_volume_name" {
  type        = string
  description = "Modal volume name for Hugging Face cache."
}

variable "modal_s3_secret_name" {
  type        = string
  description = "Modal secret name containing S3/R2 credentials."
}

variable "neon_project_name" {
  type        = string
  description = "Neon project name."
}

variable "r2_bucket_name" {
  type        = string
  description = "Private R2 bucket name used by the backend."
}
