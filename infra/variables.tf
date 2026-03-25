variable "cloudflare_api_token" {
  type      = string
  sensitive = true
}

variable "cloudflare_account_id" {
  type = string
}

variable "r2_access_id" {
  type      = string
  sensitive = true
}

variable "r2_secret_key" {
  type      = string
  sensitive = true
}

variable "environment" {
  type    = string
  default = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be dev/staging/prod"
  }
}

variable "neon_api_key" {
  type      = string
  sensitive = true
}

variable "neon_region" {
  type    = string
  default = "aws-ap-southeast-1"
}

variable "neon_org_id" {
  type      = string
  sensitive = true
}

variable "project_name" {
  type = string
}

variable "axiom_api_token" {
  type      = string
  sensitive = true
}

variable "backend_project_root" {
  type        = string
  description = "Path to the backend-api repo relative to this Terraform root"
  default     = "../backend-api"
}

variable "webui_project_root" {
  type        = string
  description = "Path to the webui repo relative to this Terraform root"
  default     = "../webui"
}

variable "modal_token_id" {
  type        = string
  sensitive   = true
  description = "Modal API token ID"
  default     = ""
}

variable "modal_token_secret" {
  type        = string
  sensitive   = true
  description = "Modal API token secret"
  default     = ""
}

variable "modal_app_name" {
  type        = string
  description = "Deployed Modal app name used by the backend"
  default     = "findmeme-gpu"
}

variable "modal_s3_secret_name" {
  type        = string
  description = "Modal secret name that stores S3/R2 credentials"
  default     = "findmeme-s3"
}

variable "modal_hf_cache_volume_name" {
  type        = string
  description = "Modal volume name for Hugging Face cache"
  default     = "findmeme-hf-cache"
}

variable "s3_region" {
  type        = string
  description = "Region value passed to boto3 for the R2-backed S3 client"
  default     = "auto"
}

variable "s3_force_path_style" {
  type        = bool
  description = "Whether the S3 client should force path-style addressing"
  default     = true
}

variable "enable_modal_deploy" {
  type        = bool
  description = "Whether terraform apply should deploy the Modal app"
  default     = true
}

variable "enable_webui_deploy" {
  type        = bool
  description = "Whether terraform apply should deploy the webui Cloudflare Worker"
  default     = true
}

variable "cloudflare_workers_subdomain" {
  type        = string
  description = "Cloudflare Workers account subdomain, used to compute workers.dev URLs"
  default     = ""
}

variable "webui_worker_name" {
  type        = string
  description = "Optional explicit Cloudflare Worker name for the webui"
  default     = ""
}

variable "webui_server_url" {
  type        = string
  description = "Optional explicit public URL for the webui Worker"
  default     = ""
}

variable "webui_vite_app_title" {
  type        = string
  description = "Optional app title injected at build time"
  default     = "Find Meme"
}

variable "api_base_url" {
  type        = string
  description = "Public base URL for the backend API, used by the webui worker"
  default     = ""
}

variable "api_key_readonly" {
  type        = string
  sensitive   = true
  description = "Readonly API key injected into the webui worker as a secret"
  default     = ""
}
