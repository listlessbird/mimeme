variable "cloudflare_account_id" {
  type = string
}

variable "project_root" {
  type = string
}

variable "worker_name" {
  type = string
}

variable "source_hash" {
  type = string
}

variable "build_command" {
  type = string
}

variable "build_env" {
  type    = map(string)
  default = {}
}

variable "plain_text_bindings" {
  type    = map(string)
  default = {}
}

variable "secret_text_bindings" {
  type      = map(string)
  default   = {}
  sensitive = true
}

variable "dist_server_dir" {
  type    = string
  default = "dist/server"
}

variable "enable_workers_dev_subdomain" {
  type    = bool
  default = true
}

variable "enable_worker_previews" {
  type    = bool
  default = false
}
