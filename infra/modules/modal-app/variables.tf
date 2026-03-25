variable "modal_token_id" {
  type      = string
  sensitive = true
}

variable "modal_token_secret" {
  type      = string
  sensitive = true
}

variable "app_name" {
  type = string
}

variable "deploy_path" {
  type = string
}

variable "deploy_target" {
  type = string
}

variable "deploy_env" {
  type    = map(string)
  default = {}
}

variable "source_hash" {
  type = string
}

variable "volume_names" {
  type    = list(string)
  default = []
}

variable "secrets" {
  type = list(object({
    name   = string
    values = map(string)
  }))
  default = []
}
