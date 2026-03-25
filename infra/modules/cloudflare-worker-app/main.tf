terraform {
  required_providers {
    cloudflare = {
      source = "cloudflare/cloudflare"
    }
    external = {
      source = "hashicorp/external"
    }
    null = {
      source = "hashicorp/null"
    }
  }
}

locals {
  build_env_hash = sha256(jsonencode(var.build_env))

  build_manifest = data.external.build_manifest.result

  compatibility_flags = try(jsondecode(local.build_manifest.compatibility_flags_json), [])
  modules = [
    for module in try(jsondecode(local.build_manifest.modules_json), []) : {
      name         = module.name
      content_file = module.content_file
      content_type = module.content_type
    }
  ]
  bindings = concat(
    [
      for name in sort(keys(var.plain_text_bindings)) : {
        name = name
        text = var.plain_text_bindings[name]
        type = "plain_text"
      }
    ],
    [
      for name in sort(keys(var.secret_text_bindings)) : {
        name = name
        text = var.secret_text_bindings[name]
        type = "secret_text"
      }
    ]
  )
}

resource "null_resource" "build" {
  triggers = {
    source_hash = var.source_hash
    build_env   = local.build_env_hash
    command     = var.build_command
  }

  provisioner "local-exec" {
    command     = var.build_command
    working_dir = var.project_root

    environment = var.build_env
  }
}

data "external" "build_manifest" {
  program = ["bash", "${path.module}/scripts/read-build-manifest.sh"]

  query = {
    build_id        = null_resource.build.id
    dist_server_dir = "${var.project_root}/${var.dist_server_dir}"
  }

  depends_on = [null_resource.build]
}

resource "cloudflare_worker" "this" {
  account_id = var.cloudflare_account_id
  name       = var.worker_name

  subdomain = {
    enabled          = var.enable_workers_dev_subdomain
    previews_enabled = var.enable_worker_previews
  }
}

resource "cloudflare_worker_version" "this" {
  account_id = var.cloudflare_account_id
  worker_id  = cloudflare_worker.this.id

  annotations = {
    workers_message = "Terraform deploy ${substr(var.source_hash, 0, 12)}"
    workers_tag     = substr(var.source_hash, 0, 12)
  }

  bindings            = local.bindings
  compatibility_date  = local.build_manifest.compatibility_date
  compatibility_flags = local.compatibility_flags
  main_module         = local.build_manifest.main_module
  modules             = local.modules
  assets              = local.build_manifest.assets_directory != "" ? { directory = local.build_manifest.assets_directory } : null
  placement           = local.build_manifest.placement_mode != "" ? { mode = local.build_manifest.placement_mode } : null

  depends_on = [cloudflare_worker.this, null_resource.build]
}

resource "cloudflare_workers_deployment" "this" {
  account_id  = var.cloudflare_account_id
  script_name = cloudflare_worker.this.name
  strategy    = "percentage"

  versions = [{
    percentage = 100
    version_id = cloudflare_worker_version.this.id
  }]
}
