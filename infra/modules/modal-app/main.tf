locals {
  secrets_json = jsonencode(var.secrets)
}

resource "null_resource" "modal_secrets" {
  count = length(var.secrets) > 0 ? 1 : 0

  triggers = {
    secrets_hash = sha256(local.secrets_json)
  }

  provisioner "local-exec" {
    command     = "${path.module}/scripts/create-secrets.sh"
    interpreter = ["bash"]

    environment = {
      MODAL_TOKEN_ID     = var.modal_token_id
      MODAL_TOKEN_SECRET = var.modal_token_secret
      SECRETS_JSON       = local.secrets_json
    }
  }
}

resource "null_resource" "modal_volume" {
  for_each = toset(var.volume_names)

  triggers = {
    volume_name = each.value
  }

  provisioner "local-exec" {
    command     = "modal volume create ${each.value} || echo 'Volume may already exist'"
    interpreter = ["bash", "-lc"]

    environment = {
      MODAL_TOKEN_ID     = var.modal_token_id
      MODAL_TOKEN_SECRET = var.modal_token_secret
    }
  }
}

resource "null_resource" "modal_deploy" {
  triggers = {
    app_name      = var.app_name
    deploy_path   = var.deploy_path
    deploy_target = var.deploy_target
    deploy_env    = sha256(jsonencode(var.deploy_env))
    source_hash   = var.source_hash
    secrets_id    = length(var.secrets) > 0 ? null_resource.modal_secrets[0].id : "no-secrets"
    volumes_hash  = sha256(join(",", sort(var.volume_names)))
  }

  provisioner "local-exec" {
    command     = "${path.module}/scripts/deploy.sh"
    interpreter = ["bash"]

    environment = merge(var.deploy_env, {
      MODAL_TOKEN_ID     = var.modal_token_id
      MODAL_TOKEN_SECRET = var.modal_token_secret
      APP_NAME           = var.app_name
      DEPLOY_PATH        = var.deploy_path
      DEPLOY_TARGET      = var.deploy_target
    })
  }

  depends_on = [null_resource.modal_secrets, null_resource.modal_volume]
}
