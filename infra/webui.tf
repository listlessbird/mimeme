module "webui_worker" {
  count  = var.enable_webui_deploy ? 1 : 0
  source = "./modules/cloudflare-worker-app"

  cloudflare_account_id = var.cloudflare_account_id

  project_root = local.webui_project_root
  worker_name  = local.webui_worker_name
  source_hash  = local.webui_source_hash

  build_command = "npm run build"

  build_env = merge(
    {
      API_BASE_URL     = var.api_base_url
      API_KEY_READONLY = var.api_key_readonly
      VITE_APP_TITLE   = var.webui_vite_app_title
    },
    local.webui_server_url != null ? {
      SERVER_URL = local.webui_server_url
    } : {}
  )

  plain_text_bindings = merge(
    {
      API_BASE_URL = var.api_base_url
    },
    local.webui_server_url != null ? {
      SERVER_URL = local.webui_server_url
    } : {}
  )

  secret_text_bindings = {
    API_KEY_READONLY = var.api_key_readonly
  }
}
