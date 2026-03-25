check "modal_deploy_inputs" {
  assert {
    condition = !var.enable_modal_deploy || (
      trimspace(var.modal_token_id) != "" &&
      trimspace(var.modal_token_secret) != ""
    )
    error_message = "enable_modal_deploy=true requires modal_token_id and modal_token_secret."
  }
}

check "webui_deploy_inputs" {
  assert {
    condition = !var.enable_webui_deploy || (
      trimspace(var.api_base_url) != "" &&
      trimspace(var.api_key_readonly) != ""
    )
    error_message = "enable_webui_deploy=true requires api_base_url and api_key_readonly."
  }
}
