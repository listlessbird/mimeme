
provider "axiom" {
  api_token = var.axiom_api_token
}

resource "axiom_dataset" "api_logs" {
  name        = "${var.project_name}-${var.environment}-api"
  description = "mimeme api and worker logs"
}
