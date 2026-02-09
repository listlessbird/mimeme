resource "cloudflare_r2_bucket" "storage" {
  account_id   = var.cloudflare_account_id
  name         = "${var.project_name}-${var.environment}-storage"
  location     = "apac"
  jurisdiction = "default"
}
