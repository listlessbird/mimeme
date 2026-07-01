resource "cloudflare_r2_bucket" "storage" {
  account_id   = var.cloudflare_account_id
  name         = local.r2_bucket_name
  location     = "apac"
  jurisdiction = "default"
}

resource "cloudflare_account_token" "app_r2" {
  account_id = var.cloudflare_account_id
  name       = local.app_r2_token_name

  policies = [{
    effect = "allow"

    permission_groups = [
      {
        id = local.r2_bucket_item_read_pg
      },
      {
        id = local.r2_bucket_item_write_pg
      },
    ]

    resources = jsonencode({
      "com.cloudflare.edge.r2.bucket.${var.cloudflare_account_id}_default_${cloudflare_r2_bucket.storage.name}" = "*"
    })
  }]
}
