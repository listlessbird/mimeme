data "cloudflare_account_permission_groups" "r2_read" {
  account_id = var.cloudflare_account_id
  name       = "Workers R2 Storage Read"
}

data "cloudflare_account_permission_groups" "r2_write" {
  account_id = var.cloudflare_account_id
  name       = "Workers R2 Storage Write"
}

resource "cloudflare_r2_bucket" "terraform_state" {
  account_id   = var.cloudflare_account_id
  name         = local.state_bucket_name
  location     = "apac"
  jurisdiction = "default"
}

resource "cloudflare_account_token" "terraform_state" {
  account_id = var.cloudflare_account_id
  name       = local.state_token_name

  policies = [{
    effect = "allow"

    permission_groups = [
      {
        id = data.cloudflare_account_permission_groups.r2_read.result[0].id
      },
      {
        id = data.cloudflare_account_permission_groups.r2_write.result[0].id
      },
    ]

    resources = jsonencode({
      "com.cloudflare.edge.r2.bucket.${var.cloudflare_account_id}_default_${cloudflare_r2_bucket.terraform_state.name}" = "*"
    })
  }]
}
