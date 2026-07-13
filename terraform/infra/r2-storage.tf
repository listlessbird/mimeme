moved {
  from = cloudflare_r2_bucket.storage
  to   = cloudflare_r2_bucket.media
}

resource "cloudflare_r2_bucket" "media" {
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
      "com.cloudflare.edge.r2.bucket.${var.cloudflare_account_id}_default_${cloudflare_r2_bucket.media.name}" = "*"
    })
  }]
}

resource "cloudflare_r2_bucket" "artifacts" {
  account_id   = var.cloudflare_account_id
  name         = "mimeme-artifacts-prod"
  location     = "apac"
  jurisdiction = "default"
}

resource "cloudflare_account_token" "artifact_r2" {
  account_id = var.cloudflare_account_id
  name       = local.artifact_r2_token_name

  policies = [{
    effect = "allow"
    permission_groups = [
      { id = local.r2_bucket_item_read_pg },
      { id = local.r2_bucket_item_write_pg },
    ]
    resources = jsonencode({
      "com.cloudflare.edge.r2.bucket.${var.cloudflare_account_id}_default_${cloudflare_r2_bucket.artifacts.name}" = "*"
    })
  }]
}

resource "cloudflare_r2_custom_domain" "media" {
  account_id  = var.cloudflare_account_id
  bucket_name = cloudflare_r2_bucket.media.name
  domain      = local.media_hostname
  enabled     = true
  zone_id     = data.cloudflare_zone.mimeme.id
  min_tls     = "1.2"
}

resource "cloudflare_r2_managed_domain" "media" {
  account_id  = var.cloudflare_account_id
  bucket_name = cloudflare_r2_bucket.media.name
  enabled     = false
}

resource "cloudflare_r2_managed_domain" "artifacts" {
  account_id  = var.cloudflare_account_id
  bucket_name = cloudflare_r2_bucket.artifacts.name
  enabled     = false
}

resource "cloudflare_r2_bucket_lifecycle" "artifact_staging" {
  account_id  = var.cloudflare_account_id
  bucket_name = cloudflare_r2_bucket.artifacts.name
  rules = [{
    id         = "expire-abandoned-staged-uploads"
    enabled    = true
    conditions = { prefix = "uploads/staging/" }
    delete_objects_transition = {
      condition = {
        type    = "Age"
        max_age = 2592000
      }
    }
  }]
}
