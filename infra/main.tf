
terraform {
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 5"
    }
    neon = {
      source  = "kislerdm/neon"
      version = "0.13.0"
    }
    axiom = {
      source  = "axiomhq/axiom"
      version = "~> 1.4"
    }
  }
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

provider "neon" {
  api_key = var.neon_api_key
}
