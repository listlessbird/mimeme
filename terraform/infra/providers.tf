terraform {
  required_providers {
    axiom = {
      source  = "axiomhq/axiom"
      version = "~> 1.4"
    }
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 5.0"
    }
    neon = {
      source  = "kislerdm/neon"
      version = "0.13.0"
    }
  }
}

provider "axiom" {
  api_token = var.axiom_api_token
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

provider "neon" {
  api_key = var.neon_api_key
}
