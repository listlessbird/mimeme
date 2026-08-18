resource "axiom_dataset" "api_logs" {
  name        = local.axiom_dataset_name
  description = "Mimeme production API logs"
}

resource "axiom_token" "api_ingest" {
  name        = local.axiom_token_name
  description = "Ingest-only token for Mimeme production API logs"

  dataset_capabilities = {
    (axiom_dataset.api_logs.name) = {
      ingest = ["create"]
    }
  }
}

resource "axiom_token" "api_query" {
  name        = local.axiom_query_token_name
  description = "Read-only token for Mimeme admin log diagnostics"

  dataset_capabilities = {
    (axiom_dataset.api_logs.name) = {
      query = ["read"]
    }
  }
}
