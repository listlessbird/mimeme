resource "axiom_monitor" "api_5xx" {
  depends_on          = [axiom_dataset.api_logs]
  name                = "Mimeme API 5xx spike"
  description         = "Alert when the API emits more than five server errors in a five-minute window."
  type                = "Threshold"
  apl_query           = <<-APL
    ${local.axiom_dataset_ref}
    | where event == "http_request"
    | summarize errors = countif(status_code >= 500) by bin(_time, 5m)
  APL
  operator            = "Above"
  threshold           = 5
  interval_minutes    = 5
  range_minutes       = 5
  trigger_from_n_runs = 1
}

resource "axiom_monitor" "api_p95_latency" {
  depends_on          = [axiom_dataset.api_logs]
  name                = "Mimeme API p95 latency"
  description         = "Alert when API p95 latency exceeds one second."
  type                = "Threshold"
  apl_query           = <<-APL
    ${local.axiom_dataset_ref}
    | where event == "http_request"
    | summarize p95_ms = percentile(duration_ms, 95) by bin(_time, 5m)
  APL
  operator            = "Above"
  threshold           = 1000
  interval_minutes    = 5
  range_minutes       = 5
  trigger_from_n_runs = 1
}

resource "axiom_monitor" "healthcheck_degraded" {
  depends_on          = [axiom_dataset.api_logs]
  name                = "Mimeme dependency degraded"
  description         = "Alert on a readiness healthcheck with one or more failed dependencies."
  type                = "MatchEvent"
  apl_query           = <<-APL
    ${local.axiom_dataset_ref}
    | where event == "healthcheck_degraded"
  APL
  interval_minutes    = 1
  range_minutes       = 1
  trigger_from_n_runs = 1
}

resource "axiom_monitor" "ingestion_failures" {
  depends_on          = [axiom_dataset.api_logs]
  name                = "Mimeme ingestion failures"
  description         = "Alert when a completed ingestion job contains more than ten failed items."
  type                = "Threshold"
  apl_query           = <<-APL
    ${local.axiom_dataset_ref}
    | where event == "ingest_job_completed"
    | summarize failed = sum(failed) by bin(_time, 5m)
  APL
  operator            = "Above"
  threshold           = 10
  interval_minutes    = 5
  range_minutes       = 5
  trigger_from_n_runs = 1
}

resource "axiom_monitor" "index_activity_failures" {
  depends_on          = [axiom_dataset.api_logs]
  name                = "Mimeme index activity failure"
  description         = "Alert when an index prepare, seal, build, or activation activity fails."
  type                = "MatchEvent"
  apl_query           = <<-APL
    ${local.axiom_dataset_ref}
    | where event == "index_activity_failed"
  APL
  interval_minutes    = 1
  range_minutes       = 1
  trigger_from_n_runs = 1
}

resource "axiom_monitor" "compute_inference_failures" {
  depends_on          = [axiom_dataset.api_logs]
  name                = "Mimeme GPU inference failure"
  description         = "Alert when the compute gateway reports a failed annotation or embedding job."
  type                = "MatchEvent"
  apl_query           = <<-APL
    ${local.axiom_dataset_ref}
    | where event == "compute_inference_job_completed" and outcome == "failed"
  APL
  interval_minutes    = 1
  range_minutes       = 1
  trigger_from_n_runs = 1
}
