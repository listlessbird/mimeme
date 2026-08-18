locals {
  axiom_dataset_ref = "['${local.axiom_dataset_name}']"

  axiom_dashboard_document = {
    name            = "Mimeme Production Overview"
    owner           = "X-AXIOM-EVERYONE"
    description     = "Operational and ingestion health for the Mimeme production API and worker."
    refreshTime     = 60
    schemaVersion   = 2
    timeWindowStart = "qr-now-24h"
    timeWindowEnd   = "qr-now"
    uid             = "mimeme-production-overview"
    charts = [
      {
        id   = "images-ingested"
        type = "Statistic"
        name = "Images ingested"
        query = {
          apl          = <<-APL
            ${local.axiom_dataset_ref}
            | where event == "ingest_job_completed"
            | summarize images_ingested = sum(processed)
          APL
          queryOptions = { displayNull = "auto" }
        }
      },
      {
        id   = "ingestion-failures"
        type = "Statistic"
        name = "Ingestion failures"
        query = {
          apl          = <<-APL
            ${local.axiom_dataset_ref}
            | where event == "ingest_job_completed"
            | summarize failures = sum(failed)
          APL
          queryOptions = { displayNull = "auto" }
        }
      },
      {
        id   = "searches"
        type = "Statistic"
        name = "Searches"
        query = {
          apl          = <<-APL
            ${local.axiom_dataset_ref}
            | where event == "search_completed"
            | summarize searches = count()
          APL
          queryOptions = { displayNull = "auto" }
        }
      },
      {
        id   = "api-error-rate"
        type = "Statistic"
        name = "API 5xx rate"
        query = {
          apl          = <<-APL
            ${local.axiom_dataset_ref}
            | where event == "http_request"
            | summarize requests = count(), errors = countif(status_code >= 500)
            | project error_rate_percent = 100.0 * errors / requests
          APL
          queryOptions = { displayNull = "auto" }
        }
      },
      {
        id   = "api-p95"
        type = "Statistic"
        name = "API p95 latency"
        query = {
          apl          = <<-APL
            ${local.axiom_dataset_ref}
            | where event == "http_request"
            | summarize p95_ms = percentile(duration_ms, 95)
          APL
          queryOptions = { displayNull = "auto" }
        }
      },
      {
        id   = "images-per-day"
        type = "TimeSeries"
        name = "Images ingested per day"
        query = {
          apl          = <<-APL
            ${local.axiom_dataset_ref}
            | where event == "ingest_job_completed"
            | summarize images = sum(processed) by bin(_time, 1d)
          APL
          queryOptions = { displayNull = "zero", timeSeriesVariant = "line" }
        }
      },
      {
        id   = "ingestion-outcomes"
        type = "TimeSeries"
        name = "Processed / duplicate / failed"
        query = {
          apl          = <<-APL
            ${local.axiom_dataset_ref}
            | where event == "ingest_job_completed"
            | summarize processed = sum(processed), duplicates = sum(duplicates), failed = sum(failed) by bin_auto(_time)
          APL
          queryOptions = { displayNull = "zero", timeSeriesVariant = "area" }
        }
      },
      {
        id   = "search-zero-results"
        type = "TimeSeries"
        name = "Searches and zero-result searches"
        query = {
          apl          = <<-APL
            ${local.axiom_dataset_ref}
            | where event == "search_completed"
            | summarize searches = count(), zero_results = countif(zero_results == true) by bin_auto(_time)
          APL
          queryOptions = { displayNull = "zero", timeSeriesVariant = "line" }
        }
      },
      {
        id   = "http-latency"
        type = "TimeSeries"
        name = "HTTP p50 / p95 / p99"
        query = {
          apl          = <<-APL
            ${local.axiom_dataset_ref}
            | where event == "http_request"
            | summarize p50 = percentile(duration_ms, 50), p95 = percentile(duration_ms, 95), p99 = percentile(duration_ms, 99) by bin_auto(_time)
          APL
          queryOptions = { displayNull = "auto", timeSeriesVariant = "line" }
        }
      },
      {
        id   = "endpoint-performance"
        type = "Table"
        name = "Endpoint performance"
        query = {
          apl          = <<-APL
            ${local.axiom_dataset_ref}
            | where event == "http_request"
            | summarize requests = count(), avg_ms = avg(duration_ms), p95_ms = percentile(duration_ms, 95), errors = countif(status_code >= 500) by method, route
            | sort by p95_ms desc
          APL
          queryOptions = { displayNull = "auto" }
        }
      },
      {
        id   = "db-pressure"
        type = "TimeSeries"
        name = "Database pressure"
        query = {
          apl          = <<-APL
            ${local.axiom_dataset_ref}
            | where event == "http_request"
            | summarize p95_pool_wait_ms = percentile(pool_wait_ms, 95), p95_db_held_ms = percentile(db_held_ms, 95), max_pool_in_use = max(pool_in_use) by bin_auto(_time)
          APL
          queryOptions = { displayNull = "auto", timeSeriesVariant = "line" }
        }
      },
      {
        id   = "source-health"
        type = "TimeSeries"
        name = "Source ingestion health"
        query = {
          apl          = <<-APL
            ${local.axiom_dataset_ref}
            | where event == "source_run_completed"
            | summarize discovered = sum(discovered), queued = sum(queued), duplicates = sum(duplicates), failed = sum(failed) by source_id, bin_auto(_time)
          APL
          queryOptions = { displayNull = "zero", timeSeriesVariant = "line" }
        }
      },
      {
        id   = "dependency-health"
        type = "TimeSeries"
        name = "Dependency health degradation"
        query = {
          apl          = <<-APL
            ${local.axiom_dataset_ref}
            | where event == "healthcheck_degraded"
            | summarize degraded = count(), postgres_failures = countif(postgres == false), media_failures = countif(media_storage == false), artifact_failures = countif(artifact_storage == false), temporal_failures = countif(temporal == false), search_failures = countif(search == false), inference_failures = countif(inference == false) by bin_auto(_time)
          APL
          queryOptions = { displayNull = "zero", timeSeriesVariant = "line" }
        }
      },
      {
        id   = "worker-health"
        type = "TimeSeries"
        name = "Worker activity failures"
        query = {
          apl          = <<-APL
            ${local.axiom_dataset_ref}
            | where event == "index_activity_failed" or event == "ingest_item_attempt_failed" or event == "source_discovery_failed"
            | summarize failures = count() by event, bin_auto(_time)
          APL
          queryOptions = { displayNull = "zero", timeSeriesVariant = "line" }
        }
      },
      {
        id   = "recent-errors"
        type = "LogStream"
        name = "Recent warnings and errors"
        query = {
          apl          = <<-APL
            ${local.axiom_dataset_ref}
            | where app_env == "production"
            | where level == "warning" or level == "error" or status_code >= 500
            | project _time, service, event, route, status_code, job_id, image_id, workflow_id, activity_name, attempt, error
            | sort by _time desc
            | limit 100
          APL
          queryOptions = { displayNull = "auto" }
        }
      },
    ]
    layout = [
      { i = "images-ingested", x = 0, y = 0, w = 3, h = 6 },
      { i = "ingestion-failures", x = 3, y = 0, w = 3, h = 6 },
      { i = "searches", x = 6, y = 0, w = 3, h = 6 },
      { i = "api-error-rate", x = 9, y = 0, w = 3, h = 6 },
      { i = "api-p95", x = 0, y = 6, w = 3, h = 6 },
      { i = "images-per-day", x = 0, y = 12, w = 6, h = 10 },
      { i = "ingestion-outcomes", x = 6, y = 12, w = 6, h = 10 },
      { i = "search-zero-results", x = 0, y = 22, w = 6, h = 10 },
      { i = "http-latency", x = 6, y = 22, w = 6, h = 10 },
      { i = "endpoint-performance", x = 0, y = 32, w = 6, h = 12 },
      { i = "db-pressure", x = 6, y = 32, w = 6, h = 12 },
      { i = "source-health", x = 0, y = 44, w = 6, h = 12 },
      { i = "dependency-health", x = 6, y = 44, w = 6, h = 12 },
      { i = "worker-health", x = 0, y = 56, w = 6, h = 10 },
      { i = "recent-errors", x = 0, y = 66, w = 12, h = 16 },
    ]
  }
}

resource "axiom_dashboard" "production_overview" {
  uid       = local.axiom_dashboard_document.uid
  overwrite = true
  dashboard = jsonencode(local.axiom_dashboard_document)

  depends_on = [axiom_dataset.api_logs]
}
