output "worker_name" {
  value = cloudflare_worker.this.name
}

output "worker_id" {
  value = cloudflare_worker.this.id
}

output "version_id" {
  value = cloudflare_worker_version.this.id
}

output "deployment_id" {
  value = cloudflare_workers_deployment.this.id
}
