resource "neon_project" "findmeme-db" {
  name                      = "${var.project_name}-${var.environment}-db"
  region_id                 = var.neon_region
  org_id                    = var.neon_org_id
  history_retention_seconds = 21600 # 6 hours - maximum allowed for this plan

  default_endpoint_settings {
    autoscaling_limit_min_cu = 0.25
    autoscaling_limit_max_cu = 1.0
    # suspend_timeout_seconds  = 300
  }

  branch {
    name          = "production"
    database_name = "app_db"
    role_name     = "app_admin"
  }
}


resource "neon_database" "app" {
  project_id = neon_project.findmeme-db.id
  branch_id  = neon_project.findmeme-db.default_branch_id
  name       = "findmeme"
  owner_name = neon_role.app.name
}

resource "neon_role" "app" {
  project_id = neon_project.findmeme-db.id
  branch_id  = neon_project.findmeme-db.default_branch_id
  name       = "findmeme_app"
}
