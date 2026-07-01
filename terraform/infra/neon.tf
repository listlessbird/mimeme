resource "neon_project" "mimeme" {
  name                      = local.neon_project_name
  region_id                 = var.neon_region
  org_id                    = var.neon_org_id
  history_retention_seconds = 21600

  default_endpoint_settings {
    autoscaling_limit_min_cu = 0.25
    autoscaling_limit_max_cu = 1.0
  }

  branch {
    name          = "production"
    database_name = "app_db"
    role_name     = "app_admin"
  }
}

resource "neon_role" "app" {
  project_id = neon_project.mimeme.id
  branch_id  = neon_project.mimeme.default_branch_id
  name       = local.database_role_name
}

resource "neon_database" "app" {
  project_id = neon_project.mimeme.id
  branch_id  = neon_project.mimeme.default_branch_id
  name       = local.database_name
  owner_name = neon_role.app.name
}
