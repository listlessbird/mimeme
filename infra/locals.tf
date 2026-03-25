locals {
  backend_project_root = abspath("${path.root}/${var.backend_project_root}")
  webui_project_root   = abspath("${path.root}/${var.webui_project_root}")

  modal_volume_names = compact([var.modal_hf_cache_volume_name])

  modal_source_files = sort(concat(
    [
      for file in fileset(local.backend_project_root, "src/modal_app/**/*.py") :
      "${local.backend_project_root}/${file}"
    ],
    [
      for file in ["pyproject.toml", "uv.lock"] :
      "${local.backend_project_root}/${file}"
      if fileexists("${local.backend_project_root}/${file}")
    ]
  ))

  modal_source_hash = sha256(join("", [
    for file in local.modal_source_files : filesha256(file)
  ]))

  webui_source_files = sort(concat(
    [
      for file in fileset(local.webui_project_root, "src/**") :
      "${local.webui_project_root}/${file}"
    ],
    [
      for file in fileset(local.webui_project_root, "public/**") :
      "${local.webui_project_root}/${file}"
    ],
    [
      for file in [
        "package.json",
        "wrangler.jsonc",
        "tsconfig.json",
        "vite.config.ts",
        "vite.config.js",
        "bun.lock",
        "bun.lockb",
        "package-lock.json",
        "pnpm-lock.yaml",
      ] :
      "${local.webui_project_root}/${file}"
      if fileexists("${local.webui_project_root}/${file}")
    ]
  ))

  webui_source_hash = sha256(join("", [
    for file in local.webui_source_files : filesha256(file)
  ]))

  webui_worker_name = trimspace(var.webui_worker_name) != "" ? var.webui_worker_name : "${var.project_name}-${var.environment}-webui"
  webui_server_url = trimspace(var.webui_server_url) != "" ? var.webui_server_url : (
    trimspace(var.cloudflare_workers_subdomain) != "" ?
    "https://${local.webui_worker_name}.${var.cloudflare_workers_subdomain}.workers.dev" :
    null
  )
}
