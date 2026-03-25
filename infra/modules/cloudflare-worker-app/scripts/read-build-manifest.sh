#!/usr/bin/env bash
set -euo pipefail

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required to read the Cloudflare build manifest" >&2
  exit 1
fi

query_json=$(cat)
dist_server_dir=$(echo "${query_json}" | jq -r '.dist_server_dir')

if [[ -z "${dist_server_dir}" || "${dist_server_dir}" == "null" ]]; then
  echo "Error: dist_server_dir is required" >&2
  exit 1
fi

manifest_path="${dist_server_dir}/wrangler.json"

if [[ ! -f "${manifest_path}" ]]; then
  echo "Error: expected build manifest at ${manifest_path}" >&2
  exit 1
fi

main_module=$(jq -r '.main // empty' "${manifest_path}")
compatibility_date=$(jq -r '.compatibility_date // empty' "${manifest_path}")
compatibility_flags_json=$(jq -c '.compatibility_flags // []' "${manifest_path}")
placement_mode=$(jq -r '.placement.mode // empty' "${manifest_path}")

assets_directory=""
assets_dir_rel=$(jq -r '.assets.directory // empty' "${manifest_path}")
if [[ -n "${assets_dir_rel}" ]]; then
  assets_directory=$(cd "${dist_server_dir}" && cd "${assets_dir_rel}" && pwd -P)
fi

modules_json=$(
  find "${dist_server_dir}" -type f \( -name '*.js' -o -name '*.mjs' -o -name '_headers' -o -name '_redirects' \) \
    ! -name 'wrangler.json' \
    | LC_ALL=C sort \
    | while IFS= read -r file_path; do
        relative_path="${file_path#${dist_server_dir}/}"

        case "${relative_path}" in
          _headers|_redirects)
            content_type="text/plain"
            ;;
          *.js|*.mjs)
            content_type="application/javascript+module"
            ;;
          *)
            continue
            ;;
        esac

        jq -nc \
          --arg name "${relative_path}" \
          --arg content_file "${file_path}" \
          --arg content_type "${content_type}" \
          '{name: $name, content_file: $content_file, content_type: $content_type}'
      done \
    | jq -sc '.'
)

jq -nc \
  --arg assets_directory "${assets_directory}" \
  --arg compatibility_date "${compatibility_date}" \
  --arg compatibility_flags_json "${compatibility_flags_json}" \
  --arg main_module "${main_module}" \
  --arg modules_json "${modules_json}" \
  --arg placement_mode "${placement_mode}" \
  '{
    assets_directory: $assets_directory,
    compatibility_date: $compatibility_date,
    compatibility_flags_json: $compatibility_flags_json,
    main_module: $main_module,
    modules_json: $modules_json,
    placement_mode: $placement_mode
  }'
