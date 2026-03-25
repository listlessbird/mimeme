#!/usr/bin/env bash
set -euo pipefail

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required to manage Modal secrets"
  exit 1
fi

if ! echo "${SECRETS_JSON}" | jq empty >/dev/null 2>&1; then
  echo "Error: SECRETS_JSON is not valid JSON"
  exit 1
fi

echo "Creating/updating Modal secrets"

echo "${SECRETS_JSON}" | jq -c '.[]' | while IFS= read -r secret; do
  secret_name=$(echo "${secret}" | jq -r '.name')

  declare -a args=()

  while IFS= read -r entry; do
    key=$(echo "${entry}" | jq -r '.key')
    value=$(echo "${entry}" | jq -r '.value')
    args+=("${key}=${value}")
  done < <(echo "${secret}" | jq -c '.values | to_entries | .[]')

  modal secret create "${secret_name}" "${args[@]}" --force
done

echo "Modal secrets ready"
