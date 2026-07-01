# Mimeme Terraform

This directory has two Terraform roots:

- `tf-state/` provisions/adopts the R2 bucket and Cloudflare token used by Terraform state.
- `infra/` provisions the production app infrastructure.

The `tf-state` root uses local Terraform state. Keep that state file private. If it is lost, import the state bucket and create a replacement backend token.

## Bootstrap State Storage

```bash
cp terraform/tf-state/terraform.tfvars.example terraform/tf-state/terraform.tfvars
terraform -chdir=terraform/tf-state init
terraform -chdir=terraform/tf-state apply
```

Use the outputs to fill `terraform/infra/backend.tfvars`:

```bash
terraform -chdir=terraform/tf-state output -raw state_s3_access_key_id
terraform -chdir=terraform/tf-state output -raw state_s3_secret_access_key
terraform -chdir=terraform/tf-state output -raw state_s3_endpoint_url
```

## Apply App Infrastructure

```bash
cp terraform/infra/backend.tfvars.example terraform/infra/backend.tfvars
cp terraform/infra/terraform.tfvars.example terraform/infra/terraform.tfvars
terraform -chdir=terraform/infra init -backend-config=backend.tfvars
terraform -chdir=terraform/infra apply
```

## Backend Environment

```bash
cd backend-api
just prod-env > .env
```
