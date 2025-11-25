# Ingestion Pipeline

Image ingestion and processing pipeline for the find-meme project. Scans local images, computes hashes, uploads to S3, and generates annotations using vision models.

## Prerequisites

- **Python 3.11 - 3.13** (required for BitsAndBytes/PyTorch compatibility)
  - Python 3.14+ is NOT supported due to PyTorch's `torch.compile` limitation
  - If using `mise`: `mise use python@3.13`
- uv package manager
- Raw meme images in `data/raw_memes/`

## Installation

```bash
# Install dependencies
uv sync
```

**Note:** If you were previously using Python 3.14, you need to:
```bash
# Switch to Python 3.13
mise use python@3.13

# Recreate virtual environment
rm -rf .venv
uv venv
uv pip install -e .
```

## Configuration

Set environment variables (optional, defaults shown):

```bash
# Database
export INGESTION_DB="data/db.sqlite3"

# Image root directory (where all datasets live)
export INGESTION_IMAGE_ROOT="data/raw_memes"

# Processing
export INGESTION_WORKERS="4"
export INGESTION_BATCH_SIZE="100"

# S3 Storage (for backup/upload)
export S3_ENDPOINT_URL=""
export S3_REGION="auto"
export S3_BUCKET=""
export S3_ACCESS_KEY_ID=""
export S3_SECRET_ACCESS_KEY=""
export S3_FORCE_PATH_STYLE="true"
export S3_PREFIX="memes"
```

## Command Reference

### Core Workflow Commands

#### 1. Scan Images

Scan a directory of images and add them to the database. Defaults to `data/raw_memes` 

```bash
# Scan all images in raw_memes directory (uses default)
uv run ingest scan

# Or specify a custom directory
uv run ingest scan /path/to/custom/directory

# Options
uv run ingest scan --workers 8 --batch-size 200 --no-estimate
```

**What it does:**
- Recursively walks the directory for image files (`.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`)
- Computes SHA256 hash, perceptual hash (phash), and extracts metadata
- Stores relative paths (e.g., `6992-meme-images-dataset-kaggle/image_100.jpg`)
- Upserts into database 

#### 2. Verify Database

Check database contents and view recent entries.

```bash
uv run ingest verify
```

**Output:**
- Total image count
- Table showing 5 most recent images with metadata

#### 3. Annotate Images

Generate captions and OCR text using vision models with real-time progress display.

```bash
# Annotate images (default: batch of 64)
uv run ingest annotate

# Custom batch size and model
uv run ingest annotate --batch-size 32 --model moondream2
```

**What it does:**
- Loads images that need processing (OCR or caption pending)
- Generates captions and extracts text using vision model
- Shows real-time progress with spinner, percentage, and success/fail counts
- Displays a summary table of the last 5 annotated images with their captions and OCR
- Updates `annotations` and `processing` tables atomically (no inconsistent states)
- Creates `artifacts` records for traceability

**Progress Display:**
- Live progress bar with current image name
- Success/fail counters (✓ X / ✗ Y)
- Summary table at the end showing recent annotations

#### 4. Reset Database

Remove all data from the database (clean slate).

```bash
# Interactive confirmation
uv run ingest reset-db

# Skip confirmation
uv run ingest reset-db --yes
```

**Warning:** This deletes ALL data from all tables (images, processing, annotations, artifacts, index_builds).

### S3 Backup/Sync Commands

#### Upload Images to S3

Upload local images to S3 object storage.

```bash
# Dry run (preview uploads, uses default data/raw_memes)
uv run ingest upload --dry-run

# Actually upload
uv run ingest upload

# Or specify custom directory
uv run ingest upload /path/to/custom/directory
```

**What it does:**
- Checks which images need uploading (missing or outdated ETag)
- Uploads to S3 with keys like `memes/{sha256}.{ext}`
- Updates `s3_key` and `s3_etag` in database

#### Backup Database

Create a snapshot of the database and upload to S3.

```bash
# Backup and compress
uv run ingest backup-db

# Backup without compression
uv run ingest backup-db --no-compress
```

**What it does:**
- Creates SQLite backup in `data/db.snapshot-{timestamp}.sqlite3`
- Optionally gzips the backup
- Uploads to S3 under `backups/` prefix

#### Restore Database

Download and restore the latest database backup from S3.

```bash
# Restore to default location
uv run ingest restore-db

# Restore to custom path
uv run ingest restore-db --dest /path/to/restore/db.sqlite3
```

#### Rehydrate Images

Download all images from S3 to local directory.

```bash
uv run ingest rehydrate /path/to/destination
```

**What it does:**
- Downloads all images that have `s3_key` set
- Preserves directory structure based on S3 keys

## Common Workflows

### Initial Setup (Fresh Start)

```bash
# 1. Place images in data/raw_memes/{dataset-name}/
# 2. Scan images
uv run ingest scan

# 3. Verify ingestion
uv run ingest verify

# 4. Generate annotations
uv run ingest annotate

# 5. (Optional) Upload to S3
uv run ingest upload
```

### Adding a New Dataset

```bash
# 1. Add new dataset to data/raw_memes/new-dataset/
# 2. Re-scan (safe, won't duplicate existing images)
uv run ingest scan

# 3. Annotate new images only
uv run ingest annotate
```

### Disaster Recovery

```bash
# 1. Restore database from S3
uv run ingest restore-db

# 2. Rehydrate images from S3
uv run ingest rehydrate data/raw_memes

# 3. Verify restoration
uv run ingest verify
```

### Complete Reset

```bash
# 1. Reset database
uv run ingest reset-db --yes

# 2. Re-scan images
uv run ingest scan

# 3. Re-annotate
uv run ingest annotate
```

### Testing/Development

```bash
# Work with small batches
uv run ingest scan --batch-size 10
uv run ingest annotate --batch-size 5

# Check what would be uploaded
uv run ingest upload --dry-run
```

## Database Schema

- **images**: Core image metadata (sha256, path, dimensions, phash, s3_key)
- **processing**: Processing status for OCR, captions, embeddings
- **annotations**: Generated text (ocr_text, caption_text, tags)
- **artifacts**: Audit trail of model outputs
- **index_builds**: FAISS index metadata for search

