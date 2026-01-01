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

## Database Migrations

This project uses [Alembic](https://alembic.sqlalchemy.org/) for database schema migrations with PostgreSQL. Alembic tracks schema changes and allows you to version control your database structure.

### Migration Workflow

#### When to Create a Migration

Create a migration whenever you:
- Add, remove, or modify columns in ORM models
- Add or remove tables
- Change column types, constraints, or indexes
- Modify relationships between tables

#### Creating a New Migration

1. **Make changes to your ORM models** in `src/api/models/orm.py`

2. **Generate a migration** (Alembic will auto-detect changes):
   ```bash
   uv run alembic revision --autogenerate -m "Description of changes"
   ```

3. **Review the generated migration** in `alembic/versions/`:
   - Check that the detected changes are correct
   - Verify both `upgrade()` and `downgrade()` functions
   - Alembic will generate proper PostgreSQL DDL

4. **Apply the migration**:
   ```bash
   uv run alembic upgrade head
   ```

#### Common Migration Commands

```bash
# Check current migration version
uv run alembic current

# View migration history
uv run alembic history

# Upgrade to latest version
uv run alembic upgrade head

# Downgrade one version
uv run alembic downgrade -1

# Downgrade to specific version
uv run alembic downgrade <revision_id>

# Show SQL without executing (dry run)
uv run alembic upgrade head --sql

# Stamp database with specific version (without running migrations)
uv run alembic stamp head
```

### PostgreSQL-Specific Features

PostgreSQL provides full DDL support, so Alembic can:
- Add/drop columns without recreating tables
- Modify column types with automatic type conversion
- Add/drop constraints and indexes efficiently
- Use transactional DDL (migrations run in transactions)

### Migration Best Practices

- **Always review** auto-generated migrations before applying
- **Test migrations** on a copy of your database first
- **Commit migrations** to version control along with model changes
- **Never edit applied migrations** - create a new one instead
- **Add descriptive messages** to help others understand the change

### Example Migration Workflow

```bash
# 1. Update your model
# Edit src/api/models/orm.py

# 2. Generate migration
uv run alembic revision --autogenerate -m "Add user_id to images table"

# 3. Review the generated file in alembic/versions/

# 4. Apply migration
uv run alembic upgrade head

# 5. Verify it worked
uv run alembic current
```

### Initial Database Setup

For a fresh PostgreSQL database:

```bash
# 1. Start PostgreSQL with Docker Compose
docker compose up postgres -d

# 2. Apply all migrations
uv run alembic upgrade head

# 3. Verify tables were created
docker exec findmeme-postgres psql -U findmeme -d findmeme -c "\dt"
```

### Troubleshooting

**Check migration status:**
```bash
# See current version
uv run alembic current

# See all migrations and which are applied
uv run alembic history --verbose
```

**Reset database (WARNING: destructive):**
```bash
# Drop all tables and recreate from migrations
docker compose down -v postgres  # Destroys data!
docker compose up postgres -d
uv run alembic upgrade head
```

**Manual database inspection:**
```bash
# Connect to PostgreSQL
docker exec -it findmeme-postgres psql -U findmeme -d findmeme

# Useful commands in psql:
# \dt              - list tables
# \d table_name    - describe table schema
# \di              - list indexes
# \df              - list functions
# \q               - quit
```

## Local Development Setup

For the best development experience, run infrastructure services in Docker and your application code locally. This gives you fast iteration, easy debugging, and the ability to set breakpoints.

### Architecture Overview

The application consists of:
- **FastAPI** - REST API server
- **Celery Workers** - Async task processors
- **Celery Beat** - Periodic task scheduler (optional)
- **PostgreSQL** - Primary database
- **Redis** - Celery broker and cache
- **MinIO** - S3-compatible object storage

### Recommended Local Setup

**Run in Docker (infrastructure):**
- PostgreSQL
- Redis
- MinIO

**Run locally (your code):**
- FastAPI server
- Celery worker(s)
- Celery beat (if needed)

### Step-by-Step Instructions

#### 1. Start Infrastructure Services

```bash
cd services

# Start only core infrastructure
docker compose up postgres redis minio createbucket

# Or with monitoring tools (Flower for Celery, Drizzle for DB)
docker compose --profile monitoring up postgres redis minio createbucket flower drizzle-gateway
```

**Access Points:**
- MinIO Console: http://localhost:9001 (minioadmin / minioadmin)
- Flower (Celery monitoring): http://localhost:5555 (with `--profile monitoring`)
- Drizzle Gateway (DB explorer): http://localhost:4983 (with `--profile monitoring`)

#### 2. Configure Environment Variables

Create or update `.env` file in `services/src/api/` directory:

```bash
# Database
DATABASE_URL=postgresql://findmeme:findmeme@localhost:5432/findmeme

# Redis (Celery broker)
REDIS_URL=redis://localhost:6379/0

# MinIO (S3 storage)
S3_ENDPOINT_URL=http://localhost:9000
S3_BUCKET=findmeme
S3_ACCESS_KEY_ID=minioadmin
S3_SECRET_ACCESS_KEY=minioadmin
S3_REGION=us-east-1
S3_FORCE_PATH_STYLE=true

# Application
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO

# Model settings
EMBED_MODEL=clip-vit-base-patch32
EMBED_DEVICE=cpu  # or 'cuda' if you have GPU
```

#### 3. Run FastAPI Server

```bash
# Terminal 1: API with hot-reload
cd services
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Access the API:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/live

#### 4. Run Celery Worker

```bash
# Terminal 2: Celery worker for task processing
cd services
uv run celery -A api.tasks.celery_app worker --loglevel=INFO --concurrency=2 --queues=default,ingest,index

# For auto-reload during development (restarts on code changes)
uv run watchfiles 'celery -A api.tasks.celery_app worker --loglevel=DEBUG --concurrency=1' src/
```

You'll see task execution logs in this terminal as they're processed.

#### 5. Run Celery Beat (Optional - for scheduled tasks)

```bash
# Terminal 3: Celery beat scheduler
cd services
uv run celery -A api.tasks.celery_app beat --loglevel=INFO
```

Only needed if you're working with periodic tasks (like daily index rebuilds).

### Docker Compose Profiles

The docker-compose.yml uses profiles to organize services:

```bash
# Default (no profile) - infrastructure only
docker compose up

# With monitoring tools
docker compose --profile monitoring up

# Everything (including GPU worker and beat)
docker compose --profile full up

# Custom combination
docker compose --profile monitoring --profile gpu up
```

**Available profiles:**
- `monitoring` - Adds Flower (Celery UI) and Drizzle Gateway (DB explorer)
- `gpu` - Adds GPU-enabled worker for embedding/ML tasks
- `full` - All services including beat scheduler

### Database Monitoring with Drizzle Gateway

Drizzle Gateway provides a modern web interface for exploring your PostgreSQL database:

1. Start with monitoring profile:
   ```bash
   docker compose --profile monitoring up drizzle-gateway
   ```

2. Access at http://localhost:4983

3. Connect to your database:
   - Host: `postgres` (from Docker) or `localhost` (if outside Docker)
   - Port: `5432`
   - Database: `findmeme`
   - User: `findmeme`
   - Password: `findmeme`

You can set a master password via environment variable:
```bash
export DRIZZLE_MASTERPASS=your_secure_password
```

### Development Tips

**Fast Iteration:**
- Use `--reload` with uvicorn for FastAPI auto-reload
- Use `watchfiles` with Celery for worker auto-reload
- Keep terminals visible to see logs in real-time

**Debugging:**
- Set breakpoints in your IDE for both FastAPI and Celery tasks
- Use `import pdb; pdb.set_trace()` for interactive debugging
- Check Flower UI (http://localhost:5555) to monitor task queues

**Testing Tasks:**
- Submit tasks via API endpoints
- Watch them execute in the Celery worker terminal
- Check results in Flower or directly via Redis

**Common Issues:**
- If services can't connect, ensure Docker services are healthy: `docker compose ps`
- If tasks aren't executing, check Redis is running and worker is connected
- For connection errors, verify environment variables match Docker service names vs localhost

### Production Deployment

For production, run everything in Docker:

```bash
# Full stack with all services
docker compose --profile full up -d

# Or customize for your needs
docker compose up -d postgres redis minio api worker worker-gpu beat
```

This ensures consistent environment and easier deployment to cloud platforms.