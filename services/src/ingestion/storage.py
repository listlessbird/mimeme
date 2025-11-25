import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Iterable, Tuple

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig

from .config import CFG


@dataclass(frozen=True)
class S3Settings:
    endpoint_url: Optional[str]
    region: str
    access_key: str
    secret_key: str
    bucket: str
    force_path_style: bool
    prefix: str


# from environment variables -> terraform
def get_s3_settings() -> S3Settings:
    return S3Settings(
        endpoint_url=CFG.s3_endpoint_url,
        region=CFG.s3_region,
        access_key=CFG.s3_access_key_id,
        secret_key=CFG.s3_secret_access_key,
        bucket=CFG.s3_bucket,
        force_path_style=CFG.s3_force_path_style,
        prefix=CFG.s3_prefix,
    )


def s3_client():
    s = get_s3_settings()
    session = boto3.Session(
        aws_access_key_id=s.access_key,
        aws_secret_access_key=s.secret_key,
        region_name=s.region,
    )

    return session.client(
        "s3",
        endpoint_url=s.endpoint_url,
        config=BotoConfig(
            s3={
                "addressing_style": "path" if s.force_path_style else "auto",
            },
        ),
    )


def build_object_key(sha256: str, rel_path: str) -> str:
    ext = Path(rel_path).suffix.lower().lstrip(".") or "bin"
    a, b = sha256[:2], sha256[2:4]
    prefix = get_s3_settings().prefix.rstrip("/")
    return f"{prefix}/{a}/{b}/{sha256}.{ext}"


def ensure_bucket_exists():
    s = get_s3_settings()
    if not s.bucket:
        raise RuntimeError("S3_BUCKET is not set")
    cli = s3_client()
    try:
        cli.head_bucket(Bucket=s.bucket)
    except Exception as e:
        try:
            cli.create_bucket(Bucket=s.bucket)
        except Exception as e:
            print(f"Error creating bucket {s.bucket}: {e}")
            pass


def head_object(key: str) -> Optional[str]:
    cli = s3_client()
    s = get_s3_settings()

    try:
        resp = cli.head_object(Bucket=s.bucket, Key=key)
        return resp.get("ETag", "").strip('"')
    except Exception as e:
        print(f"Error head_object {key}: {e}")
        return None


def upload_file(local_path: Path, key: str) -> str:
    cli = s3_client()
    s = get_s3_settings()

    cfg = TransferConfig(
        multipart_threshold=1024 * 1024 * 8,
        max_concurrency=8,
        multipart_chunksize=1024 * 1024 * 8,
        use_threads=True,
    )

    cli.upload_file(
        Filename=str(local_path),
        Bucket=s.bucket,
        Key=key,
        Config=cfg,
        ExtraArgs={"ACL": "private", "ContentType": _map_content_type(local_path)},
    )
    etag = head_object(key) or ""
    return etag


def download_file(key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cli = s3_client()
    s = get_s3_settings()
    cli.download_file(s.bucket, key, str(dest))


def list_objects(prefix: str) -> list[tuple[str, int]]:
    cli = s3_client()
    s = get_s3_settings()
    token = None
    out: List[Tuple[str, int]] = []

    while True:
        kwargs = dict(Bucket=s.bucket, Prefix=prefix)
        if token:
            kwargs["ContinuationToken"] = token
        resp = cli.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            out.append((obj["Key"], obj["Size"]))
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return out


def _map_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".json": "application/json",
        ".npy": "application/octet-stream",
        ".db": "application/octet-stream",
        ".sqlite3": "application/octet-stream",
    }.get(ext, "application/octet-stream")
