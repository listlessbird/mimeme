import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
from temporalio import activity

from activities.storage.models import DownloadImageInput, DownloadImageOutput


@activity.defn
async def download_image_activity(input: DownloadImageInput) -> DownloadImageOutput:
    try:
        parsed = urlparse(input.url)
        filename = Path(parsed.path).name or "image"

        with httpx.Client(timeout=30.0, follow_redirects=True) as httpclient:
            response = httpclient.get(input.url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            if not content_type.startswith("image/"):
                return DownloadImageOutput(
                    ingest_url_id=input.ingest_url_id,
                    local_path="",
                    filename="",
                    success=False,
                    error=f"Couldnt resolve an image from the url {input.url}, got {content_type} as content type",
                )

            suffix = Path(filename).suffix or ".jpg"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(response.content)

                return DownloadImageOutput(
                    ingest_url_id=input.ingest_url_id,
                    local_path=f.name,
                    filename=filename,
                    success=True,
                )

    except Exception as e:
        return DownloadImageOutput(
            ingest_url_id=input.ingest_url_id,
            local_path="",
            filename="",
            success=False,
            error=str(e),
        )
