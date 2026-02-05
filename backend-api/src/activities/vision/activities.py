import tempfile
from pathlib import Path
from typing import cast

from PIL import Image
from temporalio import activity

from activities.vision.models import CaptionInput, CaptionOutput, OCRInput, OCROutput
from activities.vision.moondream import Moondream2
from shared.services import StorageService, get_storage_service


@activity.defn
async def caption_activity(input: CaptionInput) -> CaptionOutput:
    storage = cast(StorageService, get_storage_service())

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        tmp_path = Path(tmp.name)
        storage.download_file(input.s3_key, tmp_path)

        pil_image = Image.open(tmp_path).convert("RGB")

        model = Moondream2.get_instance()
        result = model.caption(pil_image, length=input.length)

        return CaptionOutput(image_id=input.image_id, caption=result.caption, model=result.model)


@activity.defn
async def ocr_activity(input: OCRInput) -> OCROutput:
    storage = cast(StorageService, get_storage_service())

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        tmp_path = Path(tmp.name)
        storage.download_file(input.s3_key, tmp_path)

        pil_image = Image.open(tmp_path).convert("RGB")

        model = Moondream2.get_instance()
        result = model.ocr(pil_image)

        return OCROutput(
            image_id=input.image_id,
            text=result.text,
            model=result.model,
        )
