from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

from PIL import Image as PILImage

from activities.embedding.models import (
    EmbedBatchInput,
    EmbedBatchOutput,
    EmbedImageOutput,
    EncodeQueryInput,
    EncodeQueryOutput,
)
from activities.embedding.siglip import SiglipEmbedder
from activities.vision.models import CaptionInput, CaptionOutput, OCRInput, OCROutput
from activities.vision.moondream import Moondream2
from shared.services import StorageService, get_storage_service


class LocalGpuBackend:
    async def caption(self, input: CaptionInput) -> CaptionOutput:
        storage = cast(StorageService, get_storage_service())
        model = Moondream2.get_instance()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            tmp_path = Path(tmp.name)
            storage.download_file(input.s3_key, tmp_path)
            pil_image = PILImage.open(tmp_path).convert("RGB")
            result = model.caption(pil_image, length=input.length)
            return CaptionOutput(
                image_id=input.image_id, caption=result.caption, model=result.model
            )

    async def ocr(self, input: OCRInput) -> OCROutput:
        storage = cast(StorageService, get_storage_service())
        model = Moondream2.get_instance()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            tmp_path = Path(tmp.name)
            storage.download_file(input.s3_key, tmp_path)
            pil_image = PILImage.open(tmp_path).convert("RGB")
            result = model.ocr(pil_image)
            return OCROutput(image_id=input.image_id, text=result.text, model=result.model)

    async def embed_batch(self, input: EmbedBatchInput) -> EmbedBatchOutput:
        storage = cast(StorageService, get_storage_service())
        embedder = SiglipEmbedder.get_instance()

        results: list[EmbedImageOutput] = []
        failed_ids: list[int] = []

        for item in input.items:
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                    tmp_path = Path(tmp.name)
                    storage.download_file(item.s3_key, tmp_path)

                    pil_image = PILImage.open(tmp_path)
                    if pil_image.mode == "P" and "transparency" in pil_image.info:
                        pil_image = pil_image.convert("RGBA")
                    pil_image = pil_image.convert("RGB")

                    img_feats = embedder.encode_images([pil_image])
                    txt_feats = embedder.encode_texts([item.text])

                    dataset = item.dataset or input.dataset
                    img_embed_key = storage.build_embedding_key(
                        sha256=item.sha256,
                        model_name=embedder.image_model_name,
                        dataset=dataset,
                    )
                    text_embed_key = img_embed_key.replace(".npy", "_text.npy")

                    storage.upload_numpy(img_feats[0], img_embed_key)
                    storage.upload_numpy(txt_feats[0], text_embed_key)

                    results.append(
                        EmbedImageOutput(
                            image_id=item.image_id,
                            image_embedding_key=img_embed_key,
                            text_embedding_key=text_embed_key,
                            model=embedder.image_model_name,
                            dimension=int(img_feats.shape[-1]),
                        )
                    )
            except Exception:
                failed_ids.append(item.image_id)

        return EmbedBatchOutput(results=results, failed_ids=failed_ids)

    async def encode_query(self, input: EncodeQueryInput) -> EncodeQueryOutput:
        embedder = SiglipEmbedder.get_instance()
        feats = embedder.encode_texts([input.query])
        return EncodeQueryOutput(
            embedding=feats[0].tolist(),
            model=embedder.image_model_name,
            dimension=int(feats.shape[-1]),
        )
