from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

from domain.inference import (
    build_image_embedding_key,
    build_text_embedding_key_for_image_embedding,
    prepare_rgb_image_for_inference,
)
from PIL import Image as PILImage
from shared.services import StorageService, get_storage_service

from activities.embedding.models import (
    EmbedBatchInput,
    EmbedBatchOutput,
    EmbedImageOutput,
)
from activities.embedding.siglip import SiglipEmbedder
from activities.vision.models import AnnotateImageInput, AnnotateImageOutput
from activities.vision.moondream import Moondream2


class LocalGpuBackend:
    async def annotate_image(self, input: AnnotateImageInput) -> AnnotateImageOutput:
        storage = cast(StorageService, get_storage_service())
        model = Moondream2.get_instance()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            tmp_path = Path(tmp.name)
            storage.download_file(input.s3_key, tmp_path)
            pil_image = prepare_rgb_image_for_inference(PILImage.open(tmp_path))
            result = model.annotate_image(pil_image, length=input.length)
            return AnnotateImageOutput(
                image_id=input.image_id,
                caption=result.caption,
                caption_model=result.caption_model,
                ocr_text=result.ocr_text,
                ocr_model=result.ocr_model,
            )

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

                    pil_image = prepare_rgb_image_for_inference(PILImage.open(tmp_path))

                    img_feats = embedder.encode_images([pil_image])
                    txt_feats = embedder.encode_texts([item.text])

                    dataset = item.dataset or input.dataset
                    img_embed_key = build_image_embedding_key(
                        sha256=item.sha256,
                        model_name=embedder.image_model_name,
                        dataset=dataset,
                    )
                    text_embed_key = build_text_embedding_key_for_image_embedding(img_embed_key)

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
