from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

from PIL import Image as PILImage
from temporalio import activity

from activities.embeddding.models import EmbedBatchInput, EmbedBatchOutput, EmbedImageOutput
from activities.embeddding.siglip import SiglipEmbedder
from shared.db import session_scope
from shared.models import ORMImage
from shared.services import get_storage_service
from shared.services.storage import StorageService


@activity.defn
async def embed_batch_activity(input: EmbedBatchInput) -> EmbedBatchOutput:
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

                with session_scope() as session:
                    img = session.query(ORMImage).filter_by(id=item.image_id).first()

                    if not img:
                        raise ValueError(f"Image with id {item.image_id} not found in DB")

                    sha256 = img.sha256 if img else str(item.image_id)

                    dataset = img.dataset if img else input.dataset

                img_embed_key = storage.build_embedding_key(
                    sha256=sha256,
                    model_name=embedder.image_model_name,
                    dataset=dataset,
                )

                text_embed_key = img_embed_key.replace(".npy", "_text.npy")

                # todo: upload both embeddings in a single call concurrently
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
