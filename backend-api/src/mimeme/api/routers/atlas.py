from __future__ import annotations

from fastapi import APIRouter, HTTPException

from mimeme import storage
from mimeme.api.auth import AdminRequired
from mimeme.api.deps import ArtifactStorageDep, DbDep, SettingsDep, UrlsDep
from mimeme.api.models.errors import error_responses
from mimeme.atlas.build import NoEmbeddings, build_template_atlas, load_template_atlas
from mimeme.atlas.model import TemplateAtlas, TemplateAtlasRunRequest

router = APIRouter(prefix="/atlas", tags=["Atlas"], responses=error_responses(403, 429, 500))


@router.get("/template", response_model=TemplateAtlas, responses=error_responses(404))
async def get_template_atlas(
    _auth: AdminRequired, artifacts: ArtifactStorageDep
) -> TemplateAtlas:
    try:
        atlas = await load_template_atlas(artifacts)
    except storage.Missing:
        atlas = None
    except storage.Error as exc:
        raise HTTPException(status_code=503, detail="Atlas storage is unavailable") from exc
    if atlas is None:
        raise HTTPException(status_code=404, detail="No template atlas has been generated")
    return atlas


@router.post("/template/run", response_model=TemplateAtlas, responses=error_responses(409, 503))
async def run_template_atlas(
    _auth: AdminRequired,
    db: DbDep,
    artifacts: ArtifactStorageDep,
    media_urls: UrlsDep,
    settings: SettingsDep,
    body: TemplateAtlasRunRequest,
) -> TemplateAtlas:
    try:
        return await build_template_atlas(
            db,
            artifacts,
            media_urls,
            model=settings.inference.embed_model,
            options=body,
        )
    except NoEmbeddings as exc:
        raise HTTPException(
            status_code=409,
            detail="No completed SigLIP2 image embeddings are available for this experiment",
        ) from exc
    except storage.Error as exc:
        raise HTTPException(status_code=503, detail="Atlas storage is unavailable") from exc
