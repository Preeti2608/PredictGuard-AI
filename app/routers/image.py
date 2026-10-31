from typing import Optional

import anyio
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.config import Settings, get_settings
from app.core.rate_limit import limiter
from app.schemas import ImagePredictionResponse
from app.services.image_service import ImageService
from app.services.model_registry import ModelRegistry, get_model_registry

router = APIRouter(tags=["Inference"])


def get_image_service(
    registry: ModelRegistry = Depends(get_model_registry),
    settings: Settings = Depends(get_settings),
) -> ImageService:
    return ImageService(registry=registry, settings=settings)


@router.post(
    "/v1/predict/image",
    response_model=ImagePredictionResponse,
    summary="Classify industrial component surface defect (v1)",
)
@limiter.limit("60/minute")
async def predict_image_v1(
    request: Request,
    file: UploadFile = File(..., description="Industrial component JPEG or PNG image"),
    domain: Optional[str] = Form(None, description="Inspection domain: 'casting', 'bearing', 'neu_surface', or 'mvtec'"),
    service: ImageService = Depends(get_image_service),
) -> ImagePredictionResponse:
    """
    Classify an industrial surface image using the appropriate domain model.
    Supported domains: 'casting' (pump impeller), 'bearing' (ball/roller bearings), 'neu_surface' (steel strip).
    Inference executes inside an asynchronous threadpool worker to avoid blocking the event loop.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    active_domain = domain or request.query_params.get("domain") or "casting"
    contents = await file.read()
    return await anyio.to_thread.run_sync(service.predict, contents, request_id, active_domain)


@router.post(
    "/predict/image",
    response_model=ImagePredictionResponse,
    include_in_schema=True,
    summary="Classify surface defect (Legacy alias)",
)
@limiter.limit("60/minute")
async def predict_image_legacy(
    request: Request,
    file: UploadFile = File(...),
    domain: Optional[str] = Form(None),
    service: ImageService = Depends(get_image_service),
) -> ImagePredictionResponse:
    """Backward-compatible endpoint alias pointing to /v1/predict/image."""
    return await predict_image_v1(request=request, file=file, domain=domain, service=service)
