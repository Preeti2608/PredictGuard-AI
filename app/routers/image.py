import anyio
from fastapi import APIRouter, Depends, File, Request, UploadFile

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
    summary="Classify casting surface defect (v1)",
)
@limiter.limit("60/minute")
async def predict_image_v1(
    request: Request,
    file: UploadFile = File(..., description="Casting top surface JPEG or PNG image"),
    service: ImageService = Depends(get_image_service),
) -> ImagePredictionResponse:
    """
    Classify a casting surface image into Normal ('ok_front') or Defective ('def_front')
    using the production Convolutional Neural Network.
    Inference executes inside an asynchronous threadpool worker to avoid blocking the event loop.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    contents = await file.read()
    return await anyio.to_thread.run_sync(service.predict, contents, request_id)


@router.post(
    "/predict/image",
    response_model=ImagePredictionResponse,
    include_in_schema=True,
    summary="Classify casting surface defect (Legacy alias)",
)
@limiter.limit("60/minute")
async def predict_image_legacy(
    request: Request,
    file: UploadFile = File(...),
    service: ImageService = Depends(get_image_service),
) -> ImagePredictionResponse:
    """Backward-compatible endpoint alias pointing to /v1/predict/image."""
    return await predict_image_v1(request=request, file=file, service=service)
