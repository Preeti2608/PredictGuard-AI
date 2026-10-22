import anyio
from fastapi import APIRouter, Depends, Request

from app.config import Settings, get_settings
from app.core.rate_limit import limiter
from app.schemas import (
    SensorInput,
    SensorPredictionResponse,
    ToolWearPredictionInput,
    WearPredictionResponse,
)
from app.services.model_registry import ModelRegistry, get_model_registry
from app.services.sensor_service import SensorService

router = APIRouter(tags=["Inference"])


def get_sensor_service(
    registry: ModelRegistry = Depends(get_model_registry),
    settings: Settings = Depends(get_settings),
) -> SensorService:
    return SensorService(registry=registry, settings=settings)


@router.post(
    "/v1/predict/sensors",
    response_model=SensorPredictionResponse,
    summary="Predict machine failure from telemetry (v1)",
)
@limiter.limit("60/minute")
async def predict_sensors_v1(
    request: Request,
    data: SensorInput,
    service: SensorService = Depends(get_sensor_service),
) -> SensorPredictionResponse:
    """
    Predict machine failure from operating telemetry using the trained scikit-learn pipeline.
    Returns failure probability, risk assessment tier, and feature contribution explanations.
    Executed in a threadpool worker to ensure non-blocking concurrent request handling.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    return await anyio.to_thread.run_sync(service.predict, data, request_id)


@router.post(
    "/predict/sensors",
    response_model=SensorPredictionResponse,
    include_in_schema=True,
    summary="Predict machine failure from telemetry (Legacy alias)",
)
@limiter.limit("60/minute")
async def predict_sensors_legacy(
    request: Request,
    data: SensorInput,
    service: SensorService = Depends(get_sensor_service),
) -> SensorPredictionResponse:
    """Backward-compatible endpoint alias pointing to /v1/predict/sensors."""
    return await predict_sensors_v1(request=request, data=data, service=service)


@router.post(
    "/v1/predict/wear",
    response_model=WearPredictionResponse,
    summary="Predict tool wear degradation time (v1)",
)
@limiter.limit("60/minute")
async def predict_tool_wear_v1(
    request: Request,
    data: ToolWearPredictionInput,
    service: SensorService = Depends(get_sensor_service),
) -> WearPredictionResponse:
    """
    Predict tool wear degradation (minutes) using multivariate regression on operating conditions.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    return await anyio.to_thread.run_sync(service.predict_wear, data, request_id)
