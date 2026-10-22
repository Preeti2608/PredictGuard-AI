from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Response, status

from app.config import Settings, get_settings
from app.schemas import HealthLiveResponse, HealthReadyResponse, LegacyHealthResponse
from app.services.model_registry import ModelRegistry, get_model_registry

router = APIRouter(tags=["Health"])


@router.get("/health/live", response_model=HealthLiveResponse, summary="Liveness probe")
def health_live() -> HealthLiveResponse:
    """Always returns 200 OK while the server process is alive."""
    return HealthLiveResponse(
        status="live",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/health/ready", response_model=HealthReadyResponse, summary="Readiness probe")
def health_ready(
    response: Response,
    registry: ModelRegistry = Depends(get_model_registry),
) -> HealthReadyResponse:
    """
    Readiness probe for container orchestrators (e.g. Docker HEALTHCHECK, Kubernetes).
    Returns 200 OK only if all required machine learning models are loaded.
    Returns 503 SERVICE UNAVAILABLE if any required model is missing.
    """
    ready = registry.is_ready()
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthReadyResponse(
        status="ready" if ready else "not_ready",
        image_model_loaded=registry.loaded_status.get("image_model", False),
        sensor_model_loaded=registry.loaded_status.get("sensor_model", False),
        wear_model_loaded=registry.loaded_status.get("wear_model", False),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/api/health", response_model=LegacyHealthResponse, summary="Legacy health check alias")
def legacy_health(
    registry: ModelRegistry = Depends(get_model_registry),
    settings: Settings = Depends(get_settings),
) -> LegacyHealthResponse:
    """Backward-compatible health endpoint preserving the exact original JSON contract."""
    ready = registry.is_ready()
    return LegacyHealthResponse(
        status="online" if ready else "degraded",
        image_model_loaded=registry.loaded_status.get("image_model", False),
        sensor_model_loaded=registry.loaded_status.get("sensor_model", False),
        version=settings.app_version,
    )
