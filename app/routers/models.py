from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from app.schemas import ModelsMetadataResponse
from app.services.model_registry import ModelRegistry, get_model_registry

router = APIRouter(prefix="/v1/models", tags=["Models"])


@router.get("", response_model=ModelsMetadataResponse, summary="Get models metadata and status")
def get_models_metadata(
    registry: ModelRegistry = Depends(get_model_registry),
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns registered machine learning models metadata (name, version,
    framework, dataset, input schemas, and validation metrics).
    """
    return {"models": registry.get_metadata()}
