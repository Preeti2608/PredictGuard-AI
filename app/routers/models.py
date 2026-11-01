from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas import DomainInfo, DomainsResponse, ModelsMetadataResponse
from app.services.model_registry import ModelRegistry, get_model_registry

router = APIRouter(prefix="/v1/models", tags=["Models"])
domains_router = APIRouter(prefix="/v1/domains", tags=["Domains"])


@router.get("", response_model=ModelsMetadataResponse, summary="Get models metadata and status")
def get_models_metadata(
    registry: ModelRegistry = Depends(get_model_registry),
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns registered machine learning models metadata (name, version,
    framework, dataset, input schemas, and validation metrics).
    """
    return {"models": registry.get_metadata()}


@domains_router.get("", response_model=DomainsResponse, summary="Get inspection domains status and licenses")
def get_domains(
    registry: ModelRegistry = Depends(get_model_registry),
    settings: Settings = Depends(get_settings),
) -> DomainsResponse:
    """Returns each inspection domain with trained status, dataset, license, classes, and notes."""
    domains_data = [
        DomainInfo(
            id="casting",
            name="Submersible Pump Impeller Casting",
            trained=registry.is_domain_trained("casting", settings.allow_placeholder_models),
            status="TRAINED" if registry.is_domain_trained("casting", settings.allow_placeholder_models) else "PENDING_EVALUATION",
            dataset="Casting Product Image Data (ravirajsinh45/real-life-industrial-dataset-of-casting-product)",
            license="CC0 Public Domain",
            classes=["ok_front", "def_front"],
            metrics_summary=None,
            note="Weights loaded from models/casting_defect_classifier.h5; evaluation via training/evaluate_image.py.",
        ),
        DomainInfo(
            id="neu_surface",
            name="Hot-Rolled Steel Strip Surface",
            trained=registry.is_domain_trained("neu", settings.allow_placeholder_models),
            status="TRAINED" if registry.is_domain_trained("neu", settings.allow_placeholder_models) else "UNTRAINED_PLACEHOLDER",
            dataset="NEU Surface Defect Database (kaustubhdikshit/neu-surface-defect-database)",
            license="Academic / Educational",
            classes=["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"],
            metrics_summary=None,
            note="Untrained architectural placeholder. Download dataset to data/neu_data/ and run training/train_neu.py.",
        ),
        DomainInfo(
            id="mvtec",
            name="MVTec Industrial Anomaly Detection",
            trained=False,
            status="BLOCKED_PENDING_DOWNLOAD",
            dataset="MVTec Anomaly Detection (MVTec AD)",
            license="CC BY-NC-SA 4.0 (Non-Commercial Academic & Research Use Only)",
            classes=["good", "anomaly"],
            metrics_summary=None,
            note="Requires terms acceptance at https://www.mvtec.com/research-teaching/datasets/mvtec-ad.",
        ),
        DomainInfo(
            id="bearing",
            name="Ball & Roller Bearing Inspection",
            trained=registry.is_domain_trained("bearing", settings.allow_placeholder_models),
            status="FUTURE_WORK",
            dataset="Bearing Defect Dataset (Figshare Kangning Li doi: 10.6084/m9.figshare.25335076)",
            license="CC BY 4.0",
            classes=["normal", "defective"],
            metrics_summary=None,
            note="Moved to Future Work pending verified ground-truth defect annotations (>200 samples required).",
        ),
    ]
    return DomainsResponse(domains=domains_data)

