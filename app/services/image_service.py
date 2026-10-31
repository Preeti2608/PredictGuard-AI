import io
import time
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
from starlette import status

from app.config import Settings
from app.core.errors import ImageValidationException, ModelNotLoadedException
from app.core.logging import get_logger
from app.schemas import ImagePredictionResponse, InspectionDomain, RiskLevel
from app.services.model_registry import ModelRegistry

logger = get_logger("predictguard.image_service")

# Magic bytes signatures
JPEG_SIGNATURE = b"\xff\xd8\xff"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# NEU Surface Defect Database classes
NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]

NEU_LABELS = {
    "crazing": "Crazing (Network of Micro-Cracks)",
    "inclusion": "Non-Metallic Inclusions",
    "patches": "Surface Oxide / Slag Patches",
    "pitted_surface": "Pitted Surface Cavitations",
    "rolled_in_scale": "Rolled-in Mill Scale",
    "scratches": "Mechanical Surface Scratches",
}


def detect_image_mime(header: bytes) -> Optional[str]:
    """Detect actual image file type by magic byte header inspection."""
    if header.startswith(JPEG_SIGNATURE):
        return "image/jpeg"
    if header.startswith(PNG_SIGNATURE):
        return "image/png"
    return None


def preprocess_image(
    source: Union[bytes, Image.Image],
    target_size: Tuple[int, int] = (224, 224),
    interpolation: str = "nearest",
    max_pixels: Optional[int] = 10_000_000,
) -> np.ndarray:
    """
    Unified canonical image preprocessing function used across inference and training scripts.
    Guarantees no train/serve preprocessing skew.

    Parameters:
      source: Raw image bytes or PIL.Image instance
      target_size: Target (width, height) tuple (default 224x224)
      interpolation: 'nearest', 'bilinear', or 'bicubic'
      max_pixels: Maximum allowed pixel count before raising decompression bomb guard
    """
    if max_pixels:
        Image.MAX_IMAGE_PIXELS = max_pixels

    if isinstance(source, bytes):
        if not source:
            raise ImageValidationException("Uploaded file is empty.", status_code=status.HTTP_400_BAD_REQUEST)
        try:
            image = Image.open(io.BytesIO(source))
        except Image.DecompressionBombError:
            raise ImageValidationException(
                "Image pixel dimensions exceed safety limits (decompression bomb guard).",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        except (UnidentifiedImageError, OSError) as exc:
            raise ImageValidationException(
                "Could not parse image file. Ensure it is a valid JPG or PNG.",
                status_code=status.HTTP_400_BAD_REQUEST,
                details=str(exc),
            )
    elif isinstance(source, Image.Image):
        image = source
    else:
        raise ImageValidationException("Invalid image input type provided.", status_code=status.HTTP_400_BAD_REQUEST)

    if max_pixels and (image.width * image.height > max_pixels):
        raise ImageValidationException(
            "Image pixel dimensions exceed safety limits (decompression bomb guard).",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    # Convert to RGB (handles RGBA, grayscale, palette)
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as exc:
        raise ImageValidationException(f"Failed to convert image mode to RGB: {exc}")

    # Map interpolation algorithm
    resample_map = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
    }
    resample_mode = resample_map.get(interpolation.lower(), Image.Resampling.NEAREST)

    # Resize image
    image = image.resize(target_size, resample=resample_mode)

    # Convert to normalized float32 array in [0, 1]
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    # Add batch dimension: (1, height, width, channels)
    return np.expand_dims(image_array, axis=0)


def evaluate_image_risk(defect_probability: float, settings: Settings, domain: str = "casting") -> Tuple[RiskLevel, str]:
    dom = str(domain).lower()
    if defect_probability >= settings.image_defect_high_threshold:
        if dom == "bearing":
            action = (
                "Critical defect detected on bearing raceway/rollers. Reject bearing immediately to prevent catastrophic "
                "spindle/motor seizure. Check lubricant for metallic debris and verify contact angle alignment."
            )
        elif dom in ("neu_surface", "neu"):
            action = (
                "Severe surface defect identified on hot-rolled steel strip. Reroute coil to scrap/slitting. "
                "Check work roll wear, descale water nozzle pressure, and verify roll cooling integrity."
            )
        else:
            action = "Reject casting immediately. Reroute part to scrap or remelting. Inspect mold integrity and molten pour temperature."
        return (RiskLevel.HIGH, action)
    elif defect_probability >= settings.image_defect_medium_threshold:
        if dom == "bearing":
            action = "Minor surface irregularity observed on bearing raceway. Flag for acoustic vibration spectrum and rotational runout check."
        elif dom in ("neu_surface", "neu"):
            action = "Superficial steel strip flaw detected. Flag for secondary optical inspection or surface skin-pass skinning."
        else:
            action = "Flag for secondary manual inspection. Verify surface roughness and check for subsurface voids."
        return (RiskLevel.MEDIUM, action)
    else:
        if dom == "bearing":
            action = "Bearing surface passes visual defect inspection criteria. Approved for spindle/gearbox assembly."
        elif dom in ("neu_surface", "neu"):
            action = "Steel strip surface meets cold-rolling and stamping specifications."
        else:
            action = "Casting meets standard surface quality criteria. Approve for downstream machining and assembly."
        return (RiskLevel.LOW, action)


class ImageService:
    def __init__(self, registry: ModelRegistry, settings: Settings):
        self.registry = registry
        self.settings = settings

    def validate_file(self, contents: bytes, client_content_type: Optional[str] = None) -> None:
        """Validate payload size, non-emptiness, and magic bytes format."""
        if not contents or len(contents) == 0:
            raise ImageValidationException("Uploaded file is empty.", status_code=status.HTTP_400_BAD_REQUEST)

        if len(contents) > self.settings.max_image_size_bytes:
            raise ImageValidationException(
                f"File size ({len(contents)} bytes) exceeds maximum limit of {self.settings.max_image_size_bytes} bytes.",
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )

        # Check real magic bytes
        real_mime = detect_image_mime(contents[:16])
        if real_mime is None:
            raise ImageValidationException(
                "Please upload a JPG or PNG image. The file signature did not match supported formats.",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

    def predict(
        self,
        contents: bytes,
        request_id: str,
        domain: Union[str, InspectionDomain] = InspectionDomain.CASTING,
    ) -> ImagePredictionResponse:
        """Synchronous CPU inference executed in threadpool worker with domain routing."""
        raw_dom = getattr(domain, "value", domain)
        dom = str(raw_dom).lower()
        if dom not in ("casting", "bearing", "neu_surface", "mvtec"):
            dom = "casting"
        inspection_domain = InspectionDomain(dom)

        self.validate_file(contents)

        t_start = time.perf_counter()

        # Shared canonical preprocessing
        tensor = preprocess_image(
            source=contents,
            target_size=(224, 224),
            interpolation=self.settings.image_interpolation,
            max_pixels=self.settings.max_image_pixels,
        )

        model = self.registry.get_image_model(domain=dom)

        if dom == "bearing":
            if model is None:
                raise ModelNotLoadedException("Bearing Defect Classifier")
            raw_pred = model.predict(tensor, verbose=0)
            score = float(raw_pred[0][0])
            latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            defect_prob = round(1.0 - score, 6)
            predicted_class = "ok_front" if score >= 0.5 else "def_front"
            label = "Normal Bearing" if score >= 0.5 else "Defective Bearing"
            confidence = round(score if score >= 0.5 else defect_prob, 6)
            risk_level, recommended_action = evaluate_image_risk(defect_prob, self.settings, domain="bearing")
            return ImagePredictionResponse(
                prediction=predicted_class,
                label=label,
                defect_probability=defect_prob,
                confidence=confidence,
                risk_level=risk_level,
                recommended_action=recommended_action,
                model_version="1.0.0-bearing",
                latency_ms=latency_ms,
                request_id=request_id,
                domain=inspection_domain,
                defect_classes=None,
                score=score,
                note="Bearing defect classifier score based on Figshare bearing defect dataset.",
            )

        elif dom == "neu_surface":
            if model is None:
                raise ModelNotLoadedException("NEU Surface Defect Classifier")
            raw_pred = model.predict(tensor, verbose=0)
            probs = raw_pred[0]
            if len(probs) >= len(NEU_CLASSES):
                class_idx = int(np.argmax(probs))
                pred_cls = NEU_CLASSES[class_idx]
                conf = float(probs[class_idx])
                classes_dict = {NEU_CLASSES[i]: round(float(probs[i]), 4) for i in range(len(NEU_CLASSES))}
            else:
                score = float(probs[0])
                pred_cls = "scratches" if score < 0.5 else "normal_surface"
                conf = round(float(abs(score - 0.5) * 2), 4)
                classes_dict = None

            latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            defect_prob = round(conf, 4)
            label = NEU_LABELS.get(pred_cls, pred_cls.replace("_", " ").title())
            risk_level, recommended_action = evaluate_image_risk(defect_prob, self.settings, domain="neu_surface")
            return ImagePredictionResponse(
                prediction=pred_cls,
                label=label,
                defect_probability=defect_prob,
                confidence=conf,
                risk_level=risk_level,
                recommended_action=recommended_action,
                model_version="1.0.0-neu",
                latency_ms=latency_ms,
                request_id=request_id,
                domain=inspection_domain,
                defect_classes=classes_dict,
                score=round(1.0 - defect_prob, 4),
                note="NEU 6-Class Surface Defect model evaluation.",
            )

        else:
            # Default: casting
            if model is None:
                raise ModelNotLoadedException("Casting Defect Classifier")
            raw_pred = model.predict(tensor, verbose=0)
            score = float(raw_pred[0][0])
            latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            defect_prob = round(1.0 - score, 6)
            predicted_class = "ok_front" if score >= 0.5 else "def_front"
            label = "Normal casting" if predicted_class == "ok_front" else "Defective casting"
            confidence = round(score if predicted_class == "ok_front" else defect_prob, 6)
            risk_level, recommended_action = evaluate_image_risk(defect_prob, self.settings, domain="casting")
            return ImagePredictionResponse(
                prediction=predicted_class,
                label=label,
                defect_probability=defect_prob,
                confidence=confidence,
                risk_level=risk_level,
                recommended_action=recommended_action,
                model_version="1.0.0",
                latency_ms=latency_ms,
                request_id=request_id,
                domain=inspection_domain,
                defect_classes=None,
                score=score,
                note="Score is the model output; verify its calibration before treating it as a probability.",
            )
