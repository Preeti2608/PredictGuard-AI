import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from starlette import status

from app.config import Settings
from app.core.errors import AppException, ModelNotLoadedException
from app.core.logging import get_logger
from app.schemas import (
    FeatureContribution,
    RiskLevel,
    SensorInput,
    SensorPredictionResponse,
    ToolWearPredictionInput,
    WearPredictionResponse,
)
from app.services.model_registry import ModelRegistry

logger = get_logger("predictguard.sensor_service")

# Nominal baseline medians / means from AI4I 2020 dataset
NOMINAL_BASELINES = {
    "Torque [Nm]": 39.9,
    "Rotational speed [rpm]": 1495.0,
    "Tool wear [min]": 107.0,
    "Air temperature [K]": 300.1,
    "Process temperature [K]": 310.0,
}

FEATURE_DISPLAY_NAMES = {
    "Torque [Nm]": "Torque",
    "Rotational speed [rpm]": "Rotational Speed",
    "Tool wear [min]": "Tool Wear",
    "Air temperature [K]": "Air Temperature",
    "Process temperature [K]": "Process Temperature",
}


def evaluate_sensor_risk(failure_probability: float, settings: Settings) -> Tuple[RiskLevel, str]:
    if failure_probability >= settings.sensor_failure_high_threshold:
        return (
            RiskLevel.HIGH,
            "Critical failure imminent. Initiate emergency shutdown protocol and inspect drivetrain, spindle bearings, and tool head.",
        )
    elif failure_probability >= settings.sensor_failure_medium_threshold:
        return (
            RiskLevel.MEDIUM,
            "Elevated risk detected. Schedule maintenance window within next operational shift and reduce machine feed/speed load.",
        )
    return (
        RiskLevel.LOW,
        "Telemetry is within nominal bounds. Continue standard production cycle.",
    )


def extract_feature_contributions(
    model: Any,
    sample_dict: Dict[str, Any],
) -> List[FeatureContribution]:
    """
    Calculate top contributing factors by combining trained decision tree feature importances
    with standardized deviation from nominal operating baselines.
    """
    # Extract model importances if available
    importance_map: Dict[str, float] = {
        "Torque [Nm]": 0.446,
        "Air temperature [K]": 0.195,
        "Tool wear [min]": 0.122,
        "Rotational speed [rpm]": 0.113,
        "Process temperature [K]": 0.104,
    }

    try:
        if hasattr(model, "named_steps") and "classifier" in model.named_steps:
            clf = model.named_steps["classifier"]
            prep = model.named_steps["preprocessor"]
            if hasattr(clf, "feature_importances_") and hasattr(prep, "get_feature_names_out"):
                out_names = list(prep.get_feature_names_out())
                importances = clf.feature_importances_
                for f_name, imp in zip(out_names, importances):
                    clean_name = f_name.replace("num__", "")
                    if clean_name in importance_map:
                        importance_map[clean_name] = float(imp)
    except Exception as exc:
        logger.debug("Using default importance map: %s", exc)

    factors: List[FeatureContribution] = []

    for feature_key, baseline in NOMINAL_BASELINES.items():
        val = float(sample_dict.get(feature_key, baseline))
        deviation_pct = round(((val - baseline) / baseline) * 100.0, 1)
        importance = round(importance_map.get(feature_key, 0.1), 3)

        # Severity classification
        abs_dev = abs(deviation_pct)
        if (
            abs_dev > 30
            or (feature_key == "Torque [Nm]" and val > 60)
            or (feature_key == "Tool wear [min]" and val > 200)
        ):
            severity = "HIGH"
        elif abs_dev > 15:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        factors.append(
            FeatureContribution(
                feature=feature_key,
                label=FEATURE_DISPLAY_NAMES.get(feature_key, feature_key),
                value=round(val, 2),
                nominal=round(baseline, 2),
                deviation_percent=deviation_pct,
                importance=importance,
                severity=severity,
            )
        )

    # Sort factors by severity and importance descending
    severity_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    factors.sort(key=lambda x: (severity_order.get(x.severity, 0), x.importance), reverse=True)
    return factors


class SensorService:
    def __init__(self, registry: ModelRegistry, settings: Settings):
        self.registry = registry
        self.settings = settings

    def predict(self, input_data: SensorInput, request_id: str) -> SensorPredictionResponse:
        """Run telemetry failure prediction synchronously in threadpool."""
        model = self.registry.get_sensor_model()
        if model is None:
            raise ModelNotLoadedException("Machine Failure Predictor")

        t_start = time.perf_counter()

        sample_dict = {
            "Type": input_data.type.value,
            "Air temperature [K]": input_data.air_temperature,
            "Process temperature [K]": input_data.process_temperature,
            "Rotational speed [rpm]": input_data.rotational_speed,
            "Torque [Nm]": input_data.torque,
            "Tool wear [min]": input_data.tool_wear,
        }

        try:
            sample_df = pd.DataFrame([sample_dict])
            raw_prediction = int(model.predict(sample_df)[0])
        except Exception as exc:
            logger.error("Failed to run prediction on sensor dataframe: %s", exc)
            raise AppException(
                message="Could not predict sensor data. Verify input values.",
                error_code="INFERENCE_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST,
                details=str(exc),
            )

        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        prediction_str = "Failure" if raw_prediction == 1 else "No Failure"

        class_probabilities: Optional[Dict[str, float]] = None
        failure_prob: float = 1.0 if raw_prediction == 1 else 0.0

        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(sample_df)[0]
                classes = list(model.classes_)
                class_probabilities = {str(cls_label): round(float(p), 6) for cls_label, p in zip(classes, probs)}
                # Failure class is 1
                if 1 in classes:
                    idx = classes.index(1)
                    failure_prob = round(float(probs[idx]), 6)
                elif "1" in classes:
                    idx = classes.index("1")
                    failure_prob = round(float(probs[idx]), 6)
            except Exception as exc:
                logger.warning("predict_proba failed: %s", exc)

        risk_level, action = evaluate_sensor_risk(failure_prob, self.settings)
        factors = extract_feature_contributions(model, sample_dict)

        return SensorPredictionResponse(
            prediction=prediction_str,
            failure_probability=failure_prob,
            risk_level=risk_level,
            recommended_action=action,
            top_contributing_factors=factors,
            model_version="1.0.0",
            latency_ms=latency_ms,
            request_id=request_id,
            prediction_code=raw_prediction,
            class_probabilities=class_probabilities,
        )

    def predict_wear(self, input_data: ToolWearPredictionInput, request_id: str) -> WearPredictionResponse:
        """Run tool wear continuous degradation regression."""
        model = self.registry.get_wear_regressor()
        if model is None:
            raise ModelNotLoadedException("Tool Wear Regressor")

        t_start = time.perf_counter()
        sample_dict = {
            "Type": input_data.type.value,
            "Air temperature [K]": input_data.air_temperature,
            "Process temperature [K]": input_data.process_temperature,
            "Rotational speed [rpm]": input_data.rotational_speed,
            "Torque [Nm]": input_data.torque,
        }

        try:
            sample_df = pd.DataFrame([sample_dict])
            pred_wear = float(model.predict(sample_df)[0])
        except Exception as exc:
            logger.error("Failed to predict tool wear: %s", exc)
            raise AppException(
                message="Tool wear regression failed. Verify input values.",
                error_code="INFERENCE_FAILED",
                status_code=status.HTTP_400_BAD_REQUEST,
                details=str(exc),
            )

        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        pred_wear_clamped = max(0.0, round(pred_wear, 1))

        if pred_wear_clamped > 200.0:
            risk = RiskLevel.HIGH
            action = "Tool wear critical (>200 min). Replace insert before starting next machining sequence."
        elif pred_wear_clamped > 120.0:
            risk = RiskLevel.MEDIUM
            action = "Moderate wear detected (>120 min). Inspect cutting edge for chipping during next cycle stop."
        else:
            risk = RiskLevel.LOW
            action = "Tool wear within acceptable baseline. Continue operation."

        return WearPredictionResponse(
            predicted_tool_wear_min=pred_wear_clamped,
            risk_level=risk,
            recommended_action=action,
            model_version="1.0.0",
            latency_ms=latency_ms,
            request_id=request_id,
        )
