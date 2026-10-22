import math
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ProductType(str, Enum):
    L = "L"
    M = "M"
    H = "H"


class SensorInput(BaseModel):
    """
    Operating telemetry schema supporting both snake_case and original bracketed aliases.
    Strictly validates physical ranges from AI4I dataset and rejects NaN/Inf.
    """

    type: ProductType = Field(..., alias="Type", description="Product type variant: L (Low), M (Medium), H (High)")
    air_temperature: float = Field(
        ...,
        alias="Air temperature [K]",
        description="Air temperature in Kelvin [290.0 - 315.0]",
        ge=280.0,
        le=320.0,
    )
    process_temperature: float = Field(
        ...,
        alias="Process temperature [K]",
        description="Process temperature in Kelvin [295.0 - 330.0]",
        ge=295.0,
        le=330.0,
    )
    rotational_speed: float = Field(
        ...,
        alias="Rotational speed [rpm]",
        description="Rotational speed in RPM [1000.0 - 3000.0]",
        ge=1000.0,
        le=3000.0,
    )
    torque: float = Field(
        ...,
        alias="Torque [Nm]",
        description="Torque in Newton-meters [0.0 - 100.0]",
        ge=0.0,
        le=100.0,
    )
    tool_wear: float = Field(
        ...,
        alias="Tool wear [min]",
        description="Tool wear time in minutes [0.0 - 300.0]",
        ge=0.0,
        le=300.0,
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "Type": "M",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 100.0,
            }
        },
    }

    @field_validator("air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear")
    @classmethod
    def check_finite(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Numeric values must be finite numbers (NaN and Inf are rejected)")
        return v


class ToolWearPredictionInput(BaseModel):
    """Input for tool wear degradation regression prediction."""

    type: ProductType = Field(..., alias="Type", description="Product type variant: L, M, H")
    air_temperature: float = Field(..., alias="Air temperature [K]", ge=280.0, le=320.0)
    process_temperature: float = Field(..., alias="Process temperature [K]", ge=295.0, le=330.0)
    rotational_speed: float = Field(..., alias="Rotational speed [rpm]", ge=1000.0, le=3000.0)
    torque: float = Field(..., alias="Torque [Nm]", ge=0.0, le=100.0)

    model_config = {"populate_by_name": True}

    @field_validator("air_temperature", "process_temperature", "rotational_speed", "torque")
    @classmethod
    def check_finite(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Numeric values must be finite numbers (NaN and Inf are rejected)")
        return v


class ImagePredictionResponse(BaseModel):
    prediction: str = Field(description="'ok_front' or 'def_front'")
    label: str = Field(description="'Normal casting' or 'Defective casting'")
    defect_probability: float = Field(description="Calibrated probability of defect (1 - sigmoid_score)")
    confidence: float = Field(description="Confidence percentage (0.0 - 1.0)")
    risk_level: RiskLevel = Field(description="Risk assessment category (LOW, MEDIUM, HIGH)")
    recommended_action: str = Field(description="Recommended maintenance or inspection procedure")
    model_version: str = Field(description="Identifier and version of inference model")
    latency_ms: float = Field(description="Inference execution latency in milliseconds")
    request_id: str = Field(description="Traceable unique request identifier")

    # Backward compatibility fields
    score: float = Field(description="Raw sigmoid output (P(normal))")
    note: str = Field(default="Score is the model output; verify its calibration before treating it as a probability.")


class FeatureContribution(BaseModel):
    feature: str
    label: str
    value: float
    nominal: float
    deviation_percent: float
    importance: float
    severity: str


class SensorPredictionResponse(BaseModel):
    prediction: str = Field(description="'Failure' or 'No Failure'")
    failure_probability: float = Field(description="Probability of machine failure")
    risk_level: RiskLevel = Field(description="Risk assessment category (LOW, MEDIUM, HIGH)")
    recommended_action: str = Field(description="Operator advisory action")
    top_contributing_factors: List[FeatureContribution] = Field(
        default_factory=list,
        description="Ranked features deviating from nominal baseline or having high decision weight",
    )
    model_version: str = Field(description="Identifier and version of inference model")
    latency_ms: float = Field(description="Inference execution latency in milliseconds")
    request_id: str = Field(description="Traceable unique request identifier")

    # Backward compatibility fields
    prediction_code: int = Field(description="0 for No Failure, 1 for Failure")
    class_probabilities: Optional[Dict[str, float]] = Field(
        default=None, description="Probability map for each class label"
    )


class WearPredictionResponse(BaseModel):
    predicted_tool_wear_min: float = Field(description="Estimated tool wear in minutes")
    risk_level: RiskLevel = Field(description="Degradation risk tier")
    recommended_action: str = Field(description="Tool maintenance or replacement advisory")
    model_version: str = Field(description="Model identifier")
    latency_ms: float = Field(description="Inference latency in milliseconds")
    request_id: str = Field(description="Traceable unique request identifier")


class HealthLiveResponse(BaseModel):
    status: str = "live"
    timestamp: str


class HealthReadyResponse(BaseModel):
    status: str
    image_model_loaded: bool
    sensor_model_loaded: bool
    wear_model_loaded: bool
    timestamp: str


class LegacyHealthResponse(BaseModel):
    status: str
    image_model_loaded: bool
    sensor_model_loaded: bool
    version: str


class ModelInfo(BaseModel):
    name: str
    version: str
    framework: str
    task: str
    training_date: Optional[str] = None
    dataset: str
    input_shape_or_features: Any
    metrics: Dict[str, Any]
    status: str
    file_path: str


class ModelsMetadataResponse(BaseModel):
    models: List[ModelInfo]
