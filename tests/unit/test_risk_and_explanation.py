import pytest
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import RiskLevel, SensorInput
from app.services.image_service import evaluate_image_risk
from app.services.sensor_service import evaluate_sensor_risk, extract_feature_contributions


def test_image_risk_tiers():
    settings = get_settings()
    # High risk
    risk, action = evaluate_image_risk(0.85, settings)
    assert risk == RiskLevel.HIGH
    assert "Reject casting immediately" in action

    # Medium risk
    risk, action = evaluate_image_risk(0.45, settings)
    assert risk == RiskLevel.MEDIUM
    assert "secondary manual inspection" in action

    # Low risk
    risk, action = evaluate_image_risk(0.10, settings)
    assert risk == RiskLevel.LOW
    assert "meets standard" in action


def test_sensor_risk_tiers():
    settings = get_settings()
    risk, action = evaluate_sensor_risk(0.80, settings)
    assert risk == RiskLevel.HIGH
    assert "Critical failure" in action

    risk, action = evaluate_sensor_risk(0.50, settings)
    assert risk == RiskLevel.MEDIUM
    assert "Elevated risk" in action

    risk, action = evaluate_sensor_risk(0.05, settings)
    assert risk == RiskLevel.LOW
    assert "nominal bounds" in action


def test_feature_contributions():
    sample = {
        "Type": "L",
        "Air temperature [K]": 302.0,
        "Process temperature [K]": 311.0,
        "Rotational speed [rpm]": 1350.0,
        "Torque [Nm]": 68.0,
        "Tool wear [min]": 215.0,
    }
    factors = extract_feature_contributions(model=None, sample_dict=sample)
    assert len(factors) > 0
    # Factors should be ranked by severity
    assert any(f.severity == "HIGH" for f in factors)
    # Torque should have high deviation from ~40Nm
    torque_factor = next(f for f in factors if f.feature == "Torque [Nm]")
    assert torque_factor.value == 68.0
    assert torque_factor.deviation_percent > 50


def test_sensor_input_validation():
    # Valid input with bracketed aliases
    valid = SensorInput.model_validate(
        {
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 100.0,
        }
    )
    assert valid.type.value == "M"

    # Out of range air temperature
    with pytest.raises(ValidationError):
        SensorInput.model_validate(
            {
                "Type": "M",
                "Air temperature [K]": 500.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 100.0,
            }
        )

    # Reject NaN
    with pytest.raises(ValidationError):
        SensorInput.model_validate(
            {
                "Type": "M",
                "Air temperature [K]": float("nan"),
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 100.0,
            }
        )

    # Reject Inf
    with pytest.raises(ValidationError):
        SensorInput.model_validate(
            {
                "Type": "M",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": float("inf"),
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 100.0,
            }
        )
