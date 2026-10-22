/**
 * PredictGuard AI - Machine Health Telemetry Page (ES Module)
 */

import { api } from "../api.js";
import { renderSvgGauge } from "../components/gauge.js";
import { showToast } from "../components/toast.js";
import { addHistoryEntry, escapeHtml } from "../state.js";

let lastSensorResult = null;

export function initMachinePage() {
  const form = document.getElementById("sensor-form");
  const btnNominal = document.getElementById("preset-nominal-btn");
  const btnStress = document.getElementById("preset-stress-btn");
  const copyJsonBtn = document.getElementById("sensor-copy-json");

  if (!form) return;

  btnNominal?.addEventListener("click", () => {
    setFormValues({
      type: "M",
      air_temp: "300.0",
      process_temp: "310.0",
      speed: "1500.0",
      torque: "40.0",
      wear: "100.0",
    });
    showToast("Loaded nominal telemetry preset", "info");
  });

  btnStress?.addEventListener("click", () => {
    setFormValues({
      type: "L",
      air_temp: "302.0",
      process_temp: "311.0",
      speed: "1350.0",
      torque: "68.0",
      wear: "215.0",
    });
    showToast("Loaded high-stress failure preset", "info");
  });

  form.addEventListener("submit", handleSubmit);

  copyJsonBtn?.addEventListener("click", () => {
    if (!lastSensorResult) return;
    navigator.clipboard.writeText(JSON.stringify(lastSensorResult, null, 2))
      .then(() => showToast("Telemetry prediction JSON copied", "success"))
      .catch(() => showToast("Failed to copy JSON", "error"));
  });
}

function setFormValues(vals) {
  document.getElementById("field-type").value = vals.type;
  document.getElementById("field-air-temp").value = vals.air_temp;
  document.getElementById("field-process-temp").value = vals.process_temp;
  document.getElementById("field-rot-speed").value = vals.speed;
  document.getElementById("field-torque").value = vals.torque;
  document.getElementById("field-tool-wear").value = vals.wear;
  clearInlineErrors();
}

function clearInlineErrors() {
  document.querySelectorAll(".form-error").forEach((el) => {
    el.classList.remove("visible");
    el.textContent = "";
  });
  document.querySelectorAll(".form-control").forEach((el) => {
    el.classList.remove("is-invalid");
  });
}

function validateField(id, min, max, label) {
  const input = document.getElementById(id);
  const err = document.getElementById(`${id}-err`);
  const val = parseFloat(input.value);

  if (isNaN(val)) {
    input.classList.add("is-invalid");
    err.textContent = `Please enter a valid numeric value for ${label}.`;
    err.classList.add("visible");
    return false;
  }
  if (val < min || val > max) {
    input.classList.add("is-invalid");
    err.textContent = `${label} must be between ${min} and ${max}.`;
    err.classList.add("visible");
    return false;
  }
  return true;
}

async function handleSubmit(e) {
  e.preventDefault();
  clearInlineErrors();

  const type = document.getElementById("field-type").value;
  const validAir = validateField("field-air-temp", 280, 320, "Air temperature");
  const validProc = validateField("field-process-temp", 295, 330, "Process temperature");
  const validSpeed = validateField("field-rot-speed", 1000, 3000, "Rotational speed");
  const validTorque = validateField("field-torque", 0, 100, "Torque");
  const validWear = validateField("field-tool-wear", 0, 300, "Tool wear");

  if (!validAir || !validProc || !validSpeed || !validTorque || !validWear) {
    return;
  }

  const payload = {
    Type: type,
    "Air temperature [K]": parseFloat(document.getElementById("field-air-temp").value),
    "Process temperature [K]": parseFloat(document.getElementById("field-process-temp").value),
    "Rotational speed [rpm]": parseFloat(document.getElementById("field-rot-speed").value),
    "Torque [Nm]": parseFloat(document.getElementById("field-torque").value),
    "Tool wear [min]": parseFloat(document.getElementById("field-tool-wear").value),
  };

  const submitBtn = document.getElementById("sensor-submit-btn");
  const resultCard = document.getElementById("sensor-result-card");
  const gaugeSlot = document.getElementById("sensor-gauge-slot");
  const predElem = document.getElementById("sensor-result-pred");
  const badgeElem = document.getElementById("sensor-risk-badge");
  const actionElem = document.getElementById("sensor-recommended-action");
  const factorsContainer = document.getElementById("sensor-factors-container");
  const latencyElem = document.getElementById("sensor-latency");
  const reqIdElem = document.getElementById("sensor-request-id");

  try {
    submitBtn.disabled = true;
    submitBtn.textContent = "Forecasting Machine Failure...";

    const res = await api.predictSensors(payload);
    lastSensorResult = res;

    // Render gauge
    gaugeSlot.innerHTML = renderSvgGauge(res.failure_probability, res.risk_level, "Failure Probability");

    // Metadata
    predElem.textContent = res.prediction === "Failure" ? "Machine Failure Imminent" : "Machine Operating Nominally";
    badgeElem.textContent = `${res.risk_level} RISK`;
    badgeElem.className = `badge badge-${res.risk_level.toLowerCase()}`;
    actionElem.textContent = res.recommended_action;
    latencyElem.textContent = `${res.latency_ms} ms`;
    reqIdElem.textContent = res.request_id;

    // Contributing factors horizontal bars
    if (res.top_contributing_factors && res.top_contributing_factors.length > 0) {
      factorsContainer.innerHTML = res.top_contributing_factors
        .map((f) => {
          let barColor = "var(--status-success-dot)";
          if (f.severity === "HIGH") barColor = "var(--status-danger-dot)";
          else if (f.severity === "MEDIUM") barColor = "var(--status-warning-dot)";

          const fillPct = Math.min(100, Math.max(10, Math.abs(f.deviation_percent) * 1.5));
          const devSign = f.deviation_percent > 0 ? "+" : "";

          return `
            <div class="factor-item">
              <div class="factor-header">
                <span class="factor-name">${escapeHtml(f.label)}: <strong>${f.value}</strong> (baseline: ${f.nominal})</span>
                <span class="factor-metric">${devSign}${f.deviation_percent}% deviation</span>
              </div>
              <div class="factor-bar-track">
                <div class="factor-bar-fill" style="width: ${fillPct}%; background-color: ${barColor};"></div>
              </div>
            </div>
          `;
        })
        .join("");
    } else {
      factorsContainer.innerHTML = `<div style="font-size: var(--text-xs); color: var(--text-tertiary);">No abnormal parameter deviations detected.</div>`;
    }

    resultCard.style.display = "block";
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });

    // Store in history
    addHistoryEntry({
      type: "sensor",
      label: "Machine Telemetry",
      prediction: res.prediction,
      probability: res.failure_probability,
      risk_level: res.risk_level,
      latency_ms: res.latency_ms,
    });

    showToast("Telemetry forecast complete", "success");
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Forecast Machine Failure";
  }
}
