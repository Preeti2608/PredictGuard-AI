/**
 * PredictGuard AI - Overview Dashboard Page
 */

import { getHistory, escapeHtml } from "../state.js";
import { api } from "../api.js";

export function initOverviewPage() {
  renderOverviewKPIs();
  checkSystemStatus();
}

export async function checkSystemStatus() {
  const panel = document.getElementById("system-status-panel");
  if (!panel) return;

  try {
    const health = await api.checkHealth();
    const isReady = health.ok && health.data?.status === "ready";
    panel.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
          <div style="font-weight: var(--font-weight-semibold); color: var(--text-primary);">
            Production Cluster Status: ${isReady ? "Ready & Operational" : "Degraded / Not Ready"}
          </div>
          <div style="font-size: var(--text-xs); color: var(--text-tertiary); margin-top: 4px;">
            Inference engine active. TensorFlow CPU and scikit-learn pipelines ready.
          </div>
        </div>
        <span class="badge ${isReady ? "badge-low" : "badge-high"}">
          ${isReady ? "ALL SYSTEMS NOMINAL" : "ACTION REQUIRED"}
        </span>
      </div>
    `;
  } catch (err) {
    panel.innerHTML = `
      <div style="color: var(--status-danger-text); font-size: var(--text-sm);">
        Failed to connect to backend health check.
      </div>
    `;
  }
}

export function renderOverviewKPIs() {
  const history = getHistory();
  const kpiToday = document.getElementById("kpi-predictions-today");
  const kpiDefectRate = document.getElementById("kpi-defect-rate");
  const kpiHighRisk = document.getElementById("kpi-high-risk");
  const kpiAvgLatency = document.getElementById("kpi-avg-latency");
  const recentTable = document.getElementById("recent-activity-tbody");

  if (!kpiToday) return;

  if (!history || history.length === 0) {
    kpiToday.textContent = "0";
    kpiDefectRate.textContent = "--";
    kpiHighRisk.textContent = "0";
    kpiAvgLatency.textContent = "--";
    if (recentTable) {
      recentTable.innerHTML = `
        <tr>
          <td colspan="5" class="empty-state">
            <div class="empty-state-title">No predictions recorded yet</div>
            <div class="empty-state-desc">Run a Casting Inspection or Machine Health analysis to populate real-time analytics.</div>
          </td>
        </tr>
      `;
    }
    return;
  }

  // Calculate real metrics from history
  const todayStr = new Date().toISOString().slice(0, 10);
  const todayEntries = history.filter((h) => h.timestamp && h.timestamp.startsWith(todayStr));
  kpiToday.textContent = String(todayEntries.length);

  const imageEntries = history.filter((h) => h.type === "image");
  if (imageEntries.length > 0) {
    const defects = imageEntries.filter((h) => h.prediction === "def_front").length;
    kpiDefectRate.textContent = `${Math.round((defects / imageEntries.length) * 100)}%`;
  } else {
    kpiDefectRate.textContent = "--";
  }

  const sensorEntries = history.filter((h) => h.type === "sensor");
  const highRiskSensors = sensorEntries.filter((h) => h.risk_level === "HIGH").length;
  kpiHighRisk.textContent = String(highRiskSensors);

  const totalLatency = history.reduce((acc, h) => acc + (Number(h.latency_ms) || 0), 0);
  const avg = Math.round(totalLatency / history.length);
  kpiAvgLatency.textContent = `${avg} ms`;

  // Render recent 5 items
  if (recentTable) {
    const recent = history.slice(0, 5);
    recentTable.innerHTML = recent
      .map(
        (item) => `
        <tr>
          <td style="font-family: var(--font-mono); font-size: var(--text-xs);">${escapeHtml(item.id)}</td>
          <td>${item.type === "image" ? "Casting Inspection" : "Machine Telemetry"}</td>
          <td><strong>${escapeHtml(item.prediction)}</strong></td>
          <td>
            <span class="badge badge-${escapeHtml(item.risk_level.toLowerCase())}">
              ${escapeHtml(item.risk_level)}
            </span>
          </td>
          <td style="font-family: var(--font-mono); font-size: var(--text-xs);">${escapeHtml(item.latency_ms)} ms</td>
        </tr>
      `
      )
      .join("");
  }
}
