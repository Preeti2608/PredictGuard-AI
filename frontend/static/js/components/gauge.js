/**
 * PredictGuard AI - Accessible Inline SVG Probability Gauge Component
 */

import { escapeHtml } from "../state.js";

export function renderSvgGauge(probability, riskLevel, titleText = "Probability") {
  const pct = Math.max(0, Math.min(100, Math.round(probability * 1000) / 10));
  const r = 60;
  const strokeWidth = 10;
  const totalLength = Math.PI * r; // ~ 188.5
  const fillLength = (pct / 100) * totalLength;
  const dashOffset = totalLength - fillLength;

  let color = "var(--status-success-dot)";
  if (riskLevel === "HIGH") {
    color = "var(--status-danger-dot)";
  } else if (riskLevel === "MEDIUM") {
    color = "var(--status-warning-dot)";
  }

  return `
    <div class="gauge-container" role="meter" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="${escapeHtml(titleText)}: ${pct}%">
      <svg class="gauge-svg" viewBox="0 0 160 95">
        <!-- Background Semi-Circle Arc -->
        <path d="M 20 80 A 60 60 0 0 1 140 80" class="gauge-bg-arc" />
        <!-- Filled Active Arc -->
        <path d="M 20 80 A 60 60 0 0 1 140 80" class="gauge-fill-arc"
          style="stroke: ${color}; stroke-dasharray: ${totalLength}; stroke-dashoffset: ${dashOffset};" />
        <!-- Value Text -->
        <text x="80" y="65" class="gauge-value-text">${pct}%</text>
        <!-- Label Subtext -->
        <text x="80" y="82" class="gauge-label-text">${escapeHtml(titleText)}</text>
      </svg>
    </div>
  `;
}
