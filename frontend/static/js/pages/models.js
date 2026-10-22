/**
 * PredictGuard AI - Models & Metrics Page (ES Module)
 */

import { api } from "../api.js";
import { escapeHtml } from "../state.js";

export async function initModelsPage() {
  const container = document.getElementById("models-list-container");
  if (!container) return;

  try {
    const data = await api.getModels();
    const models = data.models || [];

    if (models.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-title">No models registered</div>
        </div>
      `;
      return;
    }

    container.innerHTML = models.map((m) => renderModelCard(m)).join("");
  } catch (err) {
    container.innerHTML = `
      <div class="card" style="border-left: 4px solid var(--status-danger-dot);">
        <div style="font-weight: var(--font-weight-semibold); color: var(--status-danger-text);">
          Failed to load models metadata
        </div>
        <div style="font-size: var(--text-xs); color: var(--text-secondary); margin-top: 4px;">
          ${escapeHtml(err.message)}
        </div>
      </div>
    `;
  }
}

function renderModelCard(model) {
  const isLoaded = model.status === "loaded";
  const statusBadge = isLoaded
    ? `<span class="badge badge-low">LOADED & ACTIVE</span>`
    : `<span class="badge badge-high">${escapeHtml(model.status.toUpperCase())}</span>`;

  let featuresContent = "";
  if (Array.isArray(model.input_shape_or_features)) {
    featuresContent = `<span style="font-family: var(--font-mono); font-size: var(--text-xs);">${escapeHtml(JSON.stringify(model.input_shape_or_features))}</span>`;
  } else {
    featuresContent = `<span style="font-family: var(--font-mono); font-size: var(--text-xs);">${escapeHtml(String(model.input_shape_or_features))}</span>`;
  }

  // Metrics block - honest states
  let metricsBlock = "";
  if (model.metrics && model.metrics.accuracy !== null && model.metrics.accuracy !== undefined) {
    metricsBlock = `
      <table class="table" style="margin-top: 8px;">
        <thead>
          <tr>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1 Score</th>
            <th>ROC-AUC</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>${model.metrics.accuracy}</td>
            <td>${model.metrics.precision}</td>
            <td>${model.metrics.recall}</td>
            <td>${model.metrics.f1_score}</td>
            <td>${model.metrics.roc_auc}</td>
          </tr>
        </tbody>
      </table>
    `;
  } else {
    metricsBlock = `
      <div style="padding: 12px; background-color: var(--bg-surface-subtle); border-radius: var(--radius-md); font-size: var(--text-xs); color: var(--text-tertiary); margin-top: 8px;">
        <strong>Validation Metrics:</strong> Not available (Dataset not yet supplied to training pipeline).
        <div style="margin-top: 4px; font-family: var(--font-mono);">${escapeHtml(model.metrics?.status || "Pending dataset")}</div>
      </div>
    `;
  }

  return `
    <div class="card" style="margin-bottom: var(--space-6);">
      <div class="card-header">
        <div>
          <div class="card-title">${escapeHtml(model.name)} <span style="font-size: var(--text-xs); color: var(--text-tertiary); font-family: var(--font-mono);">v${escapeHtml(model.version)}</span></div>
          <div class="card-subtitle">${escapeHtml(model.task)}</div>
        </div>
        ${statusBadge}
      </div>

      <div class="grid-2" style="font-size: var(--text-sm); gap: var(--space-4);">
        <div>
          <div style="color: var(--text-tertiary); font-size: var(--text-xs);">Framework</div>
          <div><strong>${escapeHtml(model.framework)}</strong></div>
        </div>
        <div>
          <div style="color: var(--text-tertiary); font-size: var(--text-xs);">Dataset</div>
          <div>${escapeHtml(model.dataset)}</div>
        </div>
        <div>
          <div style="color: var(--text-tertiary); font-size: var(--text-xs);">Artifact File</div>
          <div style="font-family: var(--font-mono); font-size: var(--text-xs);">${escapeHtml(model.file_path)}</div>
        </div>
        <div>
          <div style="color: var(--text-tertiary); font-size: var(--text-xs);">Inputs</div>
          <div>${featuresContent}</div>
        </div>
      </div>

      <div style="margin-top: var(--space-4);">
        ${metricsBlock}
      </div>
    </div>
  `;
}
