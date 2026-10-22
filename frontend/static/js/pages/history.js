/**
 * PredictGuard AI - History Store & Audit Log Page (ES Module)
 */

import { showToast } from "../components/toast.js";
import { clearHistory, exportHistoryCSV, getHistory, escapeHtml } from "../state.js";

let currentFilterType = "all";
let currentFilterRisk = "all";

export function initHistoryPage() {
  const filterType = document.getElementById("history-filter-type");
  const filterRisk = document.getElementById("history-filter-risk");
  const exportBtn = document.getElementById("history-export-btn");
  const clearBtn = document.getElementById("history-clear-btn");

  filterType?.addEventListener("change", (e) => {
    currentFilterType = e.target.value;
    renderHistoryTable();
  });

  filterRisk?.addEventListener("change", (e) => {
    currentFilterRisk = e.target.value;
    renderHistoryTable();
  });

  exportBtn?.addEventListener("click", () => {
    const csv = exportHistoryCSV();
    if (!csv) {
      showToast("No history records available to export", "info");
      return;
    }
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `predictguard_audit_log_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast("Audit log CSV exported", "success");
  });

  clearBtn?.addEventListener("click", () => {
    if (confirm("Are you sure you want to clear all prediction history? This action cannot be undone.")) {
      clearHistory();
      renderHistoryTable();
      showToast("History cleared", "info");
    }
  });

  renderHistoryTable();
}

export function renderHistoryTable() {
  const tbody = document.getElementById("history-table-tbody");
  if (!tbody) return;

  let history = getHistory();

  if (!history || history.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="6" class="empty-state">
          <div class="empty-state-title">No audit records found</div>
          <div class="empty-state-desc">Prediction logs will appear here automatically when analysis requests are processed.</div>
        </td>
      </tr>
    `;
    return;
  }

  // Filtering
  if (currentFilterType !== "all") {
    history = history.filter((h) => h.type === currentFilterType);
  }
  if (currentFilterRisk !== "all") {
    history = history.filter((h) => h.risk_level === currentFilterRisk);
  }

  if (history.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="6" class="empty-state">
          <div class="empty-state-title">No matching records</div>
          <div class="empty-state-desc">Try resetting the filters above to view other history items.</div>
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = history
    .map((item) => {
      const dateStr = item.timestamp ? new Date(item.timestamp).toLocaleString() : "--";
      const probPct = item.probability !== undefined ? (item.probability * 100).toFixed(1) + "%" : "--";
      const riskClass = item.risk_level ? item.risk_level.toLowerCase() : "neutral";

      return `
        <tr>
          <td style="font-family: var(--font-mono); font-size: var(--text-xs);">${escapeHtml(item.id)}</td>
          <td style="font-size: var(--text-xs); color: var(--text-secondary);">${escapeHtml(dateStr)}</td>
          <td>${item.type === "image" ? "Casting Inspection" : "Machine Telemetry"}</td>
          <td><strong>${escapeHtml(item.prediction)}</strong></td>
          <td style="font-family: var(--font-mono);">${probPct}</td>
          <td>
            <span class="badge badge-${escapeHtml(riskClass)}">
              ${escapeHtml(item.risk_level || "UNKNOWN")}
            </span>
          </td>
        </tr>
      `;
    })
    .join("");
}
