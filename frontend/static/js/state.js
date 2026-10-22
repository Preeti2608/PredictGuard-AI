/**
 * PredictGuard AI - Application State & Storage Management
 */

const HISTORY_STORAGE_KEY = "predictguard_history";
const THEME_STORAGE_KEY = "predictguard_theme";

// In-memory fallback if localStorage is unavailable
const memoryFallback = {
  history: [],
  theme: null,
};

export function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  const s = String(str);
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

export function getStoredTheme() {
  try {
    return localStorage.getItem(THEME_STORAGE_KEY) || memoryFallback.theme;
  } catch (err) {
    console.warn("localStorage inaccessible for theme:", err);
    return memoryFallback.theme;
  }
}

export function setStoredTheme(theme) {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch (err) {
    memoryFallback.theme = theme;
  }
}

export function getHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    return raw ? JSON.parse(raw) : memoryFallback.history;
  } catch (err) {
    console.warn("localStorage inaccessible for history:", err);
    return memoryFallback.history;
  }
}

export function addHistoryEntry(entry) {
  try {
    const history = getHistory();
    const item = {
      id: "PG-" + Math.random().toString(36).substring(2, 9).toUpperCase(),
      timestamp: new Date().toISOString(),
      ...entry,
    };
    history.unshift(item);
    // Keep last 100 entries
    const trimmed = history.slice(0, 100);
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(trimmed));
    } catch (e) {
      memoryFallback.history = trimmed;
    }
    return item;
  } catch (err) {
    console.warn("Error adding history entry:", err);
  }
}

export function clearHistory() {
  try {
    localStorage.removeItem(HISTORY_STORAGE_KEY);
  } catch (err) {
    // ignore
  }
  memoryFallback.history = [];
}

export function exportHistoryCSV() {
  const history = getHistory();
  if (!history || history.length === 0) return null;

  const headers = ["ID", "Timestamp", "Type", "Prediction", "Probability", "Risk Level", "Latency (ms)"];
  const rows = history.map((item) => [
    item.id,
    item.timestamp,
    item.type,
    item.prediction,
    (item.probability * 100).toFixed(2) + "%",
    item.risk_level,
    item.latency_ms,
  ]);

  const csvContent = [headers.join(","), ...rows.map((r) => r.map((c) => `"${c}"`).join(","))].join("\n");
  return csvContent;
}
