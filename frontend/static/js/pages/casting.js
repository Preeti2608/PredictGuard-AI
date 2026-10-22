/**
 * PredictGuard AI - Casting Inspection Page (ES Module)
 */

import { api } from "../api.js";
import { renderSvgGauge } from "../components/gauge.js";
import { showToast } from "../components/toast.js";
import { addHistoryEntry, escapeHtml } from "../state.js";

let selectedFile = null;
let lastImageResult = null;

export function initCastingPage() {
  const dropzone = document.getElementById("casting-dropzone");
  const fileInput = document.getElementById("casting-file-input");
  const analyzeBtn = document.getElementById("casting-analyze-btn");
  const btnSampleDef = document.getElementById("btn-sample-defective");
  const btnSampleNorm = document.getElementById("btn-sample-normal");
  const copyJsonBtn = document.getElementById("casting-copy-json");

  if (!dropzone || !fileInput) return;

  // File picker trigger
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  // Drag and drop events
  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    const target = e.target;
    if (target.files && target.files.length > 0) {
      handleFileSelected(target.files[0]);
    }
  });

  // Sample buttons
  btnSampleDef?.addEventListener("click", () => loadSampleImage("/sample_images/sample_defective.jpeg", "sample_defective.jpeg"));
  btnSampleNorm?.addEventListener("click", () => loadSampleImage("/sample_images/sample_normal.jpeg", "sample_normal.jpeg"));

  // Analyze button
  analyzeBtn?.addEventListener("click", handleAnalyze);

  // Copy JSON
  copyJsonBtn?.addEventListener("click", () => {
    if (!lastImageResult) return;
    navigator.clipboard.writeText(JSON.stringify(lastImageResult, null, 2))
      .then(() => showToast("Inspection JSON copied to clipboard", "success"))
      .catch(() => showToast("Failed to copy JSON", "error"));
  });
}

async function loadSampleImage(url, filename) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error("Could not fetch sample image file.");
    const blob = await res.blob();
    const file = new File([blob], filename, { type: "image/jpeg" });
    handleFileSelected(file);
    showToast(`Loaded ${filename}`, "info");
  } catch (err) {
    showToast(`Failed to load sample: ${err.message}`, "error");
  }
}

function handleFileSelected(file) {
  const errBox = document.getElementById("casting-file-error");
  const preview = document.getElementById("casting-preview-container");
  const previewImg = document.getElementById("casting-preview-img");
  const previewMeta = document.getElementById("casting-preview-meta");
  const analyzeBtn = document.getElementById("casting-analyze-btn");

  errBox.classList.remove("visible");
  errBox.textContent = "";

  // Validation
  if (!file.type.match(/^image\/(jpeg|png)$/i) && !file.name.match(/\.(jpg|jpeg|png)$/i)) {
    errBox.textContent = "Invalid file type. Please upload a JPEG or PNG image.";
    errBox.classList.add("visible");
    return;
  }

  if (file.size > 5 * 1024 * 1024) {
    errBox.textContent = "File exceeds the 5 MB maximum size limit.";
    errBox.classList.add("visible");
    return;
  }

  selectedFile = file;

  // Display preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    const imgObj = new Image();
    imgObj.onload = () => {
      previewMeta.innerHTML = `
        <div><strong>Filename:</strong> ${escapeHtml(file.name)}</div>
        <div><strong>File Size:</strong> ${(file.size / 1024).toFixed(1)} KB</div>
        <div><strong>Dimensions:</strong> ${imgObj.width} × ${imgObj.height} px</div>
      `;
      preview.style.display = "flex";
      analyzeBtn.disabled = false;
    };
    imgObj.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

async function handleAnalyze() {
  if (!selectedFile) return;

  const analyzeBtn = document.getElementById("casting-analyze-btn");
  const resultCard = document.getElementById("casting-result-card");
  const gaugeSlot = document.getElementById("casting-gauge-slot");
  const labelElem = document.getElementById("casting-result-label");
  const badgeElem = document.getElementById("casting-risk-badge");
  const actionElem = document.getElementById("casting-recommended-action");
  const latencyElem = document.getElementById("casting-latency");
  const reqIdElem = document.getElementById("casting-request-id");

  try {
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing Surface...";

    const res = await api.predictImage(selectedFile);
    lastImageResult = res;

    // Render gauge
    gaugeSlot.innerHTML = renderSvgGauge(res.defect_probability, res.risk_level, "Defect Probability");

    // Populate metadata
    labelElem.textContent = res.label;
    badgeElem.textContent = `${res.risk_level} RISK`;
    badgeElem.className = `badge badge-${res.risk_level.toLowerCase()}`;
    actionElem.textContent = res.recommended_action;
    latencyElem.textContent = `${res.latency_ms} ms`;
    reqIdElem.textContent = res.request_id;

    resultCard.style.display = "block";
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });

    // Store in history
    addHistoryEntry({
      type: "image",
      label: res.label,
      prediction: res.prediction,
      probability: res.defect_probability,
      risk_level: res.risk_level,
      latency_ms: res.latency_ms,
    });

    showToast("Inspection completed successfully", "success");
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Casting Surface";
  }
}
