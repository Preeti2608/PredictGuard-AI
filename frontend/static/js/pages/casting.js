/**
 * PredictGuard AI - Industrial Component & Surface Inspection Page (ES Module)
 * Multi-domain inspection: Casting Impellers, Ball/Roller Bearings, NEU Steel Strip
 */

import { api } from "../api.js";
import { renderSvgGauge } from "../components/gauge.js";
import { showToast } from "../components/toast.js";
import { addHistoryEntry, escapeHtml } from "../state.js";

let selectedFile = null;
let lastImageResult = null;
let currentDomain = "casting";

const DOMAIN_CONFIG = {
  casting: {
    title: "Pump Impeller Casting",
    trained: true,
    analyzeText: "Analyze Casting Surface",
    banner: "<strong>Active Domain: Submersible Pump Impeller Castings</strong> (Kaggle Casting Quality Dataset). Detects casting blowholes, pinholes, and shrinkage cavities.",
    sample1: { label: "Load Defective Casting", url: "/sample_images/sample_defective.jpeg", filename: "sample_defective.jpeg" },
    sample2: { label: "Load Normal Casting", url: "/sample_images/sample_normal.jpeg", filename: "sample_normal.jpeg" },
  },
  bearing: {
    title: "Ball & Roller Bearing",
    trained: false,
    analyzeText: "Domain Not Trained",
    banner: "<strong>Active Domain: Ball & Roller Bearing Inspection</strong> <span class='badge badge-high' style='margin-left: 8px;'>NOT TRAINED</span><br>This domain model is an untrained placeholder. Provide verified bearing dataset to enable production inference.",
    sample1: { label: "Synthetic Defective (Dev)", url: "/sample_images/synthetic_bearing_defective.jpeg", filename: "synthetic_bearing_defective.jpeg" },
    sample2: { label: "Synthetic Normal (Dev)", url: "/sample_images/synthetic_bearing_normal.jpeg", filename: "synthetic_bearing_normal.jpeg" },
  },
  neu_surface: {
    title: "Hot-Rolled Steel Strip",
    trained: false,
    analyzeText: "Domain Not Trained",
    banner: "<strong>Active Domain: NEU 6-Class Steel Strip Surface Defect Database</strong> <span class='badge badge-high' style='margin-left: 8px;'>NOT TRAINED</span><br>This domain model is an untrained placeholder. Provide NEU dataset to train and enable 6-class classification.",
    sample1: { label: "Synthetic Pitted (Dev)", url: "/sample_images/synthetic_neu_pitted.jpeg", filename: "synthetic_neu_pitted.jpeg" },
    sample2: { label: "Synthetic Scratched (Dev)", url: "/sample_images/synthetic_neu_scratches.jpeg", filename: "synthetic_neu_scratches.jpeg" },
  },
};

export function initCastingPage() {
  const dropzone = document.getElementById("casting-dropzone");
  const fileInput = document.getElementById("casting-file-input");
  const analyzeBtn = document.getElementById("casting-analyze-btn");
  const btnSampleDef = document.getElementById("btn-sample-defective");
  const btnSampleNorm = document.getElementById("btn-sample-normal");
  const copyJsonBtn = document.getElementById("casting-copy-json");
  const domainPills = document.querySelectorAll(".domain-pill");

  if (!dropzone || !fileInput) return;

  // Domain selection pills
  domainPills.forEach((pill) => {
    pill.addEventListener("click", () => {
      domainPills.forEach((p) => {
        p.classList.remove("active", "btn-primary");
        p.classList.add("btn-secondary");
      });
      pill.classList.add("active", "btn-primary");
      pill.classList.remove("btn-secondary");

      const dom = pill.getAttribute("data-domain");
      setDomain(dom);
    });
  });

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
  btnSampleDef?.addEventListener("click", () => {
    const conf = DOMAIN_CONFIG[currentDomain] || DOMAIN_CONFIG.casting;
    loadSampleImage(conf.sample1.url, conf.sample1.filename);
  });
  btnSampleNorm?.addEventListener("click", () => {
    const conf = DOMAIN_CONFIG[currentDomain] || DOMAIN_CONFIG.casting;
    loadSampleImage(conf.sample2.url, conf.sample2.filename);
  });

  // Analyze button
  analyzeBtn?.addEventListener("click", handleAnalyze);

  // Copy JSON
  copyJsonBtn?.addEventListener("click", () => {
    if (!lastImageResult) return;
    navigator.clipboard
      .writeText(JSON.stringify(lastImageResult, null, 2))
      .then(() => showToast("Inspection JSON copied to clipboard", "success"))
      .catch(() => showToast("Failed to copy JSON", "error"));
  });
}

function setDomain(dom) {
  if (!DOMAIN_CONFIG[dom]) return;
  currentDomain = dom;
  const conf = DOMAIN_CONFIG[dom];

  const banner = document.getElementById("domain-notice-banner");
  if (banner) {
    banner.innerHTML = conf.banner;
  }

  const btnSampleDef = document.getElementById("btn-sample-defective");
  const btnSampleNorm = document.getElementById("btn-sample-normal");
  if (btnSampleDef) btnSampleDef.textContent = conf.sample1.label;
  if (btnSampleNorm) btnSampleNorm.textContent = conf.sample2.label;

  const analyzeBtn = document.getElementById("casting-analyze-btn");
  if (analyzeBtn) {
    if (!conf.trained) {
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = "Domain Not Trained (Placeholder)";
    } else {
      analyzeBtn.disabled = !selectedFile;
      analyzeBtn.textContent = conf.analyzeText;
    }
  }

  showToast(`Switched inspection domain to ${conf.title}`, "info");
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
  const conf = DOMAIN_CONFIG[currentDomain] || DOMAIN_CONFIG.casting;

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
        <div><strong>Target Domain:</strong> ${escapeHtml(conf.title)}</div>
      `;
      preview.style.display = "flex";
      if (!conf.trained) {
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = "Domain Not Trained (Placeholder)";
        showToast(`The ${conf.title} domain is an untrained placeholder. Inference is disabled.`, "warning");
      } else {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = conf.analyzeText;
      }
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
  const neuSlot = document.getElementById("neu-classes-breakdown");
  const conf = DOMAIN_CONFIG[currentDomain] || DOMAIN_CONFIG.casting;

  try {
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing Surface...";

    const res = await api.predictImage(selectedFile, currentDomain);
    lastImageResult = res;

    // Render gauge
    const isPerspectiveWarning = res.prediction === "perspective_warning";
    const gaugeTitle = isPerspectiveWarning ? "Perspective Uncertainty" : "Defect Probability";
    gaugeSlot.innerHTML = renderSvgGauge(res.defect_probability, res.risk_level, gaugeTitle);

    // Populate metadata
    labelElem.textContent = res.label;
    if (isPerspectiveWarning) {
      badgeElem.textContent = "PERSPECTIVE WARNING";
      badgeElem.className = "badge badge-medium";
      actionElem.style.borderColor = "var(--status-warning-border)";
      actionElem.style.backgroundColor = "var(--status-warning-bg)";
    } else {
      badgeElem.textContent = `${res.risk_level} RISK`;
      badgeElem.className = `badge badge-${res.risk_level.toLowerCase()}`;
      actionElem.style.borderColor = "var(--border-default)";
      actionElem.style.backgroundColor = "var(--bg-surface-subtle)";
    }
    actionElem.textContent = res.recommended_action;
    latencyElem.textContent = `${res.latency_ms} ms`;
    reqIdElem.textContent = res.request_id;

    // Multi-class breakdown for NEU dataset
    if (neuSlot) {
      if (res.defect_classes) {
        let html = `
          <div style="font-size: var(--text-xs); color: var(--text-tertiary); text-transform: uppercase; font-weight: var(--font-weight-semibold); margin-bottom: 8px;">
            NEU Surface Defect Probabilities (6 Classes)
          </div>
          <div style="display: flex; flex-direction: column; gap: 6px;">
        `;
        for (const [cls, prob] of Object.entries(res.defect_classes)) {
          const pct = Math.round(prob * 100);
          const isTop = res.prediction === cls;
          const barColor = isTop ? "var(--brand-green)" : "var(--border-default)";
          const formattedCls = cls.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
          html += `
            <div style="font-size: var(--text-xs);">
              <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                <span style="font-weight: ${isTop ? "bold" : "normal"}; color: ${isTop ? "var(--text-primary)" : "var(--text-secondary)"};">
                  ${escapeHtml(formattedCls)}
                </span>
                <span style="color: var(--text-tertiary);">${pct}%</span>
              </div>
              <div style="width: 100%; background-color: var(--bg-surface-subtle); border-radius: 4px; height: 6px; overflow: hidden;">
                <div style="width: ${pct}%; background-color: ${barColor}; height: 100%; transition: width 0.3s ease;"></div>
              </div>
            </div>
          `;
        }
        html += "</div>";
        neuSlot.innerHTML = html;
        neuSlot.style.display = "block";
      } else {
        neuSlot.style.display = "none";
      }
    }

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
    analyzeBtn.textContent = conf.analyzeText;
  }
}
