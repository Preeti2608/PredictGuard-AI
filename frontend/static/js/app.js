/**
 * PredictGuard AI - Enterprise Operations Console Entry Point (ES Module)
 */

import { api } from "./api.js";
import { initCastingPage } from "./pages/casting.js";
import { initHistoryPage, renderHistoryTable } from "./pages/history.js";
import { initMachinePage } from "./pages/machine.js";
import { initModelsPage } from "./pages/models.js";
import { initOverviewPage, renderOverviewKPIs, checkSystemStatus } from "./pages/overview.js";
import { getStoredTheme, setStoredTheme } from "./state.js";

// Initialize Application on DOMContentLoaded
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initNavigation();
  initHealthPolling();

  // Initialize page-specific modules
  initOverviewPage();
  initCastingPage();
  initMachinePage();
  initHistoryPage();
  initModelsPage();
});

function initTheme() {
  const root = document.documentElement;
  const toggleBtn = document.getElementById("theme-toggle-btn");
  const themeLabel = document.getElementById("theme-toggle-label");

  // Determine initial theme: stored preference or prefers-color-scheme
  const stored = getStoredTheme();
  let activeTheme = stored;

  if (!activeTheme) {
    activeTheme = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  applyTheme(activeTheme);

  toggleBtn?.addEventListener("click", () => {
    const current = root.getAttribute("data-theme") || "light";
    const next = current === "dark" ? "light" : "dark";
    applyTheme(next);
    setStoredTheme(next);
  });

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    if (themeLabel) {
      themeLabel.textContent = theme === "dark" ? "Light Mode" : "Dark Mode";
    }
  }
}

function initNavigation() {
  const navTabs = document.querySelectorAll(".nav-tab-btn[data-page]");
  const pages = document.querySelectorAll(".page-view");
  const mobileToggle = document.getElementById("mobile-nav-toggle");
  const topNav = document.getElementById("top-nav");
  const brandHomeLink = document.getElementById("brand-home-link");
  const heroCtaCasting = document.getElementById("hero-cta-casting");
  const heroCtaSensors = document.getElementById("hero-cta-sensors");

  function switchPage(targetPageId) {
    pages.forEach((p) => {
      p.classList.remove("active");
      if (p.id === `page-${targetPageId}`) {
        p.classList.add("active");
      }
    });

    navTabs.forEach((btn) => {
      if (btn.getAttribute("data-page") === targetPageId) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });

    // Close mobile drawer if open
    topNav?.classList.remove("open");

    // Scroll smoothly to top
    window.scrollTo({ top: 0, behavior: "smooth" });

    // Page refresh callbacks
    if (targetPageId === "overview") {
      renderOverviewKPIs();
      checkSystemStatus();
    } else if (targetPageId === "history") {
      renderHistoryTable();
    } else if (targetPageId === "models") {
      initModelsPage();
    }
  }

  navTabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      const page = btn.getAttribute("data-page");
      if (page) switchPage(page);
    });
  });

  brandHomeLink?.addEventListener("click", (e) => {
    e.preventDefault();
    switchPage("overview");
  });

  heroCtaCasting?.addEventListener("click", () => {
    switchPage("casting");
  });

  heroCtaSensors?.addEventListener("click", () => {
    switchPage("machine");
  });

  // Mobile drawer controls
  mobileToggle?.addEventListener("click", () => {
    topNav?.classList.toggle("open");
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && topNav?.classList.contains("open")) {
      topNav.classList.remove("open");
    }
  });
}

function initHealthPolling() {
  const statusPill = document.getElementById("model-status-pill");
  const statusText = document.getElementById("model-status-text");

  async function poll() {
    try {
      const res = await api.checkHealth();
      if (!statusPill || !statusText) return;

      if (res.ok && res.data?.status === "ready") {
        statusPill.className = "status-pill ready";
        statusText.textContent = "MODELS READY";
      } else {
        statusPill.className = "status-pill degraded";
        statusText.textContent = "SERVICE DEGRADED";
      }
    } catch {
      if (statusPill && statusText) {
        statusPill.className = "status-pill not_ready";
        statusText.textContent = "OFFLINE";
      }
    }
  }

  poll();
  setInterval(poll, 15000); // Poll every 15 seconds
}
