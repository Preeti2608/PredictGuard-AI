# Changelog

All notable changes to **PredictGuard AI** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-10-21

### Added
- **Production Backend Restructure**:
  - Modularized into `app/main.py`, `app/config.py`, `app/schemas.py`, `app/routers/`, `app/services/`, and `app/core/`.
  - Application lifespan model loading with centralized `ModelRegistry`.
  - Non-blocking threadpool execution for CPU-intensive ML inference.
  - Request-ID tracing middleware (`X-Request-ID`) and duration tracking (`X-Response-Time-Ms`).
  - Structured JSON logging for production observability.
  - Uniform error format `{error_code, message, request_id, details}`.
  - Configurable rate limiting on inference endpoints using `slowapi`.
  - Safe CORS policy defaulting to same-origin.
  - Health check endpoints: `/health/live` (liveness), `/health/ready` (readiness with 503 on model failure), and `/api/health` (legacy compatibility).
  - Versioned `/v1/...` API endpoints with thin legacy path aliases.
  - `GET /v1/models` model metadata catalog endpoint with runtime load statuses.
  - `POST /v1/predict/wear` continuous tool degradation forecasting endpoint.
- **Image Hardening & Preprocessing**:
  - Unified canonical preprocessing function eliminating train/serve interpolation skew.
  - Magic byte header inspection verifying real JPEG/PNG signatures.
  - 5 MB payload size limit and 10 MP decompression bomb defense.
  - Quantitative defect probability (`1 - score`), confidence, risk tiering, and actionable operator advisories.
- **Sensor Telemetry Validation & Explanations**:
  - Physical range bounds validation based on AI4I dataset.
  - Explicit rejection of `NaN` and `Inf` float values.
  - Dual support for `snake_case` attributes and legacy bracketed field aliases.
  - Ranked feature contribution explanations combining tree importances with nominal baseline deviations.
- **Enterprise Operations Console (Frontend)**:
  - Enterprise UI design in the spirit of Grafana/Datadog/Linear.
  - Vanilla HTML/CSS/ES Modules architecture without runtime external CDN dependencies.
  - Light and Dark themes with `localStorage` persistence and `prefers-color-scheme` support.
  - Responsive app shell with collapsible drawer for viewports under 900px.
  - Live system status polling and KPI overview dashboard.
  - Drag-and-drop casting surface inspection with inline SVG probability gauges.
  - Validated machine health form with preset buttons and horizontal factor deviation bars.
  - Audit log table with CSV export and clear dialog.
  - WCAG 2.1 AA accessibility, keyboard navigation, and visible focus rings.
- **Reproducible Training Pipelines**:
  - `training/check_preprocessing.py`: Evaluates interpolation modes to eliminate train/serve skew.
  - `training/train_image.py`: MobileNetV2 transfer learning with data augmentation and metrics export.
  - `training/train_sensor.py`: 7-algorithm benchmark comparing classifiers on failure recall with `CalibratedClassifierCV`.
  - `training/train_regression.py`: Tool wear degradation regression comparing Linear, Polynomial, and Ridge models.
- **Testing & Tooling**:
  - Pytest test suite with 32 unit and integration tests achieving >81% code coverage.
  - Ruff linter, formatter, and Mypy static type checking configurations in `pyproject.toml`.
  - GitHub Actions CI workflow covering linting, type checks, test suite, and container build.
  - Pre-commit hook configuration.
  - Multi-stage Dockerfile with non-root unprivileged execution and `/health/ready` healthcheck.

### Changed
- Replaced insecure wildcard CORS (`allow_origins=["*"]` + `allow_credentials=True`) with configurable origins list defaulting to same-origin.
- Replaced neon/glassmorphism UI with enterprise-grade slate console.
- Pinned exact dependency versions in `requirements.txt` and `requirements-dev.txt`.
