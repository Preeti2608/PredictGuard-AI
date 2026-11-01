# PredictGuard AI — Dataset Audit & Provenance Register

This document records the exact provenance, licensing, dataset dimensions, operational scope, and audit readiness status for every dataset integrated or planned in PredictGuard AI.

---

## 1. Active & Implemented Datasets

### A. AI4I 2020 Predictive Maintenance Dataset
- **Source**: UCI Machine Learning Repository / Kaggle AI4I 2020
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Citation**: Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications with a Working Example for Manufacturing Systems*. International Conference on Artificial Intelligence and Mobile Services (AIMS).
- **Size**: 10,000 synthetic samples mirroring real-world CNC milling machine operational telemetry.
- **Features**: Product variant (`Type` L/M/H), Air Temperature [K], Process Temperature [K], Rotational Speed [rpm], Torque [Nm], Tool Wear [min].
- **Target**: Binary `Machine failure` (339 failure positive instances, 9,661 nominal instances; ~3.39% severe class imbalance).
- **How Used in PredictGuard AI**:
  - Binary machine breakdown forecasting (`training/train_sensor.py`).
  - Served model: `models/sensor_failure_calibrated.joblib` (Calibrated Support Vector Machine).
  - Explicit selection rule: Primary metric = Failure Recall (Class 1) with Platt scaling calibration.
  - Continuous tool wear degradation forecasting (`training/train_regression.py` using Random Forest regressor).
- **Readiness Status**: **TRAINED & ACTIVE** (Dataset present at `data/ai4i2020.csv`).

---

### B. Casting Product Image Quality Dataset
- **Source**: Kaggle (`ravirajsinh45/real-life-industrial-dataset-of-casting-product`)
- **License**: CC0: Public Domain
- **Citation**: Pilot production manufacturing plant submersible pump impeller inspection camera captures.
- **Scope**: Top-down ($90^\circ$ perpendicular), planar surface inspection of submersible pump impeller castings on conveyor belt fixtures.
- **Classes**:
  - `ok_front`: Normal, defect-free cast metal face.
  - `def_front`: Defective casting displaying voids, blowholes, porosity pinholes, or cracks.
- **Model Weights**: `models/casting_defect_classifier.h5` (MobileNetV2 transfer learning).
- **Readiness Status**: **PENDING TEST SPLIT EVALUATION**
  - Model weights are present in `models/casting_defect_classifier.h5`.
  - Full held-out test split evaluation via `training/evaluate_image.py` requires raw test images at `data/casting_data/test/`.
  - Metrics in `docs/metrics/image_metrics.json` and `models/model_metadata.json` remain explicitly marked pending until real test evaluation completes.

---

## 2. Multi-Domain Inspection Datasets (Pending Download)

### C. NEU Surface Defect Database
- **Source**: Northeastern University (NEU) / Kaggle (`kaustubhdikshit/neu-surface-defect-database`)
- **License**: Academic & Educational Use
- **Citation**: Song, K., & Yan, Y. (2013). *A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defect detection*. Applied Surface Science, 285, 858-864.
- **Size**: 1,800 grayscale images ($200 \times 200$ px), 300 samples per class across 6 surface defect types:
  1. `crazing` (network of micro-cracks)
  2. `inclusion` (non-metallic inclusions)
  3. `patches` (surface oxide / slag scale patches)
  4. `pitted_surface` (pitting cavitations)
  5. `rolled_in_scale` (rolled-in mill scale)
  6. `scratches` (mechanical abrasion scratches)
- **Important Domain Note**: NEU has **no "normal" or "good" class**. It is strictly a defect-type categorization database. The API does not output a normal/defective binary risk for this domain without explicit operator mapping.
- **Training Pipeline**: `training/train_neu.py` (two-stage MobileNetV2 with stratified 70/15/15 split saved to `data/neu_splits.json`).
- **Readiness Status**: **BLOCKED_PENDING_DOWNLOAD**
  - Architectural placeholder present at `models/neu_defect_classifier.keras`.
  - API returns HTTP 503 `MODEL_NOT_TRAINED` until real data is extracted into `data/neu_data/` and `train_neu.py` is executed.

---

### D. MVTec Anomaly Detection (MVTec AD)
- **Source**: MVTec Software GmbH (`https://www.mvtec.com/research-teaching/datasets/mvtec-ad`)
- **License**: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**
  - *Non-commercial research and academic use only.*
  - *Terms must be reviewed and accepted by the user prior to download.*
- **Citation**: Bergmann, P., Batzner, K., Fassbender, M., Löwe, P., Steger, C., & Wheeler, D. (2021). *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. International Journal of Computer Vision, 129(4), 1038-1059.
- **Scope**: Industrial unsupervised anomaly detection. Training split contains strictly defect-free nominal images (`train/good`). Testing split contains nominal and various anomalous defect types.
- **Target Categories**: `metal_nut`, `screw`, `transistor`.
- **Pipeline**: `training/train_anomaly.py` (MobileNetV2 feature embeddings + Isolation Forest & kNN distance scoring).
- **Readiness Status**: **BLOCKED_PENDING_DOWNLOAD** (Requires user terms acceptance).

---

## 3. Future Work & Audited Exclusions

### E. Bearing Defect Dataset (Kangning Li, PLOS ONE 2024)
- **Source**: Figshare (`doi: 10.6084/m9.figshare.25335076`)
- **Citation**: Li, K., et al. (2024). *A bearing defect detection method based on high-speed imaging*. PLOS ONE.
- **Audit Findings**:
  - The dataset consists of high-speed camera acquisitions from a bearing test rig.
  - Many public uploads under this DOI contain unsegmented, unlabeled time-series frames rather than clean categorical image sets.
  - In compliance with Non-Negotiable Rule 2 (minimum 200 verified labeled examples required), classifier training is **rejected** until verified ground-truth annotations are provided.
- **Readiness Status**: **FUTURE_WORK** (Removed from production enabled list).

---

### F. Visual Anomaly (VisA) Dataset
- **Source**: Amazon Science (`https://github.com/amazon-science/spot-diff`)
- **Citation**: Zou, Y., et al. (2022). *Spot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation*. ECCV.
- **Audit Findings**:
  - VisA contains 10,821 high-resolution images across 12 industrial components.
  - Licensing is reported inconsistently across public mirrors (CC BY vs CC BY-NC-SA).
  - Assumed non-commercial until formally verified on the official repository.
- **Readiness Status**: **FUTURE_WORK** (Deferred from current release).
