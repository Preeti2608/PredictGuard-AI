# PredictGuard AI — Predictive Maintenance Command Center

PredictGuard AI is an industrial predictive maintenance and quality assurance platform. It unifies computer vision and multi-sensor machine learning to automate equipment diagnostics, prevent unexpected mechanical downtime, and detect manufacturing defects before components leave the production line.

The system features a **futuristic industrial AI dashboard** interface backed by an asynchronous **FastAPI** microservice serving two distinct, pre-trained machine learning workflows.

---

## 1. Project Overview & Objectives

Predictive maintenance in modern smart factories requires dual-horizon diagnostics:
1. **Visual Quality Assurance:** Identifying surface faults and casting irregularities immediately after component fabrication.
2. **Operational Telemetry Health:** Monitoring in-service machine vitals (temperatures, rotational speeds, drive torques, and cumulative tool wear) to preempt catastrophic mechanical failure.

PredictGuard AI bridges these two operational domains into a unified command center, providing instant diagnostic feedback, transparent model confidence indicators, and interactive telemetry testing.

---

## 2. Key Features

- **Casting Defect Classification (Computer Vision):**
  - Accepts industrial casting surface images (JPG, JPEG, PNG).
  - Preprocesses inputs to standardized $224 \times 224 \times 3$ normalized RGB tensors.
  - Classifies parts into **Normal Casting** (`ok_front`) or **Defective Casting** (`def_front`).
  - Transparently displays the raw sigmoid activation score alongside a calibrated threshold gauge ($0.50$ decision boundary).
- **Machine Failure Prediction (Sensor Telemetry):**
  - Analyzes 6 continuous and categorical operating parameters (`Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`).
  - Automatically standardizes and encodes features via an integrated scikit-learn `ColumnTransformer`.
  - Produces binary failure verdicts (**No Failure** vs. **Failure**) and reports decision tree leaf class distributions.
- **Futuristic AI Command Center Interface:**
  - Dark-mode industrial interface built with charcoal tones (`#070908`), subtle glassmorphism surfaces (`backdrop-filter`), and refined lime/acid-green accents (`#a3e635`).
  - Real-time connection badge verifying continuous connectivity with `/api/health`.
  - Interactive drag-and-drop file ingestion zone with live image preview and metadata chips.
  - Instant one-click sample loaders for both normal and defective casting images.
  - Telemetry quick-fill presets for nominal and high-stress operating states.
  - Full keyboard navigation support and accessible ARIA live regions.
- **Production-Ready FastAPI Microservice:**
  - Automated OpenAPI documentation at `/docs` (Swagger UI) and `/redoc`.
  - Dynamic host and port binding supporting containerized and PaaS deployments.
  - Strict Pydantic v2 request validation and centralized exception handling.

---

## 3. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend Framework** | [FastAPI](https://fastapi.tiangolo.com/) `0.148+`, [Uvicorn](https://www.uvicorn.org/) `0.54+` (ASGI server) |
| **Deep Learning** | [TensorFlow](https://www.tensorflow.org/) `2.20.0`, [Keras](https://keras.io/) `3.15+` |
| **Machine Learning** | [scikit-learn](https://scikit-learn.org/) `1.6.1`, [Joblib](https://joblib.readthedocs.io/) `1.6.0` |
| **Data & Image Processing** | [NumPy](https://numpy.org/) `2.5+`, [Pandas](https://pandas.pydata.org/) `3.0+`, [Pillow (PIL)](https://python-pillow.org/) `13.0+` |
| **Data Validation** | [Pydantic v2](https://docs.pydantic.dev/) `2.14+` |
| **Frontend** | Semantic HTML5, Vanilla CSS3 (Glassmorphism, CSS Custom Properties), Vanilla JavaScript (ES6+ Fetch & Drag/Drop APIs) |
| **Typography** | [Outfit](https://fonts.google.com/specimen/Outfit) (UI) & [JetBrains Mono](https://fonts.google.com/specimen/JetBrains+Mono) (Telemetry/Metrics) |
| **Deployment** | Docker, OCI Container Specification, Render/Railway Procfile |

---

## 4. Architecture & Request Flow

```
[ Industrial User / Operator ]
             │
             ▼
[ PredictGuard Web Command Center (index.html) ]
     │                              │
     │ (Multipart/form-data)        │ (JSON Payload)
     ▼                              ▼
[ POST /predict/image ]        [ POST /predict/sensors ]
     │                              │
     ▼                              ▼
[ PIL & NumPy Preprocessing ]   [ Pydantic & Pandas DataFrame ]
  • RGB conversion                • Schema validation
  • 224x224 Bicubic resize        • Exact column matching
  • Normalize ([0, 1])            • ColumnTransformer pipeline
     │                              │
     ▼                              ▼
[ TensorFlow CNN (.h5) ]        [ Scikit-Learn DecisionTree (.joblib) ]
  • Sigmoid output                • Feature scaling & one-hot encoding
  • Threshold >= 0.50             • Leaf traversal
     │                              │
     └───────────────┬──────────────┘
                     ▼
       [ Standardized JSON Response ]
                     │
                     ▼
  [ Interactive Command Center Visualizer ]
  • Status verdicts & glowing badges
  • Decision threshold indicator gauge
  • Uncalibrated leaf probability breakdown
```

---

## 5. Project Folder Structure

```
AIML_Project/
├── .dockerignore                 # Excludes caches and datasets from Docker context
├── .gitignore                    # Excludes virtual environments and training datasets
├── Dockerfile                    # Production container build specification
├── Procfile                      # Process configuration for PaaS platforms (Render/Railway)
├── README.md                     # Technical documentation and operations guide
├── index.html                    # Redesigned futuristic AI command center frontend
├── main.py                       # FastAPI application, routing, and model inference service
├── requirements.txt              # Pinned production runtime dependencies
├── inspect_sensor_model.py       # Diagnostic script inspecting decision tree pipeline steps
├── test_image_model.py           # Verification script for Keras casting classifier
├── test_sensor_model.py          # Verification script for scikit-learn sensor pipeline
├── test_api_endpoints.py         # End-to-end integration and API test suite (8 tests)
├── models/                       # Trained machine learning model artifacts
│   ├── casting_defect_classifier.h5   # Trained TensorFlow/Keras CNN (~20.2 MB)
│   └── decision_tree.joblib           # Trained scikit-learn decision tree pipeline (~12.5 KB)
└── sample_images/                # Verified lightweight sample assets for live evaluation
    ├── sample_defective.jpeg     # Defective casting front sample
    └── sample_normal.jpeg        # Normal casting front sample
```

---

## 6. Trained Models & Specifications

### A. Casting Defect Classifier
- **File:** `models/casting_defect_classifier.h5`
- **Model Type:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras H5
- **Input Dimensions:** `(None, 224, 224, 3)` normalized float array in $[0.0, 1.0]$.
- **Output Layer:** Single unit with sigmoid activation.
- **Classification Mapping:**
  - $\text{Score} < 0.50 \implies$ `def_front` (**Defective casting**)
  - $\text{Score} \ge 0.50 \implies$ `ok_front` (**Normal casting**)

### B. Machine Failure Predictor
- **File:** `models/decision_tree.joblib`
- **Model Type:** Scikit-Learn `Pipeline`
- **Compatibility Requirement:** `scikit-learn==1.6.1`
- **Internal Pipeline Architecture:**
  1. `preprocessor` (`ColumnTransformer`):
     - `num`: `StandardScaler` applied across 5 continuous features (`Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`).
     - `cat`: `OneHotEncoder` applied to the categorical column `Type` (`L`, `M`, `H`).
  2. `classifier` (`DecisionTreeClassifier`):
     - Binary decision tree with 101 nodes.
- **Classification Mapping:**
  - Class `0` $\implies$ **No Failure** (Nominal operating conditions)
  - Class `1` $\implies$ **Failure** (Critical maintenance required)

---

## 7. Setup Instructions (Windows PowerShell)

Follow these steps to set up and run PredictGuard AI on Windows:

### Step 1: Open PowerShell and Navigate to the Project Directory
```powershell
cd d:\VS_Code_Projects\AIML_Project
```

### Step 2: Create and Activate Virtual Environment
```powershell
# Create virtual environment using Python 3.13
python -m venv .venv

# If script execution is restricted, temporarily permit execution for the session
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the environment
.\.venv\Scripts\Activate.ps1
```

### Step 3: Upgrade pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Running the Application

### Option A: Direct Python Execution
```powershell
python main.py
```
*The server binds automatically to `0.0.0.0` or `$HOST` and port `8000` or `$PORT`.*

### Option B: Uvicorn Development Mode (with Live Reload)
```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Accessing the Application
- **Command Center Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **System Health Verification:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 9. API Reference & Example Requests

### 1. Health Status
- **Method / Path:** `GET /api/health`
- **Description:** Verifies API liveness and model readiness.
- **Example Response:**
  ```json
  {
    "status": "online",
    "image_model_loaded": true,
    "sensor_model_loaded": true,
    "version": "1.0.0"
  }
  ```

---

### 2. Casting Defect Prediction
- **Method / Path:** `POST /predict/image`
- **Content-Type:** `multipart/form-data`
- **Form Field:** `file` (JPG or PNG binary)

**cURL Example:**
```bash
curl -X POST "http://127.0.0.1:8000/predict/image" \
  -H "accept: application/json" \
  -F "file=@sample_images/sample_normal.jpeg;type=image/jpeg"
```

**Example Response (Normal Casting):**
```json
{
  "prediction": "ok_front",
  "label": "Normal casting",
  "score": 0.9998854398727417,
  "note": "Score is the model output; verify its calibration before treating it as a probability."
}
```

**Example Response (Defective Casting):**
```json
{
  "prediction": "def_front",
  "label": "Defective casting",
  "score": 0.0002611843519844,
  "note": "Score is the model output; verify its calibration before treating it as a probability."
}
```

---

### 3. Sensor Failure Prediction
- **Method / Path:** `POST /predict/sensors`
- **Content-Type:** `application/json`

**cURL Example (Nominal Conditions):**
```bash
curl -X POST "http://127.0.0.1:8000/predict/sensors" \
  -H "Content-Type: application/json" \
  -d '{
    "Type": "M",
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 100
  }'
```

**Example Response (No Failure):**
```json
{
  "prediction": "No Failure",
  "prediction_code": 0,
  "class_probabilities": {
    "0": 0.9987474557695318,
    "1": 0.0012525442304681385
  }
}
```

**cURL Example (High-Stress Failure Conditions):**
```bash
curl -X POST "http://127.0.0.1:8000/predict/sensors" \
  -H "Content-Type: application/json" \
  -d '{
    "Type": "L",
    "Air temperature [K]": 302.0,
    "Process temperature [K]": 311.0,
    "Rotational speed [rpm]": 1350,
    "Torque [Nm]": 68.0,
    "Tool wear [min]": 215
  }'
```

**Example Response (Failure):**
```json
{
  "prediction": "Failure",
  "prediction_code": 1,
  "class_probabilities": {
    "0": 0.0,
    "1": 1.0
  }
}
```

---

## 10. Verified Testing & Validation Results

An automated integration test script (`test_api_endpoints.py`) is included to validate the microservice end-to-end.

To execute the test suite:
```powershell
python test_api_endpoints.py
```

### Verified Test Run Output:
```
============================================================
PREDICTGUARD AI — INTEGRATION & REGRESSION TEST SUITE
============================================================

[Test 1/8] Checking GET /api/health ...
 PASS: API Online. Models loaded: {'status': 'online', 'image_model_loaded': True, 'sensor_model_loaded': True, 'version': '1.0.0'}

[Test 2/8] Checking GET / (Dashboard UI Delivery) ...
 PASS: Homepage HTML served successfully (79766 bytes).

[Test 3/8] Checking POST /predict/sensors [Nominal Telemetry] ...
 PASS: Normal sensor prediction verified -> No Failure (Code: 0)
       Class probabilities: {'0': 0.9987474557695318, '1': 0.0012525442304681385}

[Test 4/8] Checking POST /predict/sensors [High-Stress Telemetry] ...
 PASS: Failure sensor prediction verified -> Failure (Code: 1)
       Class probabilities: {'0': 0.0, '1': 1.0}

[Test 5/8] Checking POST /predict/sensors [Invalid Machine Type] ...
 PASS: Request correctly rejected by Pydantic validation (HTTP 422).

[Test 6/8] Checking POST /predict/image [Defective Sample Image] ...
 PASS: Defective casting classified correctly -> Label: 'Defective casting' (Score: 0.000261)

[Test 7/8] Checking POST /predict/image [Normal Sample Image] ...
 PASS: Normal casting classified correctly -> Label: 'Normal casting' (Score: 0.999885)

[Test 8/8] Checking POST /predict/image [Invalid File Type Reject] ...
 PASS: Invalid file type rejected with HTTP 400: 'Please upload a JPG or PNG image.'

============================================================
TEST SUMMARY: 8/8 TESTS PASSED (100% SUCCESS)
============================================================
```

Individual unit scripts can also be run independently:
```powershell
python test_image_model.py
python test_sensor_model.py
python inspect_sensor_model.py
```

---

## 11. Deployment Instructions

PredictGuard AI is container-ready and configured for standard cloud environments.

### A. Docker Container Deployment

1. **Build the container image:**
   ```bash
   docker build -t predictguard-ai:latest .
   ```

2. **Run the container locally:**
   ```bash
   docker run -d --name predictguard -p 8000:8000 -e PORT=8000 predictguard-ai:latest
   ```

3. **Verify running container health:**
   ```bash
   curl http://localhost:8000/api/health
   ```

### B. Cloud Platform Deployment (Render / Railway / Hugging Face Spaces)

- **Render / Railway:**
  - Connect your Git repository.
  - Choose **Docker** environment (or Python runtime with start command `uvicorn main:app --host 0.0.0.0 --port $PORT`).
  - Set the health check path to `/api/health`.
- **Hugging Face Spaces (Recommended Free Tier):**
  - Create a new Space with **Docker SDK** (Blank template).
  - Push the repository files (`Dockerfile`, `requirements.txt`, `main.py`, `index.html`, `models/`, `sample_images/`).
  - Hugging Face provides generous memory allocation (16 GB RAM), perfectly accommodating TensorFlow's memory footprint without incurring cost.

### Resource & Memory Notice:
- Loading TensorFlow and Keras into memory consumes approximately **$350\text{ MB} - 480\text{ MB}$** of RAM on cold start.
- On free-tier platforms with strict $512\text{ MB}$ limits (such as Render Free Tier), transient memory spikes during initial model weight loading can cause Out-Of-Memory (OOM) exit codes. Allocate at least **$1\text{ GB}$** of container RAM for reliable multi-user production serving.

---

## 12. Known Limitations & Future Scope

### Known Limitations
1. **Uncalibrated Model Scores:**
   - The casting CNN outputs raw sigmoid activations. While activations near $0$ or $1$ reflect strong separation on the test distribution, they represent uncalibrated activations rather than Bayesian posterior probabilities.
   - The decision tree sensor model yields leaf-node sample frequencies. Under extreme pruning or deep single-sample leaves, probabilities can exhibit step-function behavior ($0.0$ or $1.0$).
2. **Camera Angle Constraints:**
   - The image classifier is specifically trained on front-facing surface imagery (`ok_front` and `def_front`). Oblique, lateral, or poorly illuminated casting views may yield degraded accuracy.
3. **Point-in-Time Telemetry:**
   - The sensor predictor evaluates instantaneous readings rather than continuous time-series trajectories (vibration spectrograms or temporal wear degradation).

### Future Scope
- **Post-Hoc Probability Calibration:** Incorporate Platt Scaling or Isotonic Regression on held-out validation sets.
- **Explainable AI (XAI):**
  - Implement Grad-CAM overlays highlighting specific defective surface pixels on casting images.
  - Implement TreeSHAP to provide waterfall feature-attribution charts detailing which sensor reading pushed the decision boundary toward mechanical failure.
- **Sequential Prognostics:** Upgrade the sensor pipeline from static Decision Trees to an LSTM / Temporal Convolutional Network (TCN) predicting Remaining Useful Life (RUL) in hours.

---

## 13. License & Academic Attribution
Developed as an applied predictive maintenance AI/ML project. Trained model weights and datasets are provided for academic research and educational evaluation.
