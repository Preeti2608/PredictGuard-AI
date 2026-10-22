from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "decision_tree.joblib"

model = joblib.load(MODEL_PATH)

sample = pd.DataFrame(
    [
        {
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 100,
        }
    ]
)

prediction = model.predict(sample)[0]

print("Sensor model loaded successfully!")
print("Prediction:", "Failure" if prediction == 1 else "No Failure")
