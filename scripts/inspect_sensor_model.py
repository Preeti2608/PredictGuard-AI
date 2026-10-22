from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "decision_tree.joblib"

model = joblib.load(MODEL_PATH)

print("Sensor model loaded successfully!")
print("Model type:", type(model).__name__)

if hasattr(model, "feature_names_in_"):
    print("Expected features:", list(model.feature_names_in_))

if hasattr(model, "named_steps"):
    print("Pipeline steps:", list(model.named_steps.keys()))

    for name, step in model.named_steps.items():
        print(f"\nStep: {name}")
        print("Type:", type(step).__name__)

        if hasattr(step, "feature_names_in_"):
            print("Input features:", list(step.feature_names_in_))

        if hasattr(step, "get_feature_names_out"):
            try:
                print("Output features:", list(step.get_feature_names_out()))
            except Exception:
                pass

if hasattr(model, "classes_"):
    print("\nClasses:", model.classes_)

if hasattr(model, "n_features_in_"):
    print("Number of input features:", model.n_features_in_)
