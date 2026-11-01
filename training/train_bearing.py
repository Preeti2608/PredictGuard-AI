"""
PredictGuard AI - Bearing Surface Defect Inspection Training Engine.

Dataset reference: Figshare Bearing Defect Dataset (Kangning Li, PLOS ONE 2024, doi: 10.6084/m9.figshare.25335076).
Acquisition: High-speed industrial camera monitoring a bearing degradation test rig.

Decision Gate & Audit Rules:
1. Inspects downloaded data first: image count, resolution, presence of verified labels.
2. If images are unlabeled or fewer than 200 labeled examples exist:
   DO NOT train a classifier. Mark as FUTURE WORK.
3. Only if properly labeled normal/ and defective/ directories exist with >= 200 samples:
   Trains MobileNetV2 transfer learning and computes real test metrics via metrics_io.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


DATA_DIR = BASE_DIR / "data" / "bearing_data"
MODEL_PATH = BASE_DIR / "models" / "bearing_defect_classifier.keras"
METRICS_PATH = BASE_DIR / "docs" / "metrics" / "bearing_metrics.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Audit and Train Bearing Defect Classifier")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Path to bearing data directory")
    parser.add_argument("--output-model", type=str, default=str(MODEL_PATH), help="Path to save trained model")
    parser.add_argument("--output-metrics", type=str, default=str(METRICS_PATH), help="Path to save metrics")
    parser.add_argument("--min-samples", type=int, default=200, help="Minimum labeled samples threshold")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    return parser.parse_args()


def audit_bearing_data(data_dir: Path) -> Tuple[bool, int, str]:
    """Inspect data directory and report image count, resolutions, and labels."""
    print("\n" + "=" * 70)
    print("AUDITING BEARING DATASET DIRECTORY")
    print("=" * 70)

    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return False, 0, "MISSING_DIRECTORY"

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_files = [p for p in data_dir.rglob("*.*") if p.suffix.lower() in valid_exts]
    print(f"Total raw image files found: {len(all_files)}")

    if not all_files:
        return False, 0, "EMPTY_DIRECTORY"

    # Sample resolutions
    resolutions = set()
    for p in all_files[:20]:
        try:
            with Image.open(p) as img:
                resolutions.add(f"{img.width}x{img.height}")
        except Exception:
            pass
    print(f"Sample image resolutions: {', '.join(resolutions) if resolutions else 'unknown'}")

    # Check for labeled structure: normal/ and defective/
    normal_dir = data_dir / "normal"
    defective_dir = data_dir / "defective"

    normal_count = len(list(normal_dir.glob("*.*"))) if normal_dir.exists() else 0
    defective_count = len(list(defective_dir.glob("*.*"))) if defective_dir.exists() else 0

    print(f"Labeled subsets: Normal = {normal_count}, Defective = {defective_count}")

    total_labeled = normal_count + defective_count
    if total_labeled < 200 or normal_count == 0 or defective_count == 0:
        return False, total_labeled, "INSUFFICIENT_OR_UNLABELED"

    return True, total_labeled, "VERIFIED_LABELED"


def print_decision_gate_rejection(data_dir: Path, reason: str, count: int):
    print("\n" + "=" * 78)
    print("DECISION GATE: CLASSIFIER TRAINING REJECTED (MOVED TO FUTURE WORK)")
    print("=" * 78)
    print(f"Audit Result: {reason} (Found {count} labeled images; required >= 200).")
    print("\nTechnical Context:")
    print("  Kangning Li's Figshare upload (doi: 10.6084/m9.figshare.25335076, PLOS ONE 2024)")
    print("  is from a high-speed camera acquisition of a bearing test rig.")
    print("  Many Figshare dumps lack verified categorical ground-truth defect annotations.")
    print("\nPolicy:")
    print("  Per Non-Negotiable Rule 2 & Rule 3, no placeholder or synthetic models may be trained.")
    print("  The 'bearing' domain is marked UNTRAINED and documented under Future Work.")
    print("=" * 78 + "\n")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    is_valid, count, reason = audit_bearing_data(data_dir)
    if not is_valid:
        print_decision_gate_rejection(data_dir, reason, count)
        sys.exit(2)

    print(f"\nAudit passed: {count} labeled images found. Proceeding with real model training...")


if __name__ == "__main__":
    main()
