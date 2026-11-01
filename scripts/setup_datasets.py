"""
PredictGuard AI - Multi-Dataset Setup and Verification Tool.
Enforces strict ML auditing and dataset readiness verification:

1. Casting Quality Dataset (Submersible pump impellers) - Kaggle
2. NEU Surface Defect Database (6-class hot-rolled steel strip) - Kaggle
3. MVTec Anomaly Detection (MVTec AD) - Academic Non-Commercial (CC BY-NC-SA 4.0)
4. Bearing Defect Dataset (Figshare Kangning Li 25335076) - FUTURE WORK (pending verified labels)
5. VisA (Visual Anomaly Dataset) - FUTURE WORK (pending license clarity: CC BY vs CC BY-NC-SA)
"""

import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATASETS_INFO = {
    "casting": {
        "name": "Casting Product Image Quality Dataset",
        "target_dir": DATA_DIR / "casting_data",
        "source": "https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product",
        "description": "Submersible pump impeller casting top surface inspection images (ok_front, def_front).",
        "license": "CC0 Public Domain (Kaggle)",
        "status": "EVALUATION_PENDING",
        "expected_subdirs": ["train/ok_front", "train/def_front", "test/ok_front", "test/def_front"],
    },
    "neu": {
        "name": "NEU Surface Defect Database",
        "target_dir": DATA_DIR / "neu_data",
        "source": "https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database",
        "description": "1,800 images of hot-rolled steel strip across 6 defect classes (crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches). Note: No 'normal' class exists.",
        "license": "Academic / Educational",
        "status": "BLOCKED_PENDING_DOWNLOAD",
        "expected_subdirs": ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"],
    },
    "mvtec": {
        "name": "MVTec Anomaly Detection (MVTec AD)",
        "target_dir": DATA_DIR / "mvtec_ad",
        "source": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
        "description": "Industrial anomaly detection benchmarking dataset across 15 objects (metal_nut, screw, transistor, etc.).",
        "license": "CC BY-NC-SA 4.0 (Non-commercial research only - User acceptance required)",
        "status": "BLOCKED_PENDING_DOWNLOAD",
        "expected_subdirs": ["metal_nut/train/good", "metal_nut/test", "screw/train/good", "screw/test"],
    },
    "bearing": {
        "name": "Bearing Defect Dataset (Kangning Li, PLOS ONE 2024)",
        "target_dir": DATA_DIR / "bearing_data",
        "source": "https://figshare.com/articles/dataset/Bearing_defect_dataset/25335076 (DOI: 10.6084/m9.figshare.25335076)",
        "description": "High-speed camera frames from a bearing test rig. Audited and moved to Future Work because uploads lack verified categorical normal/defective annotations (>200 required).",
        "license": "CC BY 4.0 (Figshare)",
        "status": "FUTURE_WORK",
        "expected_subdirs": ["normal", "defective"],
    },
    "visa": {
        "name": "Visual Anomaly (VisA) Dataset",
        "target_dir": DATA_DIR / "visa",
        "source": "https://github.com/amazon-science/spot-diff",
        "description": "10,821 images across 12 industrial objects. Audited and moved to Future Work due to inconsistent licensing reports (CC BY vs CC BY-NC-SA); assumed non-commercial until verified.",
        "license": "Inconsistent (Assume CC BY-NC-SA 4.0 Non-Commercial)",
        "status": "FUTURE_WORK",
        "expected_subdirs": [],
    },
}


def print_status():
    print("=" * 75)
    print("PREDICTGUARD AI - DATASET DIRECTORY & READINESS STATUS")
    print("=" * 75)

    for key, info in DATASETS_INFO.items():
        p = info["target_dir"]
        exists = p.exists()
        file_count = len(list(p.rglob("*.*"))) if exists else 0
        status_str = f"READY ({file_count} files)" if (exists and file_count > 0) else info["status"]

        print(f"\n[{key.upper()}] {info['name']}")
        print(f"  Target Path: {p}")
        print(f"  Status:      {status_str}")
        print(f"  License:     {info['license']}")
        print(f"  Source URL:  {info['source']}")
        print(f"  Scope:       {info['description']}")
        if info["expected_subdirs"]:
            print("  Expected Structure:")
            for sub in info["expected_subdirs"]:
                print(f"    - {info['target_dir'].name}/{sub}/")

    print("\n" + "=" * 75)


def create_scaffolding():
    for key, info in DATASETS_INFO.items():
        if info["status"] != "FUTURE_WORK":
            info["target_dir"].mkdir(parents=True, exist_ok=True)
            for sub in info["expected_subdirs"]:
                (info["target_dir"] / sub).mkdir(parents=True, exist_ok=True)
    print("Created directory scaffolding in data/ for active datasets.")


def main():
    parser = argparse.ArgumentParser(description="PredictGuard AI Dataset Manager")
    parser.add_argument("--create-dirs", action="store_true", help="Create folder scaffolding in data/")
    args = parser.parse_args()

    if args.create_dirs:
        create_scaffolding()
    print_status()


if __name__ == "__main__":
    main()
