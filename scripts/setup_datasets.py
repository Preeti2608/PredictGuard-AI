"""
PredictGuard AI - Multi-Dataset Setup and Verification Tool.
Helps organize, verify, and document datasets for:
1. Casting Quality Dataset (Submersible pump impellers)
2. Bearing Defect Dataset (Figshare Kangning Li 25335076)
3. NEU Surface Defect Database (Kaggle 6-class steel strip)
4. MVTec Anomaly Detection (MVTec AD)
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
        "expected_subdirs": ["train/ok_front", "train/def_front", "test/ok_front", "test/def_front"],
    },
    "bearing": {
        "name": "Bearing Defect Dataset (Kangning Li, PLOS ONE 2024)",
        "target_dir": DATA_DIR / "bearing_data",
        "source": "https://figshare.com/articles/dataset/Bearing_defect_dataset/25335076 (DOI: 10.6084/m9.figshare.25335076)",
        "description": "High-resolution inspection images of ball & roller bearings (normal, surface spalling, raceway cracks, fatigue wear).",
        "expected_subdirs": ["normal", "defective"],
    },
    "neu": {
        "name": "NEU Surface Defect Database",
        "target_dir": DATA_DIR / "neu_data",
        "source": "https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database",
        "description": "1,800 images of hot-rolled steel strip with 6 defect types: crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches.",
        "expected_subdirs": ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"],
    },
    "mvtec": {
        "name": "MVTec Anomaly Detection (MVTec AD)",
        "target_dir": DATA_DIR / "mvtec_ad",
        "source": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
        "description": "5,354 industrial images across 15 objects (metal nut, screw, capsule, bottle, grid, etc.).",
        "expected_subdirs": ["metal_nut", "screw"],
    },
}


def print_status():
    print("=" * 70)
    print("PREDICTGUARD AI - DATASET DIRECTORY STATUS")
    print("=" * 70)

    for key, info in DATASETS_INFO.items():
        p = info["target_dir"]
        exists = p.exists()
        file_count = len(list(p.rglob("*.*"))) if exists else 0
        status_str = f"READY ({file_count} files)" if (exists and file_count > 0) else "MISSING / EMPTY"

        print(f"\n[{key.upper()}] {info['name']}")
        print(f"  Target Path: {p}")
        print(f"  Status:      {status_str}")
        print(f"  Source URL:  {info['source']}")
        print(f"  Scope:       {info['description']}")
        print("  Expected Structure:")
        for sub in info["expected_subdirs"]:
            print(f"    - {info['target_dir'].name}/{sub}/")

    print("\n" + "=" * 70)
    print("INSTRUCTIONS TO ADD YOUR DATASET:")
    print("1. Download your dataset archive from the links above.")
    print("2. Extract the contents into the respective target path inside data/.")
    print("3. Run the training script for the dataset:")
    print("     python training/train_bearing.py   (for Bearing Dataset)")
    print("     python training/train_neu.py       (for NEU Surface Defect Database)")
    print("     python training/train_image.py     (for Pump Impeller Casting Dataset)")
    print("=" * 70)


def create_scaffolding():
    for key, info in DATASETS_INFO.items():
        info["target_dir"].mkdir(parents=True, exist_ok=True)
        for sub in info["expected_subdirs"]:
            (info["target_dir"] / sub).mkdir(parents=True, exist_ok=True)
    print("Created directory scaffolding in data/ for all supported datasets.")


def main():
    parser = argparse.ArgumentParser(description="PredictGuard AI Dataset Manager")
    parser.add_argument("--create-dirs", action="store_true", help="Create folder scaffolding in data/")
    args = parser.parse_args()

    if args.create_dirs:
        create_scaffolding()
    print_status()


if __name__ == "__main__":
    main()
