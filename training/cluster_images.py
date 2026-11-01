"""
PredictGuard AI - NEU Image Embedding Clustering (Practical 06).

Applies unsupervised clustering on MobileNetV2 1,280-D dense image embeddings:
1. K-Means Clustering across k in [2..8]:
   - Computes inertia / sum of squared errors (SSE) for elbow analysis (highlighting k=3 and k=4).
   - Computes silhouette scores to identify optimal cluster separation.
2. Hierarchical (Agglomerative) Clustering using Ward's minimum variance linkage.
3. Compares unsupervised cluster structure against ground-truth defect taxonomies.

Outputs metrics to docs/metrics/clustering_neu.json.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from training.metrics_io import get_git_sha, get_library_versions

DATA_DIR = BASE_DIR / "data" / "neu_data"
SPLITS_FILE = BASE_DIR / "data" / "neu_splits.json"
OUTPUT_METRICS = BASE_DIR / "docs" / "metrics" / "clustering_neu.json"


def extract_embeddings(image_paths: List[str], batch_size: int = 32) -> np.ndarray:
    feature_extractor = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    def load_img(path_str):
        b = tf.io.read_file(path_str)
        img = tf.image.decode_image(b, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224], method="nearest")
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img

    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    embeddings = feature_extractor.predict(ds, verbose=0)
    return np.asarray(embeddings)


def main():
    print("=" * 70)
    print("NEU IMAGE EMBEDDING CLUSTERING & ELBOW / SILHOUETTE (PRACTICAL 06)")
    print("=" * 70)

    if not SPLITS_FILE.exists() or not DATA_DIR.exists():
        print("\nSTATUS: BLOCKED - NEU dataset or splits missing.")
        print(f"Expected data at {DATA_DIR} and splits at {SPLITS_FILE}.")
        OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
        if not OUTPUT_METRICS.exists():
            with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "status": "BLOCKED_PENDING_NEU_DATASET",
                        "message": "Requires data/neu_data and data/neu_splits.json. See DATASETS.md.",
                        "kmeans": {},
                        "hierarchical": {},
                    },
                    f,
                    indent=2,
                )
        sys.exit(2)

    with open(SPLITS_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Use test images for clustering evaluation
    sample_paths = splits["test"]
    print(f"Extracting embeddings for {len(sample_paths)} evaluation images...")
    X = extract_embeddings(sample_paths)

    k_range = list(range(2, 9))
    kmeans_results: Dict[str, Dict[str, float]] = {}
    best_k = 2
    best_sil = -1.0

    print("\nRunning K-Means sweep across k in [2..8]...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sse = float(km.inertia_)
        sil = float(silhouette_score(X, labels))

        kmeans_results[f"k_{k}"] = {
            "k": k,
            "inertia_sse": round(sse, 2),
            "silhouette_score": round(sil, 4),
        }
        print(f"  k={k} | SSE (Inertia): {sse:10.2f} | Silhouette Score: {sil:.4f}")

        if sil > best_sil:
            best_sil = sil
            best_k = k

    # Highlight k=3 and k=4 comparison per instructions
    k3_sse = kmeans_results["k_3"]["inertia_sse"]
    k4_sse = kmeans_results["k_4"]["inertia_sse"]
    sse_reduction_k3_k4 = round(float((k3_sse - k4_sse) / k3_sse * 100.0), 2)

    # Hierarchical Clustering comparison with optimal k and k=6 (ground-truth defect classes)
    print("\nFitting Agglomerative Hierarchical Clustering (Ward linkage)...")
    hier_opt = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hier_opt_labels = hier_opt.fit_predict(X)
    hier_opt_sil = float(silhouette_score(X, hier_opt_labels))

    hier_6 = AgglomerativeClustering(n_clusters=6, linkage="ward")
    hier_6_labels = hier_6.fit_predict(X)
    hier_6_sil = float(silhouette_score(X, hier_6_labels))

    payload = {
        "status": "COMPLETED",
        "dataset": "NEU Surface Defect Database",
        "sample_count": len(sample_paths),
        "embedding_dimensions": int(X.shape[1]),
        "git_sha": get_git_sha(),
        "library_versions": get_library_versions(),
        "kmeans": {
            "sweep": kmeans_results,
            "k3_sse": k3_sse,
            "k4_sse": k4_sse,
            "sse_reduction_k3_to_k4_percent": sse_reduction_k3_k4,
            "optimal_k_by_silhouette": best_k,
            "optimal_silhouette": round(best_sil, 4),
        },
        "hierarchical": {
            "linkage": "ward",
            f"silhouette_at_optimal_k_{best_k}": round(hier_opt_sil, 4),
            "silhouette_at_k_6": round(hier_6_sil, 4),
        },
    }

    OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved clustering metrics to: {OUTPUT_METRICS}")


if __name__ == "__main__":
    main()
