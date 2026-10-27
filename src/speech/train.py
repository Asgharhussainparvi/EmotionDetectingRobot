"""Train an SVM baseline for speech emotion recognition (RAVDESS, 5 classes).

Fitting uses the train split (Actor_01..16). Hyperparameters are chosen by
macro-F1 on the validation split (Actor_17..20). The test split (Actor_21..24)
is deliberately NOT touched here; it is reserved for evaluate.py.

The StandardScaler is fitted on the training features only.

Artifacts written to --output-dir:
    svm.joblib           scaler, classifier, feature config and class order
    label_mapping.json   class order and RAVDESS mapping/exclusions
    grid_results.csv     validation scores of every hyperparameter combination
    training_config.json seed, grid, chosen settings, split sizes, versions
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import librosa
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.common.emotions import (
    CANONICAL_EMOTIONS,
    RAVDESS_EXCLUDED,
    RAVDESS_TO_CANONICAL,
)
from src.speech.data import DEFAULT_DIR, load_ravdess
from src.speech.features import FeatureConfig, extract_for_samples

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "speech"

DEFAULT_SEED = 42
C_GRID = (0.1, 1.0, 10.0, 100.0)
GAMMA_GRID = ("scale", 0.001, 0.01)
NUM_CLASSES = len(CANONICAL_EMOTIONS)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    return {name: int(counts[i]) for i, name in enumerate(CANONICAL_EMOTIONS)}


def search_hyperparameters(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> list[dict]:
    """Fit one SVM per grid point and score it on the validation split."""
    rows: list[dict] = []
    for c_value, gamma in itertools.product(C_GRID, GAMMA_GRID):
        classifier = SVC(kernel="rbf", C=c_value, gamma=gamma, class_weight="balanced")
        classifier.fit(train_x, train_y)
        predictions = classifier.predict(validation_x)
        rows.append(
            {
                "C": c_value,
                "gamma": gamma,
                "val_accuracy": float(accuracy_score(validation_y, predictions)),
                "val_macro_f1": float(
                    f1_score(validation_y, predictions, average="macro", zero_division=0)
                ),
            }
        )
    return rows


def pick_best(rows: list[dict]) -> dict:
    """Highest validation macro-F1; the first grid point wins ties."""
    best = rows[0]
    for row in rows[1:]:
        if row["val_macro_f1"] > best["val_macro_f1"]:
            best = row
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the speech emotion SVM.")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    start = time.perf_counter()
    config = FeatureConfig()
    try:
        splits = load_ravdess(args.dir)
        print("Extracting features (train and validation only) ...")
        train_x, train_y, train_actors = extract_for_samples(splits.train, config)
        validation_x, validation_y, validation_actors = extract_for_samples(
            splits.validation, config
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if set(train_actors.tolist()) & set(validation_actors.tolist()):
        print("ERROR: an actor appears in both train and validation.", file=sys.stderr)
        return 1
    print(f"Train: {train_x.shape}, validation: {validation_x.shape}")

    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    validation_scaled = scaler.transform(validation_x)

    print(f"Searching {len(C_GRID) * len(GAMMA_GRID)} hyperparameter combinations ...")
    rows = search_hyperparameters(train_scaled, train_y, validation_scaled, validation_y)
    best = pick_best(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "grid_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["C", "gamma", "val_accuracy", "val_macro_f1"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'C':>8} {'gamma':>8} {'val_acc':>9} {'val_macroF1':>12}")
    for row in rows:
        marker = "  <- chosen" if row is best else ""
        print(f"{row['C']:>8} {str(row['gamma']):>8} {row['val_accuracy']:>9.4f} "
              f"{row['val_macro_f1']:>12.4f}{marker}")

    classifier = SVC(
        kernel="rbf",
        C=best["C"],
        gamma=best["gamma"],
        class_weight="balanced",
        probability=True,
        random_state=args.seed,
    )
    classifier.fit(train_scaled, train_y)
    if not np.array_equal(classifier.classes_, np.arange(NUM_CLASSES)):
        print(f"ERROR: unexpected classes in training data: {classifier.classes_}",
              file=sys.stderr)
        return 1

    # Same decision rule that inference will use: argmax of predict_proba.
    probabilities = classifier.predict_proba(validation_scaled)
    predictions = np.argmax(probabilities, axis=1)
    final_accuracy = float(accuracy_score(validation_y, predictions))
    final_macro_f1 = float(f1_score(validation_y, predictions, average="macro", zero_division=0))

    joblib.dump(
        {
            "scaler": scaler,
            "classifier": classifier,
            "feature_config": config.to_dict(),
            "class_order": list(CANONICAL_EMOTIONS),
        },
        args.output_dir / "svm.joblib",
    )
    write_json(
        args.output_dir / "label_mapping.json",
        {
            "class_order": list(CANONICAL_EMOTIONS),
            "index_to_emotion": {str(i): n for i, n in enumerate(CANONICAL_EMOTIONS)},
            "ravdess_to_canonical": {str(k): v for k, v in RAVDESS_TO_CANONICAL.items()},
            "ravdess_excluded": {str(k): v for k, v in RAVDESS_EXCLUDED.items()},
        },
    )
    elapsed = time.perf_counter() - start
    write_json(
        args.output_dir / "training_config.json",
        {
            "seed": args.seed,
            "model": "SVC (rbf kernel, class_weight=balanced, probability=True)",
            "scaler": "StandardScaler fitted on the training split only",
            "feature_config": config.to_dict(),
            "grid": {"C": list(C_GRID), "gamma": list(GAMMA_GRID)},
            "selection_rule": "highest validation macro-F1, first grid point wins ties",
            "chosen": {"C": best["C"], "gamma": best["gamma"]},
            "validation_metrics_final_model": {
                "accuracy": final_accuracy,
                "macro_f1": final_macro_f1,
                "decision_rule": "argmax of predict_proba",
            },
            "split_policy": {
                "train": "Actor_01..16",
                "validation": "Actor_17..20",
                "test": "Actor_21..24 (not used in this script)",
            },
            "train_samples": int(len(train_y)),
            "validation_samples": int(len(validation_y)),
            "train_class_counts": class_counts(train_y),
            "validation_class_counts": class_counts(validation_y),
            "total_seconds": round(elapsed, 1),
            "versions": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "numpy": np.__version__,
                "librosa": librosa.__version__,
            },
        },
    )

    print("\n=== Training finished ===")
    print(f"Chosen            : C={best['C']}, gamma={best['gamma']}")
    print(f"Validation (final): accuracy={final_accuracy:.4f}, macro-F1={final_macro_f1:.4f}")
    print(f"Total time        : {elapsed:.1f} s")
    print(f"Artifacts saved in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())