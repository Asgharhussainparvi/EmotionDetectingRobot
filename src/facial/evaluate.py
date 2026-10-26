"""Evaluate the trained facial emotion CNN on a FER2013 split.

Reports accuracy, macro-F1, per-class precision/recall/F1 and a confusion
matrix. Defaults to the PrivateTest split ("test"). Use --split validation
for a dry run that does not touch the test data.

Outputs (in --results-dir):
    <split>_metrics.json
    <split>_confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save files only; no GUI window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tensorflow import keras  # noqa: E402

from src.common.emotions import CANONICAL_EMOTIONS  # noqa: E402
from src.facial.data import DEFAULT_CSV, load_fer2013  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "facial"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "facial"
PREDICT_BATCH_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the facial emotion CNN.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    return parser.parse_args()


def check_class_order(model_dir: Path) -> None:
    """Make sure the saved label mapping matches the current canonical order."""
    mapping_path = model_dir / "label_mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Label mapping not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as handle:
        saved_order = json.load(handle)["class_order"]
    if saved_order != list(CANONICAL_EMOTIONS):
        raise ValueError(
            f"Class order mismatch. Saved: {saved_order}, current: {list(CANONICAL_EMOTIONS)}"
        )


def save_confusion_matrix_plot(matrix: np.ndarray, split: str, output_path: Path) -> None:
    """Draw a row-normalized confusion matrix annotated with counts."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros(matrix.shape), where=row_sums > 0)

    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=axis, label="Fraction of true class")
    ticks = np.arange(len(CANONICAL_EMOTIONS))
    axis.set_xticks(ticks, labels=CANONICAL_EMOTIONS, rotation=45, ha="right")
    axis.set_yticks(ticks, labels=CANONICAL_EMOTIONS)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"Facial CNN confusion matrix ({split} split)")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if normalized[row, col] > 0.5 else "black"
            axis.text(
                col,
                row,
                f"{matrix[row, col]}\n{100 * normalized[row, col]:.0f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    model_path = args.model_dir / "best.keras"

    try:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        check_class_order(args.model_dir)
        dataset = load_fer2013(args.csv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    split = dataset.test if args.split == "test" else dataset.validation
    print(f"Evaluating {model_path} on the '{args.split}' split ({len(split.labels)} samples)")

    model = keras.models.load_model(model_path)
    probabilities = model.predict(split.images, batch_size=PREDICT_BATCH_SIZE, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    class_indices = list(range(len(CANONICAL_EMOTIONS)))
    accuracy = float(accuracy_score(split.labels, predictions))
    macro_f1 = float(f1_score(split.labels, predictions, labels=class_indices,
                              average="macro", zero_division=0))
    report = classification_report(
        split.labels,
        predictions,
        labels=class_indices,
        target_names=list(CANONICAL_EMOTIONS),
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(split.labels, predictions, labels=class_indices)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(f"\n{'class':<9}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    for name in CANONICAL_EMOTIONS:
        row = report[name]
        print(f"{name:<9}{row['precision']:>10.4f}{row['recall']:>9.4f}"
              f"{row['f1-score']:>8.4f}{int(row['support']):>9}")
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(f"{'':<9}" + "".join(f"{name:>9}" for name in CANONICAL_EMOTIONS))
    for index, name in enumerate(CANONICAL_EMOTIONS):
        print(f"{name:<9}" + "".join(f"{value:>9}" for value in matrix[index]))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / f"{args.split}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split": args.split,
                "model_path": str(model_path.relative_to(PROJECT_ROOT)
                                  if model_path.is_relative_to(PROJECT_ROOT) else model_path),
                "samples": int(len(split.labels)),
                "class_order": list(CANONICAL_EMOTIONS),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "per_class": {name: report[name] for name in CANONICAL_EMOTIONS},
                "confusion_matrix_rows_true_cols_pred": matrix.tolist(),
            },
            handle,
            indent=2,
        )
    plot_path = args.results_dir / f"{args.split}_confusion_matrix.png"
    save_confusion_matrix_plot(matrix, args.split, plot_path)

    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())