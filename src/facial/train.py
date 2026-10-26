
"""Train the facial emotion CNN on FER2013 (5 canonical classes).

Uses Training for fitting and PublicTest for validation. The PrivateTest
split is deliberately NOT touched here; it is reserved for evaluate.py.

Artifacts written to --output-dir:
    best.keras           best checkpoint by validation loss
    history.csv          per-epoch metrics
    label_mapping.json   class order and FER2013 mapping/exclusions
    training_config.json hyperparameters, split sizes, seed, library versions
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.utils.class_weight import compute_class_weight

from src.common.emotions import (
    CANONICAL_EMOTIONS,
    FER2013_EXCLUDED,
    FER2013_TO_CANONICAL,
)
from src.facial.data import DEFAULT_CSV, FacialSplit, load_fer2013
from src.facial.model import build_model

import tensorflow as tf  # noqa: E402  (after model import, which sets TF log level)
from tensorflow import keras  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "facial"

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 60
DEFAULT_BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 4
MIN_LEARNING_RATE = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the facial emotion CNN.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Use a random subset of the training split (for quick smoke runs only).",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be at least 1")
    if args.max_samples is not None and args.max_samples < args.batch_size:
        parser.error("--max-samples must be at least --batch-size")
    return args


def subsample(split: FacialSplit, max_samples: int, seed: int) -> FacialSplit:
    """Return a random subset of a split (used only for smoke runs)."""
    if max_samples >= len(split.labels):
        return split
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(split.labels), size=max_samples, replace=False))
    return FacialSplit(images=split.images[chosen], labels=split.labels[chosen])


def class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels, minlength=len(CANONICAL_EMOTIONS))
    return {name: int(counts[i]) for i, name in enumerate(CANONICAL_EMOTIONS)}


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_history_csv(path: Path, history: dict[str, list[float]]) -> None:
    keys = list(history)
    epochs = len(history[keys[0]])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch"] + keys)
        for epoch in range(epochs):
            writer.writerow([epoch + 1] + [float(history[key][epoch]) for key in keys])


def main() -> int:
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    print("Loading FER2013 ...")
    try:
        dataset = load_fer2013(args.csv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    train = dataset.train
    if args.max_samples is not None:
        train = subsample(train, args.max_samples, args.seed)
        print(f"WARNING: smoke run, using only {len(train.labels)} training samples.")
    validation = dataset.validation

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CANONICAL_EMOTIONS)),
        y=train.labels,
    )
    class_weight = {index: float(weight) for index, weight in enumerate(weights)}
    print("Class weights (balanced):")
    for index, name in enumerate(CANONICAL_EMOTIONS):
        print(f"  {index} {name:<8} {class_weight[index]:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.keras"

    model = build_model(seed=args.seed)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1,
        ),
    ]

    start = time.perf_counter()
    history = model.fit(
        train.images,
        train.labels,
        validation_data=(validation.images, validation.labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.perf_counter() - start

    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    write_history_csv(args.output_dir / "history.csv", history_dict)

    write_json(
        args.output_dir / "label_mapping.json",
        {
            "class_order": list(CANONICAL_EMOTIONS),
            "index_to_emotion": {str(i): n for i, n in enumerate(CANONICAL_EMOTIONS)},
            "fer2013_to_canonical": {str(k): v for k, v in FER2013_TO_CANONICAL.items()},
            "fer2013_excluded": {str(k): v for k, v in FER2013_EXCLUDED.items()},
        },
    )

    best_epoch = int(np.argmin(history_dict["val_loss"])) + 1
    write_json(
        args.output_dir / "training_config.json",
        {
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "epochs_run": len(history_dict["val_loss"]),
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "learning_rate_initial": LEARNING_RATE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "smoke_run_max_samples": args.max_samples,
            "split_policy": {
                "train": "Training",
                "validation": "PublicTest",
                "test": "PrivateTest (not used in this script)",
            },
            "train_samples": int(len(train.labels)),
            "validation_samples": int(len(validation.labels)),
            "train_class_counts": class_counts(train.labels),
            "validation_class_counts": class_counts(validation.labels),
            "class_weights": {CANONICAL_EMOTIONS[i]: w for i, w in class_weight.items()},
            "best_epoch_by_val_loss": best_epoch,
            "training_seconds": round(elapsed, 1),
            "versions": {
                "python": platform.python_version(),
                "tensorflow": tf.__version__,
                "keras": keras.__version__,
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
    )

    print("\n=== Training finished ===")
    print(f"Epochs run            : {len(history_dict['val_loss'])}")
    print(f"Training time         : {elapsed:.1f} s")
    print(f"Best epoch (val_loss) : {best_epoch}")
    print(f"  val_loss            : {history_dict['val_loss'][best_epoch - 1]:.4f}")
    print(f"  val_accuracy        : {history_dict['val_accuracy'][best_epoch - 1]:.4f}")
    print(f"Artifacts saved in    : {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())