"""Save a grid of random FER2013 samples per label for visual verification.

The CSV stores numeric labels only. The names used here are the commonly
documented order and must be checked by eye before training.
Output: results/facial/label_preview.png
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI window; we only save a file
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "datasets" / "facial" / "fer2013.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "facial" / "label_preview.png"

IMAGE_SIDE = 48
SAMPLES_PER_LABEL = 10
RANDOM_SEED = 42

ASSUMED_LABEL_NAMES: dict[int, str] = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


def sample_pixels_per_label(
    csv_path: Path, samples_per_label: int, seed: int
) -> dict[int, list[list[int]]]:
    """Reservoir-sample pixel rows per label so memory stays small."""
    rng = random.Random(seed)
    seen: Counter[int] = Counter()
    reservoirs: dict[int, list[list[int]]] = {label: [] for label in ASSUMED_LABEL_NAMES}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # skip header
        for row in reader:
            label = int(row[0])
            if label not in reservoirs:
                raise ValueError(f"Unexpected label {label} in CSV")
            seen[label] += 1
            pixels = [int(value) for value in row[1].split()]
            bucket = reservoirs[label]
            if len(bucket) < samples_per_label:
                bucket.append(pixels)
            else:
                slot = rng.randrange(seen[label])
                if slot < samples_per_label:
                    bucket[slot] = pixels
    return reservoirs


def save_grid(reservoirs: dict[int, list[list[int]]], output_path: Path) -> None:
    """Draw one row per label and save the figure."""
    rows = len(reservoirs)
    cols = SAMPLES_PER_LABEL
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    for row_index, label in enumerate(sorted(reservoirs)):
        for col_index in range(cols):
            axis = axes[row_index][col_index]
            axis.set_xticks([])
            axis.set_yticks([])
            samples = reservoirs[label]
            if col_index < len(samples):
                image = np.array(samples[col_index], dtype=np.uint8).reshape(
                    IMAGE_SIDE, IMAGE_SIDE
                )
                axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[row_index][0].set_ylabel(
            f"{label}\n{ASSUMED_LABEL_NAMES[label]}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
        )
    fig.suptitle("FER2013 samples per label (names are ASSUMED, verify by eye)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Save label preview grid for FER2013.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"ERROR: file not found: {args.csv}", file=sys.stderr)
        return 1
    reservoirs = sample_pixels_per_label(args.csv, SAMPLES_PER_LABEL, RANDOM_SEED)
    save_grid(reservoirs, args.output)
    print(f"Saved preview image: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())