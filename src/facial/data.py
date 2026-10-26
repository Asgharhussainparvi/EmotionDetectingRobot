"""Load FER2013 from CSV into train/validation/test arrays.

Only the five canonical emotions are kept (see src/common/emotions.py).
Rows with excluded labels are counted and skipped, never relabelled.
Images are returned as uint8 in [0, 255]; scaling to [0, 1] happens inside
the model so training and inference share the same normalization.

Split policy (dataset-provided Usage column):
    Training    -> train
    PublicTest  -> validation
    PrivateTest -> test
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.common.emotions import CANONICAL_EMOTIONS, fer2013_label_to_index

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "datasets" / "facial" / "fer2013.csv"

IMAGE_SIDE = 48
PIXELS_PER_IMAGE = IMAGE_SIDE * IMAGE_SIDE
EXPECTED_HEADER = ["emotion", "pixels", "Usage"]
USAGE_TO_SPLIT: dict[str, str] = {
    "Training": "train",
    "PublicTest": "validation",
    "PrivateTest": "test",
}


@dataclass(frozen=True)
class FacialSplit:
    """Images shaped (N, 48, 48, 1) uint8 and canonical class indices shaped (N,)."""

    images: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class FacialDataset:
    """The three splits plus the number of rows skipped by design."""

    train: FacialSplit
    validation: FacialSplit
    test: FacialSplit
    excluded_rows: int


def load_fer2013(csv_path: Path = DEFAULT_CSV) -> FacialDataset:
    """Parse the FER2013 CSV and return the three splits."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"FER2013 CSV not found: {csv_path}")

    images: dict[str, list[np.ndarray]] = {name: [] for name in USAGE_TO_SPLIT.values()}
    labels: dict[str, list[int]] = {name: [] for name in USAGE_TO_SPLIT.values()}
    excluded_rows = 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != EXPECTED_HEADER:
            raise ValueError(f"Unexpected CSV header {header}, expected {EXPECTED_HEADER}")

        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(EXPECTED_HEADER):
                raise ValueError(f"Line {line_number}: expected 3 fields, got {len(row)}")
            emotion_text, pixels_text, usage = row

            if usage not in USAGE_TO_SPLIT:
                raise ValueError(f"Line {line_number}: unknown Usage value '{usage}'")

            class_index = fer2013_label_to_index(int(emotion_text))
            if class_index is None:
                excluded_rows += 1
                continue

            pixel_values = np.array([int(v) for v in pixels_text.split()], dtype=np.uint8)
            if pixel_values.size != PIXELS_PER_IMAGE:
                raise ValueError(
                    f"Line {line_number}: {pixel_values.size} pixels, "
                    f"expected {PIXELS_PER_IMAGE}"
                )

            split_name = USAGE_TO_SPLIT[usage]
            images[split_name].append(pixel_values)
            labels[split_name].append(class_index)

    def build(split_name: str) -> FacialSplit:
        stacked = np.stack(images[split_name]).reshape(-1, IMAGE_SIDE, IMAGE_SIDE, 1)
        return FacialSplit(images=stacked, labels=np.array(labels[split_name], dtype=np.int64))

    return FacialDataset(
        train=build("train"),
        validation=build("validation"),
        test=build("test"),
        excluded_rows=excluded_rows,
    )


def main() -> int:
    start = time.perf_counter()
    try:
        dataset = load_fer2013()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start

    print(f"Loaded in {elapsed:.1f} s")
    print(f"Excluded rows (Disgust/Surprise): {dataset.excluded_rows}")
    for name, split in (
        ("train", dataset.train),
        ("validation", dataset.validation),
        ("test", dataset.test),
    ):
        print(f"\n[{name}]")
        print(f"  images : shape={split.images.shape} dtype={split.images.dtype} "
              f"min={split.images.min()} max={split.images.max()}")
        print(f"  labels : shape={split.labels.shape} dtype={split.labels.dtype}")
        counts = np.bincount(split.labels, minlength=len(CANONICAL_EMOTIONS))
        for index, emotion in enumerate(CANONICAL_EMOTIONS):
            print(f"    {index} {emotion:<8} {counts[index]:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())