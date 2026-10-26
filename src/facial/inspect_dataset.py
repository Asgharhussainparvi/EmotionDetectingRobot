"""Read-only inspection of the FER2013 CSV dataset.

Prints column names, row counts per emotion label and per Usage split,
a label-by-split cross table, and pixel-data sanity checks.
Nothing is modified or written to disk.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "datasets" / "facial" / "fer2013.csv"

EXPECTED_COLUMNS = ["emotion", "pixels", "Usage"]
IMAGE_SIDE = 48
EXPECTED_PIXEL_COUNT = IMAGE_SIDE * IMAGE_SIDE
MAX_BAD_ROWS_SHOWN = 5

# Label order as commonly documented for FER2013. This mapping is NOT stored
# inside the CSV, so it is an assumption we must verify (e.g. by viewing
# sample images per label) before relying on it for training.
ASSUMED_LABEL_NAMES: dict[int, str] = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


def inspect_csv(csv_path: Path) -> None:
    """Scan the CSV once and print an inspection report."""
    emotion_counts: Counter[int] = Counter()
    usage_counts: Counter[str] = Counter()
    cross_counts: Counter[tuple[int, str]] = Counter()
    bad_rows: list[tuple[int, str]] = []
    total_rows = 0
    pixel_min = 255
    pixel_max = 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        print(f"File           : {csv_path}")
        print(f"Header         : {header}")
        if header != EXPECTED_COLUMNS:
            print(f"WARNING: expected columns {EXPECTED_COLUMNS}")
            print("Stopping, because the layout differs from what this script supports.")
            return

        for line_number, row in enumerate(reader, start=2):
            total_rows += 1
            if len(row) != len(EXPECTED_COLUMNS):
                bad_rows.append((line_number, f"expected 3 fields, got {len(row)}"))
                continue
            emotion_text, pixels_text, usage = row
            try:
                emotion = int(emotion_text)
                pixel_values = [int(value) for value in pixels_text.split()]
            except ValueError as exc:
                bad_rows.append((line_number, f"non-integer value: {exc}"))
                continue
            if len(pixel_values) != EXPECTED_PIXEL_COUNT:
                bad_rows.append(
                    (line_number, f"{len(pixel_values)} pixels, expected {EXPECTED_PIXEL_COUNT}")
                )
                continue

            emotion_counts[emotion] += 1
            usage_counts[usage] += 1
            cross_counts[(emotion, usage)] += 1
            pixel_min = min(pixel_min, min(pixel_values))
            pixel_max = max(pixel_max, max(pixel_values))

    valid_rows = sum(emotion_counts.values())
    print(f"\nTotal data rows : {total_rows}")
    print(f"Valid rows      : {valid_rows}")
    print(f"Bad rows        : {len(bad_rows)}")
    for line_number, reason in bad_rows[:MAX_BAD_ROWS_SHOWN]:
        print(f"  line {line_number}: {reason}")

    print("\n--- Rows per emotion label (names are ASSUMED, see script comment) ---")
    for label in sorted(emotion_counts):
        name = ASSUMED_LABEL_NAMES.get(label, "UNKNOWN LABEL")
        count = emotion_counts[label]
        percent = 100.0 * count / valid_rows
        print(f"{label}  {name:<9} {count:>6}  ({percent:5.1f}%)")

    print("\n--- Rows per Usage split ---")
    for usage in sorted(usage_counts):
        print(f"{usage:<12} {usage_counts[usage]:>6}")

    print("\n--- Label x Usage ---")
    usages = sorted(usage_counts)
    print(f"{'label':<14}" + "".join(f"{usage:>12}" for usage in usages))
    for label in sorted(emotion_counts):
        name = ASSUMED_LABEL_NAMES.get(label, "UNKNOWN")
        cells = "".join(f"{cross_counts[(label, usage)]:>12}" for usage in usages)
        print(f"{label} {name:<12}{cells}")

    print("\n--- Pixel checks ---")
    print(f"Pixels per image (expected) : {EXPECTED_PIXEL_COUNT} ({IMAGE_SIDE}x{IMAGE_SIDE})")
    print(f"Pixel value range found     : {pixel_min} to {pixel_max}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the FER2013 CSV file.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to fer2013.csv")
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"ERROR: file not found: {args.csv}", file=sys.stderr)
        return 1
    inspect_csv(args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())