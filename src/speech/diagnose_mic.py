"""Check whether microphone recordings look unusual compared with the training data.

The speech SVM was trained on RAVDESS studio recordings. The StandardScaler
stored in svm.joblib holds the per-feature training mean and standard deviation
(fitted on Actor_01..16 only), so the feature vector of any recording can be
compared with the distribution the model saw during training.

Reference: the same comparison for every file of the test split (Actor_21..24),
i.e. unseen speakers recorded in the same studio conditions as the training data.

A feature is "flagged" when its |z-score| exceeds --z-threshold. If microphone
recordings flag far more features than the reference files do, they are unusual
for the model, which is one possible reason for overconfident predictions. This
script cannot prove the cause.

Privacy: recordings stay in memory and are never saved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.common.emotions import CANONICAL_EMOTIONS
from src.speech.data import DEFAULT_DIR, load_ravdess
from src.speech.evaluate import DEFAULT_MODEL_DIR, load_bundle
from src.speech.features import FeatureConfig, extract_features, extract_for_samples
from src.speech.microphone import prepare_waveform, record

DEFAULT_SECONDS = 4.0
MIN_SECONDS = 1.0
MAX_SECONDS = 10.0
DEFAULT_RECORDINGS = 3
MAX_RECORDINGS = 10
DEFAULT_Z_THRESHOLD = 3.0
TOP_FEATURES_SHOWN = 5
QUIT_COMMANDS = {"q", "quit", "exit"}


def spread(values: np.ndarray) -> str:
    """Summarize a set of numbers as min / median / 95th percentile / max."""
    return (
        f"min {float(np.min(values)):.1f}  median {float(np.median(values)):.1f}  "
        f"95th pct {float(np.percentile(values, 95)):.1f}  max {float(np.max(values)):.1f}"
    )


def class_share(class_indices: np.ndarray) -> str:
    counts = np.bincount(class_indices, minlength=len(CANONICAL_EMOTIONS))
    return ", ".join(f"{name} {counts[i]}" for i, name in enumerate(CANONICAL_EMOTIONS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare microphone features with the training distribution."
    )
    parser.add_argument("--recordings", type=int, default=DEFAULT_RECORDINGS,
                        help="Number of microphone recordings (0 = reference only).")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--device", type=int, default=None, help="Input device index.")
    parser.add_argument("--z-threshold", type=float, default=DEFAULT_Z_THRESHOLD)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    if not 0 <= args.recordings <= MAX_RECORDINGS:
        parser.error(f"--recordings must be between 0 and {MAX_RECORDINGS}")
    if not MIN_SECONDS <= args.seconds <= MAX_SECONDS:
        parser.error(f"--seconds must be between {MIN_SECONDS} and {MAX_SECONDS}")
    if args.z_threshold <= 0:
        parser.error("--z-threshold must be positive")
    return args


def main() -> int:
    args = parse_args()
    threshold = args.z_threshold

    try:
        bundle = load_bundle(args.model_dir)
        config = FeatureConfig(**bundle["feature_config"])
        scaler = bundle["scaler"]
        classifier = bundle["classifier"]
        if scaler.mean_.shape != (config.feature_dim,):
            raise ValueError(f"Scaler has {scaler.mean_.shape} features, expected "
                             f"({config.feature_dim},)")
        splits = load_ravdess(args.dir)
        print("Extracting reference features from the test split ...")
        reference_features, _, reference_actors = extract_for_samples(splits.test, config)
    except (FileNotFoundError, ValueError, RuntimeError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    names = config.feature_names()
    reference_z = scaler.transform(reference_features)
    reference_counts = np.sum(np.abs(reference_z) > threshold, axis=1)
    reference_max = np.max(np.abs(reference_z), axis=1)
    reference_predictions = np.argmax(classifier.predict_proba(reference_z), axis=1)

    print(f"\n=== Reference: {len(reference_features)} test files, actors "
          f"{sorted(set(reference_actors.tolist()))} (scaler was fitted on Actor_01..16) ===")
    print(f"Features with |z| > {threshold:g} (out of {config.feature_dim}) per file:")
    print(f"  {spread(reference_counts)}")
    print("Largest |z| per file:")
    print(f"  {spread(reference_max)}")
    print(f"Predicted classes on the reference files: {class_share(reference_predictions)}")

    microphone_counts: list[int] = []
    microphone_predictions: list[int] = []
    if args.recordings == 0:
        print("\nNo microphone recordings requested.")
    else:
        print("\nPrivacy: recordings stay in memory and are NOT saved.")
        print("Suggestion: record a normal sentence, the English sentence "
              "'Kids are talking by the door' in a neutral voice, and some other speech.")

    for number in range(1, args.recordings + 1):
        try:
            answer = input(f"\nRecording {number}/{args.recordings}: press Enter, then speak "
                           f"for {args.seconds:.0f} s (q + Enter to stop): ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if answer.strip().lower() in QUIT_COMMANDS:
            break

        try:
            print("Recording ... speak now.")
            audio, native_rate = record(args.seconds, args.device)
            waveform = prepare_waveform(audio, native_rate, config)
        except ValueError as exc:
            print(f"Recording {number} skipped: {exc}")
            continue
        except Exception as exc:  # no device, permission, ...
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

        features = extract_features(waveform, config)
        z_row = scaler.transform(features.reshape(1, -1))
        probabilities = classifier.predict_proba(z_row)[0]
        z_values = z_row[0]
        best = int(np.argmax(probabilities))
        flagged = int(np.sum(np.abs(z_values) > threshold))
        microphone_counts.append(flagged)
        microphone_predictions.append(best)

        print(f"\n--- Recording {number} ---")
        print(f"  speech after trimming : {waveform.size / config.sample_rate:.2f} s")
        print(f"  prediction            : {CANONICAL_EMOTIONS[best]} {probabilities[best]:.2f}")
        print(f"  features with |z| > {threshold:g} : {flagged} of {config.feature_dim}  "
              f"(reference median {float(np.median(reference_counts)):.0f}, "
              f"95th pct {float(np.percentile(reference_counts, 95)):.0f})")
        print(f"  largest |z|           : {float(np.max(np.abs(z_values))):.1f}  "
              f"(reference median {float(np.median(reference_max)):.1f})")
        print(f"  {TOP_FEATURES_SHOWN} most unusual features:")
        for index in np.argsort(-np.abs(z_values))[:TOP_FEATURES_SHOWN]:
            print(f"    {names[index]:<10} value {features[index]:>9.3f}  "
                  f"train mean {scaler.mean_[index]:>9.3f}  z {z_values[index]:>7.2f}")

    if microphone_counts:
        print("\n=== Summary ===")
        print(f"Reference files, features flagged per file : {spread(reference_counts)}")
        print(f"Microphone recordings, features flagged    : {microphone_counts}")
        print(f"Reference predictions : {class_share(reference_predictions)}")
        print(f"Microphone predictions: "
              f"{class_share(np.array(microphone_predictions, dtype=np.int64))}")
        print("Note: far more flagged features than the reference range means the recordings "
              "are unusual for the model. That is one possible reason for overconfident "
              "predictions, not proof.")
    return 0


if __name__ == "__main__":
    sys.exit(main())