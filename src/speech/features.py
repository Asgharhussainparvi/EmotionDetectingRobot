"""Acoustic feature extraction for speech emotion recognition.

Each audio file is converted to one fixed-length vector:
    mean and std over time of N_MFCC MFCCs  -> 2 * n_mfcc values
    mean and std over time of RMS energy    -> 2 values

All settings live in FeatureConfig so the exact same configuration can be
saved with the model and reused at inference time. No feature scaling is done
here; scaling must be fitted on the training split only (see the training
script).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np

from src.common.emotions import CANONICAL_EMOTIONS
from src.speech.data import SpeechSample, load_ravdess


@dataclass(frozen=True)
class FeatureConfig:
    """Settings that fully determine how audio becomes a feature vector."""

    sample_rate: int = 16000
    n_mfcc: int = 40
    n_fft: int = 1024  # 64 ms window at 16 kHz
    hop_length: int = 256  # 16 ms hop at 16 kHz
    trim_top_db: float = 30.0  # silence trimming threshold

    @property
    def feature_dim(self) -> int:
        return 2 * self.n_mfcc + 2

    def feature_names(self) -> list[str]:
        names = [f"mfcc{i + 1}_mean" for i in range(self.n_mfcc)]
        names += [f"mfcc{i + 1}_std" for i in range(self.n_mfcc)]
        return names + ["rms_mean", "rms_std"]

    def to_dict(self) -> dict:
        return asdict(self)


def load_audio(path: Path, config: FeatureConfig) -> np.ndarray:
    """Load a file as mono float audio at config.sample_rate with silence trimmed."""
    waveform, _ = librosa.load(str(path), sr=config.sample_rate, mono=True)
    if waveform.size == 0:
        raise ValueError(f"Empty audio: {path}")
    trimmed, _ = librosa.effects.trim(waveform, top_db=config.trim_top_db)
    if trimmed.size < config.n_fft:
        raise ValueError(f"Audio too short after trimming: {path}")
    return trimmed


def extract_features(waveform: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Convert a mono waveform (already at config.sample_rate) into a feature vector."""
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    rms = librosa.feature.rms(
        y=waveform, frame_length=config.n_fft, hop_length=config.hop_length
    )
    vector = np.concatenate(
        [mfcc.mean(axis=1), mfcc.std(axis=1), rms.mean(axis=1), rms.std(axis=1)]
    ).astype(np.float32)
    if vector.shape != (config.feature_dim,):
        raise ValueError(f"Unexpected feature shape {vector.shape}")
    return vector


def extract_from_file(path: Path, config: FeatureConfig) -> np.ndarray:
    """Load one audio file and return its feature vector."""
    return extract_features(load_audio(path, config), config)


def extract_for_samples(
    samples: list[SpeechSample], config: FeatureConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (features, labels, actors) for a list of samples."""
    rows = []
    for sample in samples:
        try:
            rows.append(extract_from_file(sample.path, config))
        except Exception as exc:
            raise RuntimeError(f"Feature extraction failed for {sample.path}: {exc}") from exc
    features = np.stack(rows) if rows else np.empty((0, config.feature_dim), np.float32)
    labels = np.array([s.label for s in samples], dtype=np.int64)
    actors = np.array([s.actor for s in samples], dtype=np.int64)
    return features, labels, actors


def evenly_spaced_subset(samples: list[SpeechSample], limit: int) -> list[SpeechSample]:
    """Pick up to `limit` samples spread across the list (for quick tests only)."""
    step = max(1, len(samples) // limit)
    return samples[::step][:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract and check speech features.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only this many files per split (quick test only).",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    config = FeatureConfig()
    print(f"Feature config: {config.to_dict()}")
    print(f"Feature dimension: {config.feature_dim}")

    try:
        splits = load_ravdess()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for name, samples in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        if args.limit is not None:
            samples = evenly_spaced_subset(samples, args.limit)
        start = time.perf_counter()
        try:
            features, labels, actors = extract_for_samples(samples, config)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        elapsed = time.perf_counter() - start

        counts = np.bincount(labels, minlength=len(CANONICAL_EMOTIONS))
        print(f"\n[{name}] files={len(samples)} time={elapsed:.1f} s "
              f"({1000 * elapsed / max(len(samples), 1):.0f} ms/file)")
        print(f"  features: shape={features.shape} dtype={features.dtype} "
              f"all_finite={bool(np.isfinite(features).all())}")
        print(f"  actors  : {sorted(set(actors.tolist()))}")
        print("  labels  : " + ", ".join(
            f"{CANONICAL_EMOTIONS[i]}={counts[i]}" for i in range(len(CANONICAL_EMOTIONS))))
    return 0


if __name__ == "__main__":
    sys.exit(main())