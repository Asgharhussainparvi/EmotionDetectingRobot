"""Record a short utterance from the microphone and predict its emotion.

Privacy: the recording stays in memory. It is written to disk only when
--save-wav is given explicitly.

The microphone is recorded at its own default sample rate and then resampled
to the model's sample rate, so devices that do not support 16 kHz still work.

Note: the model was trained on acted English statements (RAVDESS). Its
behavior on spontaneous speech, other languages or other microphones is
unknown and must be evaluated separately.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

from src.common.emotions import CANONICAL_EMOTIONS
from src.speech.features import FeatureConfig, extract_features
from src.speech.predict import (
    DEFAULT_MODEL_DIR,
    LOW_CONFIDENCE_DISPLAY_THRESHOLD,
    SpeechEmotionPredictor,
)

DEFAULT_SECONDS = 4.0
MIN_SECONDS = 1.0
MAX_SECONDS = 10.0
# Heuristic: below this peak amplitude (float audio in [-1, 1]) the recording
# is treated as silence. Not a calibrated value.
MIN_PEAK_AMPLITUDE = 0.01


def record(seconds: float, device: int | None) -> tuple[np.ndarray, int]:
    """Record mono audio at the device's default sample rate."""
    info = sd.query_devices(device, "input")
    native_rate = int(info["default_samplerate"])
    print(f"Input device : {info['name']} ({native_rate} Hz)")
    frames = int(seconds * native_rate)
    audio = sd.rec(frames, samplerate=native_rate, channels=1, dtype="float32", device=device)
    sd.wait()
    return audio[:, 0], native_rate


def prepare_waveform(audio: np.ndarray, native_rate: int, config: FeatureConfig) -> np.ndarray:
    """Check loudness, resample to the model's rate and trim silence."""
    peak = float(np.max(np.abs(audio)))
    print(f"Peak amplitude: {peak:.4f}")
    if peak < MIN_PEAK_AMPLITUDE:
        raise ValueError(
            f"Recording is too quiet (peak {peak:.4f} < {MIN_PEAK_AMPLITUDE}). "
            "Check the microphone, its Windows input level, and speak closer."
        )
    if native_rate != config.sample_rate:
        audio = librosa.resample(audio, orig_sr=native_rate, target_sr=config.sample_rate)
    trimmed, _ = librosa.effects.trim(audio, top_db=config.trim_top_db)
    if trimmed.size < config.n_fft:
        raise ValueError("Too little speech left after trimming silence.")
    return trimmed


def main() -> int:
    parser = argparse.ArgumentParser(description="Microphone speech emotion inference.")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--device", type=int, default=None, help="Input device index.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--save-wav", type=Path, default=None,
                        help="Optionally save the raw recording here (off by default).")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return 0
    if not MIN_SECONDS <= args.seconds <= MAX_SECONDS:
        parser.error(f"--seconds must be between {MIN_SECONDS} and {MAX_SECONDS}")

    try:
        predictor = SpeechEmotionPredictor(args.model_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.save_wav is None:
        print("Privacy: the recording stays in memory and is NOT saved.")
    else:
        print(f"Privacy: the raw recording WILL be saved to {args.save_wav}")
    print(f"Recording for {args.seconds:.1f} s. Speak now ...")

    try:
        audio, native_rate = record(args.seconds, args.device)
        print("Recording finished.")
        if args.save_wav is not None:
            args.save_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(args.save_wav), audio, native_rate)
            print(f"Saved: {args.save_wav}")

        start = time.perf_counter()
        waveform = prepare_waveform(audio, native_rate, predictor.feature_config)
        features = extract_features(waveform, predictor.feature_config)
        prediction = predictor.predict_features(features)
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
    except Exception as exc:  # report the real reason (no device, too quiet, ...)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"\nSpeech after trimming : {waveform.size / predictor.feature_config.sample_rate:.2f} s")
    print(f"Predicted emotion     : {prediction.emotion}")
    note = "  (low confidence)" if prediction.confidence < LOW_CONFIDENCE_DISPLAY_THRESHOLD else ""
    print(f"Confidence            : {prediction.confidence:.4f}{note}")
    print("Probabilities         :")
    for name in CANONICAL_EMOTIONS:
        print(f"  {name:<8} {prediction.probabilities[name]:.4f}")
    print(f"Processing time       : {elapsed_ms:.0f} ms (excludes recording)")
    return 0


if __name__ == "__main__":
    sys.exit(main())