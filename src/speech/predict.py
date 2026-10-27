"""Speech emotion inference for a single audio file.

Uses the scaler, SVM and FeatureConfig stored in models/speech/svm.joblib, so
inference uses exactly the feature settings used during training.

Security note: joblib files are pickles; only load files you created yourself.

Note: the reported test accuracy was measured on RAVDESS (acted speech, two
fixed sentences). Real-world speech may behave very differently, and SVM
probabilities are not calibrated confidence values.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from src.common.emotions import CANONICAL_EMOTIONS
from src.speech.features import FeatureConfig, extract_from_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "speech"
REQUIRED_BUNDLE_KEYS = {"scaler", "classifier", "feature_config", "class_order"}

# Display only. SVM probabilities are not calibrated confidence values.
LOW_CONFIDENCE_DISPLAY_THRESHOLD = 0.40


@dataclass(frozen=True)
class SpeechPrediction:
    """Result for one utterance."""

    emotion: str
    confidence: float
    probabilities: dict[str, float]


class SpeechEmotionPredictor:
    """Loads the trained SVM bundle and predicts emotions from audio features."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        bundle = self._load_bundle(model_dir)
        self._scaler = bundle["scaler"]
        self._classifier = bundle["classifier"]
        self._config = FeatureConfig(**bundle["feature_config"])

    @property
    def feature_config(self) -> FeatureConfig:
        return self._config

    @staticmethod
    def _load_bundle(model_dir: Path) -> dict:
        model_path = model_dir / "svm.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        bundle = joblib.load(model_path)
        missing = REQUIRED_BUNDLE_KEYS - set(bundle)
        if missing:
            raise ValueError(f"svm.joblib is missing keys: {sorted(missing)}")
        if bundle["class_order"] != list(CANONICAL_EMOTIONS):
            raise ValueError(
                f"Class order mismatch. Saved: {bundle['class_order']}, "
                f"current: {list(CANONICAL_EMOTIONS)}"
            )
        if not np.array_equal(bundle["classifier"].classes_, np.arange(len(CANONICAL_EMOTIONS))):
            raise ValueError(f"Unexpected classifier classes: {bundle['classifier'].classes_}")
        return bundle

    def predict_features(self, features: np.ndarray) -> SpeechPrediction:
        """Predict from one feature vector produced by the same FeatureConfig."""
        if features.shape != (self._config.feature_dim,):
            raise ValueError(
                f"Expected feature shape ({self._config.feature_dim},), got {features.shape}"
            )
        scaled = self._scaler.transform(features.reshape(1, -1))
        probabilities = self._classifier.predict_proba(scaled)[0]
        best = int(np.argmax(probabilities))
        return SpeechPrediction(
            emotion=CANONICAL_EMOTIONS[best],
            confidence=float(probabilities[best]),
            probabilities={
                name: float(probabilities[i]) for i, name in enumerate(CANONICAL_EMOTIONS)
            },
        )

    def predict_file(self, audio_path: Path) -> SpeechPrediction:
        """Load an audio file, extract features and predict its emotion."""
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return self.predict_features(extract_from_file(audio_path, self._config))


def main() -> int:
    parser = argparse.ArgumentParser(description="Speech emotion inference for one audio file.")
    parser.add_argument("--audio", type=Path, required=True, help="Path to an audio file.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    try:
        predictor = SpeechEmotionPredictor(args.model_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    start = time.perf_counter()
    try:
        prediction = predictor.predict_file(args.audio)
    except Exception as exc:  # report the real reason (missing file, unreadable audio, ...)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    elapsed_ms = 1000.0 * (time.perf_counter() - start)

    print(f"Audio file        : {args.audio}")
    print(f"Predicted emotion : {prediction.emotion}")
    note = "  (low confidence)" if prediction.confidence < LOW_CONFIDENCE_DISPLAY_THRESHOLD else ""
    print(f"Confidence        : {prediction.confidence:.4f}{note}")
    print("Probabilities     :")
    for name in CANONICAL_EMOTIONS:
        print(f"  {name:<8} {prediction.probabilities[name]:.4f}")
    print(f"Processing time   : {elapsed_ms:.0f} ms (single file, includes any warm-up)")
    return 0


if __name__ == "__main__":
    sys.exit(main())