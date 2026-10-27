"""Shared prediction type produced by every modality (facial, speech, ...).

A modality can be in one of three states:
    AVAILABLE   - a prediction was produced
    NO_INPUT    - the sensor works but there was nothing to classify
                  (no face in the frame, silence, ...)
    UNAVAILABLE - the sensor or model could not be used (camera missing, error)

The emotion and confidence are always derived from the probability
distribution, so they can never disagree with it. Probabilities follow the
order and names of CANONICAL_EMOTIONS.

This module does not import any model code (no TensorFlow, no scikit-learn).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum

from src.common.emotions import CANONICAL_EMOTIONS

# Float32 softmax outputs can sum to e.g. 0.99999994, so exact equality
# with 1.0 would be wrong.
PROBABILITY_SUM_TOLERANCE = 1e-3


class ModalityStatus(str, Enum):
    AVAILABLE = "available"
    NO_INPUT = "no_input"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ModalityPrediction:
    """One modality's result for one moment in time."""

    modality: str
    status: ModalityStatus
    probabilities: dict[str, float] | None = None
    message: str = ""
    created_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if not self.modality:
            raise ValueError("modality must be a non-empty string")
        if self.status is ModalityStatus.AVAILABLE:
            self._validate_probabilities()
        elif self.probabilities is not None:
            raise ValueError(
                f"probabilities must be None when status is {self.status.value}"
            )

    def _validate_probabilities(self) -> None:
        probabilities = self.probabilities
        if probabilities is None:
            raise ValueError("an AVAILABLE prediction needs probabilities")
        if set(probabilities) != set(CANONICAL_EMOTIONS):
            raise ValueError(
                f"probabilities must have exactly the keys {list(CANONICAL_EMOTIONS)}, "
                f"got {sorted(probabilities)}"
            )
        for name, value in probabilities.items():
            if not math.isfinite(value) or value < 0.0 or value > 1.0 + PROBABILITY_SUM_TOLERANCE:
                raise ValueError(f"invalid probability for {name}: {value}")
        total = sum(probabilities.values())
        if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError(f"probabilities must sum to 1, got {total:.6f}")

    @property
    def is_available(self) -> bool:
        return self.status is ModalityStatus.AVAILABLE

    @property
    def emotion(self) -> str | None:
        """Most probable emotion; on an exact tie the earlier canonical class wins."""
        if self.probabilities is None:
            return None
        return max(CANONICAL_EMOTIONS, key=lambda name: self.probabilities[name])

    @property
    def confidence(self) -> float | None:
        """Probability of the most probable emotion (not a calibrated confidence)."""
        if self.probabilities is None:
            return None
        return self.probabilities[self.emotion]

    def vector(self) -> tuple[float, ...]:
        """Probabilities as a tuple in CANONICAL_EMOTIONS order."""
        if self.probabilities is None:
            raise ValueError(f"no probabilities: {self.modality} is {self.status.value}")
        return tuple(self.probabilities[name] for name in CANONICAL_EMOTIONS)

    @classmethod
    def available(
        cls, modality: str, probabilities: dict[str, float], message: str = ""
    ) -> "ModalityPrediction":
        return cls(modality, ModalityStatus.AVAILABLE, dict(probabilities), message)

    @classmethod
    def no_input(cls, modality: str, message: str) -> "ModalityPrediction":
        return cls(modality, ModalityStatus.NO_INPUT, None, message)

    @classmethod
    def unavailable(cls, modality: str, message: str) -> "ModalityPrediction":
        return cls(modality, ModalityStatus.UNAVAILABLE, None, message)


def _expect_value_error(label: str, build) -> None:
    try:
        build()
    except ValueError as exc:
        print(f"  rejected as expected ({label}): {exc}")
        return
    raise AssertionError(f"'{label}' should have been rejected")


def main() -> None:
    """Self-check of the type with hand-written values (no model is involved)."""
    happy = {"Angry": 0.05, "Fear": 0.05, "Happy": 0.80, "Sad": 0.05, "Neutral": 0.05}
    prediction = ModalityPrediction.available("facial", happy)
    assert prediction.is_available
    assert prediction.emotion == "Happy"
    assert prediction.confidence == 0.80
    assert prediction.vector() == (0.05, 0.05, 0.80, 0.05, 0.05)
    print(f"Valid prediction: {prediction.modality} -> {prediction.emotion} "
          f"({prediction.confidence:.2f}), vector order {list(CANONICAL_EMOTIONS)}")

    float32_like = {"Angry": 0.99999994, "Fear": 0.0, "Happy": 0.0, "Sad": 0.0, "Neutral": 0.0}
    ModalityPrediction.available("speech", float32_like)
    print("Float32-style sum (0.99999994) accepted.")

    uniform = {name: 0.2 for name in CANONICAL_EMOTIONS}
    tie = ModalityPrediction.available("speech", uniform)
    assert tie.emotion == CANONICAL_EMOTIONS[0]
    print(f"Exact tie resolves to the first canonical class: {tie.emotion}")

    silent = ModalityPrediction.no_input("speech", "recording too quiet")
    assert not silent.is_available and silent.emotion is None and silent.confidence is None
    _expect_value_error("vector() without probabilities", silent.vector)
    camera_off = ModalityPrediction.unavailable("facial", "camera not found")
    assert camera_off.status is ModalityStatus.UNAVAILABLE
    print("NO_INPUT and UNAVAILABLE predictions carry no emotion, as intended.")

    print("\nInvalid inputs must be rejected:")
    without_fear = {k: v for k, v in uniform.items() if k != "Fear"}
    _expect_value_error("missing class", lambda: ModalityPrediction.available("facial", without_fear))
    _expect_value_error("extra class", lambda: ModalityPrediction.available(
        "facial", {**uniform, "Surprise": 0.0}))
    _expect_value_error("negative value", lambda: ModalityPrediction.available(
        "facial", {**happy, "Happy": -0.1}))
    _expect_value_error("NaN value", lambda: ModalityPrediction.available(
        "facial", {**happy, "Happy": float("nan")}))
    _expect_value_error("sum is not 1", lambda: ModalityPrediction.available(
        "facial", {name: 0.1 for name in CANONICAL_EMOTIONS}))
    _expect_value_error("AVAILABLE without probabilities", lambda: ModalityPrediction(
        "facial", ModalityStatus.AVAILABLE, None))
    _expect_value_error("NO_INPUT with probabilities", lambda: ModalityPrediction(
        "facial", ModalityStatus.NO_INPUT, dict(uniform)))
    _expect_value_error("empty modality name", lambda: ModalityPrediction.available("", uniform))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()