"""Adapters from model-specific predictions to ModalityPrediction.

The facial and speech predictors return their own result types. Fusion works
on ModalityPrediction, which also says why a modality has no result:

    NO_INPUT    the sensor works but nothing was found (no face, no speech)
    UNAVAILABLE the sensor or model could not be used (camera missing, error)

This module deliberately does not import the predictor modules (they load
TensorFlow or OpenCV). It accepts any object with a ``probabilities`` mapping
keyed by emotion name, which both FacialPrediction and SpeechPrediction have.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from src.common.prediction import ModalityPrediction

# These names must match the keys of the fusion weights (see fuse.py).
FACIAL_MODALITY = "facial"
SPEECH_MODALITY = "speech"

NO_FACE_MESSAGE = "no face detected in the frame"
NO_SPEECH_MESSAGE = "no usable speech detected"


class HasProbabilities(Protocol):
    """Anything that carries a probability per emotion name."""

    @property
    def probabilities(self) -> Mapping[str, float]: ...


def _convert(
    modality: str, prediction: HasProbabilities | None, no_input_message: str
) -> ModalityPrediction:
    if prediction is None:
        return ModalityPrediction.no_input(modality, no_input_message)
    probabilities = getattr(prediction, "probabilities", None)
    if not isinstance(probabilities, Mapping):
        raise TypeError(
            f"{modality}: expected an object with a 'probabilities' mapping, "
            f"got {type(prediction).__name__}"
        )
    return ModalityPrediction.available(
        modality, {name: float(value) for name, value in probabilities.items()}
    )


def from_facial(prediction: HasProbabilities | None) -> ModalityPrediction:
    """Convert a facial prediction; None means no face was found."""
    return _convert(FACIAL_MODALITY, prediction, NO_FACE_MESSAGE)


def from_speech(prediction: HasProbabilities | None) -> ModalityPrediction:
    """Convert a speech prediction; None means no usable speech was found."""
    return _convert(SPEECH_MODALITY, prediction, NO_SPEECH_MESSAGE)


def facial_unavailable(reason: str) -> ModalityPrediction:
    """The camera or the facial model could not be used."""
    return ModalityPrediction.unavailable(FACIAL_MODALITY, reason)


def speech_unavailable(reason: str) -> ModalityPrediction:
    """The microphone or the speech model could not be used."""
    return ModalityPrediction.unavailable(SPEECH_MODALITY, reason)


# ---------------------------------------------------------------------------
# Self-check with hand-written stand-ins (no model, camera or microphone)
# ---------------------------------------------------------------------------


def _dist(top: str, confidence: float) -> dict[str, float]:
    from src.common.emotions import CANONICAL_EMOTIONS

    rest = (1.0 - confidence) / (len(CANONICAL_EMOTIONS) - 1)
    return {name: (confidence if name == top else rest) for name in CANONICAL_EMOTIONS}


def _expect_error(label: str, build) -> None:
    try:
        build()
    except (ValueError, TypeError) as exc:
        print(f"  rejected as expected ({label}): {type(exc).__name__}: {exc}")
        return
    raise AssertionError(f"'{label}' should have been rejected")


def main() -> None:
    import math
    from dataclasses import dataclass

    from src.common.prediction import ModalityStatus
    from src.fusion.fuse import DEFAULT_WEIGHTS, FusionDecision, fuse
    from src.fusion.smoothing import EmotionStabilizer

    @dataclass(frozen=True)
    class StandIn:
        """Hand-written stand-in for FacialPrediction / SpeechPrediction."""

        probabilities: dict[str, float]

    print("Adapter self-check (hand-written stand-ins, no model involved)\n")

    facial = from_facial(StandIn(_dist("Happy", 0.80)))
    assert facial.is_available and facial.modality == FACIAL_MODALITY
    assert facial.emotion == "Happy" and math.isclose(facial.confidence, 0.80)
    print(f"A facial stand-in      -> {facial.status.value}, {facial.emotion} "
          f"({facial.confidence:.2f})")

    almost = {"Angry": 0.99999994, "Fear": 0.0, "Happy": 0.0, "Sad": 0.0, "Neutral": 0.0}
    speech = from_speech(StandIn(almost))
    assert speech.is_available and speech.emotion == "Angry"
    print("B float32-style sum (0.99999994) accepted")

    no_face = from_facial(None)
    assert no_face.status is ModalityStatus.NO_INPUT and no_face.emotion is None
    assert "no face" in no_face.message
    print(f"C facial None          -> {no_face.status.value}: {no_face.message}")

    no_speech = from_speech(None)
    assert no_speech.status is ModalityStatus.NO_INPUT and "speech" in no_speech.message
    print(f"D speech None          -> {no_speech.status.value}: {no_speech.message}")

    camera_off = facial_unavailable("camera not found")
    assert camera_off.status is ModalityStatus.UNAVAILABLE
    assert camera_off.modality == FACIAL_MODALITY and camera_off.message == "camera not found"
    mic_off = speech_unavailable("microphone error")
    assert mic_off.status is ModalityStatus.UNAVAILABLE and mic_off.modality == SPEECH_MODALITY
    print(f"E unavailable helpers  -> {camera_off.modality}, {mic_off.modality}")

    assert FACIAL_MODALITY in DEFAULT_WEIGHTS and SPEECH_MODALITY in DEFAULT_WEIGHTS
    print(f"F modality names match the fusion weights: {sorted(DEFAULT_WEIGHTS)}")

    result = fuse([from_facial(None), from_speech(StandIn(_dist("Happy", 0.60)))])
    assert result.decision is FusionDecision.SINGLE_MODALITY and result.emotion == "Happy"
    assert "no_input" in result.excluded[FACIAL_MODALITY]
    print(f"G fuse: no face + speech Happy -> {result.decision.value}, {result.emotion}; "
          f"excluded: {result.excluded}")

    both = fuse([
        from_facial(StandIn(_dist("Happy", 0.80))),
        from_speech(StandIn(_dist("Happy", 0.60))),
    ])
    assert both.decision is FusionDecision.FUSED and math.isclose(both.confidence, 0.70)
    stabilizer = EmotionStabilizer()
    tokens = []
    for _ in range(3):
        stabilized = stabilizer.update_fusion(both)
        tokens.append(f"{stabilized.status.value}:{stabilized.emotion}")
    assert tokens == ["warming_up:None", "warming_up:None", "switched:Happy"], tokens
    print(f"H fuse + stabilizer, 3 updates -> {', '.join(tokens)}")

    nothing = fuse([facial_unavailable("camera not found"), from_speech(None)])
    assert nothing.decision is FusionDecision.NO_PREDICTION and nothing.emotion is None
    fresh = EmotionStabilizer().update_fusion(nothing)
    assert fresh.status.value == "no_data"
    print(f"I nothing usable       -> {nothing.decision.value}, stabilizer: {fresh.status.value}")

    print("\nInvalid usage must be rejected:")
    _expect_error("object without probabilities", lambda: from_facial("not a prediction"))
    _expect_error("extra emotion key",
                  lambda: from_speech(StandIn({**_dist("Happy", 0.8), "Surprise": 0.0})))
    _expect_error("non-numeric value",
                  lambda: from_facial(StandIn({**_dist("Happy", 0.8), "Happy": "high"})))
    _expect_error("probabilities not summing to 1",
                  lambda: from_speech(StandIn({name: 0.1 for name in _dist("Happy", 0.8)})))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()