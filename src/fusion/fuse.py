"""Decision-level fusion of per-modality emotion predictions.

Each modality (facial, speech, ...) supplies a ModalityPrediction. This module
combines the usable ones into one FusionResult and always reports which
modalities contributed, which were excluded and why, and whether the
modalities disagreed.

Limitations (to be documented in the paper):
- Scores from different models are not calibrated and not directly comparable.
- The default weights and thresholds are untuned placeholders.
- No temporal smoothing here; that is stateful and lives in a separate module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from src.common.emotions import CANONICAL_EMOTIONS
from src.common.prediction import ModalityPrediction

DEFAULT_WEIGHTS: dict[str, float] = {"facial": 0.5, "speech": 0.5}
DEFAULT_MIN_FUSED_CONFIDENCE = 0.40
TIE_TOLERANCE = 1e-9


class DisagreementPolicy(str, Enum):
    """What to do when usable modalities disagree on the top emotion."""

    WEIGHTED_AVERAGE = "weighted_average"
    MORE_CONFIDENT = "more_confident"
    UNCERTAIN = "uncertain"


class FusionDecision(str, Enum):
    FUSED = "fused"
    SINGLE_MODALITY = "single_modality"
    DISAGREEMENT_RESOLVED = "disagreement_resolved"
    DISAGREEMENT_UNCERTAIN = "disagreement_uncertain"
    LOW_CONFIDENCE = "low_confidence"
    NO_PREDICTION = "no_prediction"


@dataclass(frozen=True)
class FusionConfig:
    """Tunable fusion settings (defaults are untuned placeholders)."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    min_modality_confidence: float = 0.0  # 0.0 = never exclude a modality for low confidence
    min_fused_confidence: float = DEFAULT_MIN_FUSED_CONFIDENCE
    disagreement_policy: DisagreementPolicy = DisagreementPolicy.WEIGHTED_AVERAGE

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("weights must not be empty")
        for name, weight in self.weights.items():
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"invalid weight for '{name}': {weight}")
        if sum(self.weights.values()) <= 0.0:
            raise ValueError("at least one weight must be positive")
        for label, value in (
            ("min_modality_confidence", self.min_modality_confidence),
            ("min_fused_confidence", self.min_fused_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be between 0 and 1, got {value}")


@dataclass(frozen=True)
class FusionResult:
    """Outcome of fusing one set of modality predictions."""

    decision: FusionDecision
    emotion: str | None  # final decision; None when uncertain or nothing usable
    probabilities: dict[str, float] | None
    contributing: tuple[str, ...]
    weights_used: dict[str, float]  # normalized over contributing modalities
    excluded: dict[str, str]  # modality -> reason it was not used
    modalities_disagree: bool
    message: str

    @property
    def best_guess(self) -> str | None:
        """Most probable emotion even when the final decision is None."""
        if self.probabilities is None:
            return None
        return max(CANONICAL_EMOTIONS, key=lambda name: self.probabilities[name])

    @property
    def confidence(self) -> float | None:
        """Probability of the best guess (not a calibrated confidence)."""
        if self.probabilities is None:
            return None
        return self.probabilities[self.best_guess]


def _check_predictions(predictions: Sequence[ModalityPrediction], config: FusionConfig) -> None:
    seen: set[str] = set()
    for prediction in predictions:
        if prediction.modality not in config.weights:
            raise ValueError(f"No fusion weight configured for modality '{prediction.modality}'")
        if prediction.modality in seen:
            raise ValueError(f"Modality '{prediction.modality}' was given more than once")
        seen.add(prediction.modality)


def _weighted_average(
    contributors: list[ModalityPrediction], weights: dict[str, float]
) -> tuple[dict[str, float], dict[str, float]]:
    total = sum(weights[p.modality] for p in contributors)
    used = {p.modality: weights[p.modality] / total for p in contributors}
    probabilities = {
        name: sum(used[p.modality] * p.probabilities[name] for p in contributors)
        for name in CANONICAL_EMOTIONS
    }
    return probabilities, used


def fuse(
    predictions: Sequence[ModalityPrediction], config: FusionConfig | None = None
) -> FusionResult:
    """Fuse modality predictions according to the configuration."""
    config = config if config is not None else FusionConfig()
    _check_predictions(predictions, config)

    excluded: dict[str, str] = {}
    usable: list[ModalityPrediction] = []
    for prediction in predictions:
        name = prediction.modality
        if not prediction.is_available:
            reason = prediction.status.value
            if prediction.message:
                reason += f": {prediction.message}"
            excluded[name] = reason
        elif config.weights[name] <= 0.0:
            excluded[name] = "weight is 0"
        elif prediction.confidence < config.min_modality_confidence:
            excluded[name] = (
                f"confidence {prediction.confidence:.2f} is below the modality "
                f"minimum {config.min_modality_confidence:.2f}"
            )
        else:
            usable.append(prediction)

    if not usable:
        detail = "; ".join(f"{m} ({r})" for m, r in excluded.items()) or "no predictions given"
        return FusionResult(
            decision=FusionDecision.NO_PREDICTION,
            emotion=None,
            probabilities=None,
            contributing=(),
            weights_used={},
            excluded=excluded,
            modalities_disagree=False,
            message=f"No usable modality: {detail}",
        )

    disagree = len({p.emotion for p in usable}) > 1
    contributors = list(usable)
    decision = (
        FusionDecision.FUSED if len(usable) > 1 else FusionDecision.SINGLE_MODALITY
    )
    uncertain = disagree and config.disagreement_policy is DisagreementPolicy.UNCERTAIN

    if disagree and config.disagreement_policy is DisagreementPolicy.MORE_CONFIDENT:
        ranked = sorted(
            usable,
            key=lambda p: (p.confidence, config.weights[p.modality]),
            reverse=True,
        )
        first, second = ranked[0], ranked[1]
        is_tie = math.isclose(
            first.confidence, second.confidence, abs_tol=TIE_TOLERANCE
        ) and math.isclose(
            config.weights[first.modality], config.weights[second.modality], abs_tol=TIE_TOLERANCE
        )
        if is_tie:
            uncertain = True  # no transparent way to pick a winner
        else:
            contributors = [first]
            for other in usable:
                if other is not first:
                    excluded[other.modality] = (
                        f"disagreed with and was less confident than {first.modality}"
                    )
            decision = FusionDecision.DISAGREEMENT_RESOLVED

    probabilities, weights_used = _weighted_average(contributors, config.weights)
    best = max(CANONICAL_EMOTIONS, key=lambda name: probabilities[name])
    emotion: str | None = best

    if uncertain:
        decision = FusionDecision.DISAGREEMENT_UNCERTAIN
        emotion = None
    elif probabilities[best] < config.min_fused_confidence:
        decision = FusionDecision.LOW_CONFIDENCE
        emotion = None

    parts = [f"{decision.value}: used {'+'.join(p.modality for p in contributors)}"]
    if disagree:
        parts.append("modalities disagree on the top emotion")
    if excluded:
        parts.append("excluded: " + "; ".join(f"{m} ({r})" for m, r in excluded.items()))

    return FusionResult(
        decision=decision,
        emotion=emotion,
        probabilities=probabilities,
        contributing=tuple(p.modality for p in contributors),
        weights_used=weights_used,
        excluded=excluded,
        modalities_disagree=disagree,
        message="; ".join(parts),
    )


# ---------------------------------------------------------------------------
# Self-check with hand-written predictions (no model is involved)
# ---------------------------------------------------------------------------


def _dist(top: str, confidence: float) -> dict[str, float]:
    rest = (1.0 - confidence) / (len(CANONICAL_EMOTIONS) - 1)
    return {name: (confidence if name == top else rest) for name in CANONICAL_EMOTIONS}


def _pred(modality: str, top: str, confidence: float) -> ModalityPrediction:
    return ModalityPrediction.available(modality, _dist(top, confidence))


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-9)


def _show(label: str, result: FusionResult) -> None:
    emotion = "None" if result.emotion is None else result.emotion
    confidence = "-" if result.confidence is None else f"{result.confidence:.4f}"
    print(f"{label:<34} {result.decision.value:<23} emotion={emotion:<8} "
          f"best_guess_conf={confidence}")


def _expect_error(label: str, build) -> None:
    try:
        build()
    except ValueError as exc:
        print(f"  rejected as expected ({label}): {exc}")
        return
    raise AssertionError(f"'{label}' should have been rejected")


def main() -> None:
    facial_happy_80 = _pred("facial", "Happy", 0.80)
    facial_happy_70 = _pred("facial", "Happy", 0.70)
    speech_happy_60 = _pred("speech", "Happy", 0.60)
    speech_sad_50 = _pred("speech", "Sad", 0.50)

    print("Fusion self-check (hand-written inputs, no model involved)\n")

    result = fuse([facial_happy_80, speech_happy_60])
    _show("A agreement", result)
    assert result.decision is FusionDecision.FUSED and result.emotion == "Happy"
    assert _close(result.confidence, 0.70) and not result.modalities_disagree
    assert result.contributing == ("facial", "speech")

    result = fuse(
        [facial_happy_80, speech_happy_60],
        FusionConfig(weights={"facial": 3.0, "speech": 1.0}),
    )
    _show("B weights 3:1 (normalized)", result)
    assert _close(result.confidence, 0.75) and _close(result.weights_used["facial"], 0.75)

    result = fuse([facial_happy_70, speech_sad_50])
    _show("C disagreement, weighted average", result)
    assert result.decision is FusionDecision.FUSED and result.emotion == "Happy"
    assert _close(result.confidence, 0.4125) and result.modalities_disagree

    result = fuse(
        [facial_happy_70, speech_sad_50],
        FusionConfig(disagreement_policy=DisagreementPolicy.MORE_CONFIDENT),
    )
    _show("D disagreement, more_confident", result)
    assert result.decision is FusionDecision.DISAGREEMENT_RESOLVED
    assert result.emotion == "Happy" and result.contributing == ("facial",)
    assert "speech" in result.excluded and _close(result.confidence, 0.70)

    result = fuse(
        [facial_happy_70, speech_sad_50],
        FusionConfig(disagreement_policy=DisagreementPolicy.UNCERTAIN),
    )
    _show("E disagreement, uncertain", result)
    assert result.decision is FusionDecision.DISAGREEMENT_UNCERTAIN
    assert result.emotion is None and result.best_guess == "Happy"

    tie_pair = [_pred("facial", "Happy", 0.60), _pred("speech", "Sad", 0.60)]
    more_confident = DisagreementPolicy.MORE_CONFIDENT
    result = fuse(tie_pair, FusionConfig(disagreement_policy=more_confident))
    _show("F1 exact tie, more_confident", result)
    assert result.decision is FusionDecision.DISAGREEMENT_UNCERTAIN and result.emotion is None
    result = fuse(
        tie_pair,
        FusionConfig(
            weights={"facial": 0.6, "speech": 0.4}, disagreement_policy=more_confident
        ),
    )
    _show("F2 tie, higher weight wins", result)
    assert result.decision is FusionDecision.DISAGREEMENT_RESOLVED and result.emotion == "Happy"

    no_face = ModalityPrediction.no_input("facial", "no face detected")
    result = fuse([no_face, speech_happy_60])
    _show("G facial has no input", result)
    assert result.decision is FusionDecision.SINGLE_MODALITY and result.emotion == "Happy"
    assert _close(result.confidence, 0.60) and result.contributing == ("speech",)
    assert "no_input" in result.excluded["facial"]

    result = fuse([
        ModalityPrediction.unavailable("facial", "camera not found"),
        ModalityPrediction.unavailable("speech", "microphone error"),
    ])
    _show("H nothing available", result)
    assert result.decision is FusionDecision.NO_PREDICTION
    assert result.emotion is None and result.probabilities is None and result.confidence is None

    result = fuse(
        [_pred("facial", "Sad", 0.30), speech_happy_60],
        FusionConfig(min_modality_confidence=0.40),
    )
    _show("I weak modality excluded", result)
    assert result.decision is FusionDecision.SINGLE_MODALITY and result.emotion == "Happy"
    assert "facial" in result.excluded

    result = fuse([_pred("facial", "Happy", 0.50), _pred("speech", "Sad", 0.45)])
    _show("J low fused confidence", result)
    assert result.decision is FusionDecision.LOW_CONFIDENCE and result.emotion is None
    assert result.best_guess == "Happy" and _close(result.confidence, 0.31875)

    result = fuse([_pred("speech", "Happy", 0.30)])
    _show("K single weak modality", result)
    assert result.decision is FusionDecision.LOW_CONFIDENCE and result.emotion is None

    print("\nInvalid usage must be rejected:")
    _expect_error("unknown modality", lambda: fuse([_pred("gesture", "Happy", 0.8)]))
    _expect_error("duplicate modality", lambda: fuse([facial_happy_80, facial_happy_70]))
    _expect_error("empty weights", lambda: FusionConfig(weights={}))
    _expect_error("negative weight", lambda: FusionConfig(weights={"facial": -1.0, "speech": 1.0}))
    _expect_error("all weights zero", lambda: FusionConfig(weights={"facial": 0.0, "speech": 0.0}))
    _expect_error("threshold above 1", lambda: FusionConfig(min_fused_confidence=1.5))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()