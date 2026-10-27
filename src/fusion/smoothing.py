"""Temporal stabilization of emotion estimates (smoothing + hysteresis).

Per update the stabilizer receives a probability distribution (for example
FusionResult.probabilities) or None when nothing usable was observed.

Smoothing : exponential moving average of the probability vector.
Hysteresis: a new emotion is adopted only when its smoothed probability is at
            least min_confidence, it leads the current emotion by at least
            switch_margin, and this holds for min_consecutive updates in a row.
Missing   : up to max_missing_updates consecutive missing observations keep
            the last emotion (HOLDING); after that the state is cleared.

The defaults are untuned placeholders. Updates are counted, not timed, so the
behavior is deterministic and testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from src.common.emotions import CANONICAL_EMOTIONS
from src.common.prediction import PROBABILITY_SUM_TOLERANCE, ModalityPrediction
from src.fusion.fuse import FusionResult, fuse


class StabilizerStatus(str, Enum):
    WARMING_UP = "warming_up"  # no stable emotion yet
    STABLE = "stable"  # current emotion unchanged
    SWITCHED = "switched"  # an emotion was adopted or replaced in this update
    HOLDING = "holding"  # observation missing, last emotion kept
    EXPIRED = "expired"  # too many missing updates, state cleared in this update
    NO_DATA = "no_data"  # nothing observed and no state


@dataclass(frozen=True)
class StabilizerConfig:
    """Tunable settings (defaults are untuned placeholders)."""

    ema_alpha: float = 0.3  # weight of the newest observation, in (0, 1]
    min_confidence: float = 0.40  # smoothed probability needed to adopt an emotion
    switch_margin: float = 0.10  # lead over the current emotion needed to switch
    min_consecutive: int = 3  # updates a candidate must lead before adoption
    max_missing_updates: int = 10  # consecutive missing updates before clearing

    def __post_init__(self) -> None:
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {self.min_confidence}")
        if not 0.0 <= self.switch_margin <= 1.0:
            raise ValueError(f"switch_margin must be in [0, 1], got {self.switch_margin}")
        if self.min_consecutive < 1:
            raise ValueError(f"min_consecutive must be at least 1, got {self.min_consecutive}")
        if self.max_missing_updates < 0:
            raise ValueError(
                f"max_missing_updates must be 0 or more, got {self.max_missing_updates}"
            )


@dataclass(frozen=True)
class StabilizedEmotion:
    """Snapshot after one update."""

    status: StabilizerStatus
    emotion: str | None  # current stable emotion (None while warming up or expired)
    confidence: float | None  # smoothed probability of that emotion
    smoothed_probabilities: dict[str, float] | None
    candidate: str | None  # emotion currently trying to replace the stable one
    candidate_streak: int
    missing_updates: int
    message: str


class EmotionStabilizer:
    """Stateful smoothing and hysteresis over a stream of probability vectors."""

    def __init__(self, config: StabilizerConfig | None = None) -> None:
        self._config = config if config is not None else StabilizerConfig()
        self.reset()

    def reset(self) -> None:
        """Forget all state."""
        self._ema: dict[str, float] | None = None
        self._current: str | None = None
        self._candidate: str | None = None
        self._streak = 0
        self._missing = 0

    @staticmethod
    def _validate(probabilities: dict[str, float]) -> None:
        if set(probabilities) != set(CANONICAL_EMOTIONS):
            raise ValueError(
                f"probabilities must have exactly the keys {list(CANONICAL_EMOTIONS)}, "
                f"got {sorted(probabilities)}"
            )
        for name, value in probabilities.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid probability for {name}: {value}")
        total = sum(probabilities.values())
        if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError(f"probabilities must sum to 1, got {total:.6f}")

    def _snapshot(self, status: StabilizerStatus, message: str = "") -> StabilizedEmotion:
        confidence = None
        if self._current is not None and self._ema is not None:
            confidence = self._ema[self._current]
        return StabilizedEmotion(
            status=status,
            emotion=self._current,
            confidence=confidence,
            smoothed_probabilities=None if self._ema is None else dict(self._ema),
            candidate=self._candidate,
            candidate_streak=self._streak,
            missing_updates=self._missing,
            message=message,
        )

    def _handle_missing(self) -> StabilizedEmotion:
        if self._ema is None:
            return self._snapshot(StabilizerStatus.NO_DATA, "nothing observed yet")
        self._missing += 1
        if self._missing > self._config.max_missing_updates:
            self.reset()
            return self._snapshot(
                StabilizerStatus.EXPIRED, "too many missing observations, state cleared"
            )
        return self._snapshot(
            StabilizerStatus.HOLDING,
            f"observation missing ({self._missing}/{self._config.max_missing_updates}), "
            "keeping last state",
        )

    def update(self, probabilities: dict[str, float] | None) -> StabilizedEmotion:
        """Feed one observation (or None when nothing usable was observed)."""
        if probabilities is None:
            return self._handle_missing()
        self._validate(probabilities)
        config = self._config
        self._missing = 0

        observation = {name: float(probabilities[name]) for name in CANONICAL_EMOTIONS}
        if self._ema is None:
            self._ema = observation
        else:
            alpha = config.ema_alpha
            self._ema = {
                name: alpha * observation[name] + (1.0 - alpha) * self._ema[name]
                for name in CANONICAL_EMOTIONS
            }

        best = max(CANONICAL_EMOTIONS, key=lambda name: self._ema[name])
        candidate = None
        if best != self._current and self._ema[best] >= config.min_confidence:
            leads_enough = (
                self._current is None
                or self._ema[best] - self._ema[self._current] >= config.switch_margin
            )
            if leads_enough:
                candidate = best

        if candidate is None:
            self._candidate = None
            self._streak = 0
        elif candidate == self._candidate:
            self._streak += 1
        else:
            self._candidate = candidate
            self._streak = 1

        status = (
            StabilizerStatus.STABLE if self._current is not None else StabilizerStatus.WARMING_UP
        )
        message = ""
        if self._candidate is not None and self._streak >= config.min_consecutive:
            previous = self._current
            self._current = self._candidate
            self._candidate = None
            self._streak = 0
            status = StabilizerStatus.SWITCHED
            message = (
                f"adopted {self._current}"
                if previous is None
                else f"switched {previous} -> {self._current}"
            )
        return self._snapshot(status, message)

    def update_fusion(self, result: FusionResult) -> StabilizedEmotion:
        """Feed a FusionResult; a result without probabilities counts as missing."""
        return self.update(result.probabilities)


# ---------------------------------------------------------------------------
# Self-check with hand-written observations (no model is involved)
# ---------------------------------------------------------------------------


def _dist(top: str, confidence: float) -> dict[str, float]:
    rest = (1.0 - confidence) / (len(CANONICAL_EMOTIONS) - 1)
    return {name: (confidence if name == top else rest) for name in CANONICAL_EMOTIONS}


def _token(result: StabilizedEmotion) -> str:
    emotion = "-" if result.emotion is None else result.emotion
    return f"{result.status.value}:{emotion}"


def _check(label: str, results: list[StabilizedEmotion], expected: list[str]) -> None:
    tokens = [_token(result) for result in results]
    print(f"{label}\n  {', '.join(tokens)}")
    assert tokens == expected, f"{label}: expected {expected}, got {tokens}"


def _expect_error(label: str, build) -> None:
    try:
        build()
    except ValueError as exc:
        print(f"  rejected as expected ({label}): {exc}")
        return
    raise AssertionError(f"'{label}' should have been rejected")


def main() -> None:
    config = StabilizerConfig(
        ema_alpha=0.5,
        min_confidence=0.40,
        switch_margin=0.10,
        min_consecutive=3,
        max_missing_updates=2,
    )
    happy = _dist("Happy", 0.80)
    sad = _dist("Sad", 0.90)
    uniform = {name: 0.2 for name in CANONICAL_EMOTIONS}

    def run(observations: list) -> list[StabilizedEmotion]:
        stabilizer = EmotionStabilizer(config)
        return [stabilizer.update(observation) for observation in observations]

    print("Stabilizer self-check (hand-written inputs, no model involved)\n")

    _check("A steady Happy (needs 3 updates to adopt)", run([happy] * 4),
           ["warming_up:-", "warming_up:-", "switched:Happy", "stable:Happy"])

    _check("B single Sad glitch is ignored", run([happy] * 3 + [sad, happy]),
           ["warming_up:-", "warming_up:-", "switched:Happy", "stable:Happy", "stable:Happy"])

    _check("C genuine change to Sad needs 3 leading updates", run([happy] * 3 + [sad] * 4),
           ["warming_up:-", "warming_up:-", "switched:Happy", "stable:Happy",
            "stable:Happy", "stable:Happy", "switched:Sad"])

    _check("D low confidence never adopts an emotion", run([uniform] * 6),
           ["warming_up:-"] * 6)

    _check("E missing data: hold, expire, restart",
           run([None] + [happy] * 3 + [None] * 4 + [happy] * 3),
           ["no_data:-", "warming_up:-", "warming_up:-", "switched:Happy",
            "holding:Happy", "holding:Happy", "expired:-", "no_data:-",
            "warming_up:-", "warming_up:-", "switched:Happy"])

    agree = fuse([
        ModalityPrediction.available("facial", happy),
        ModalityPrediction.available("speech", happy),
    ])
    nothing = fuse([ModalityPrediction.unavailable("facial", "camera not found")])
    stabilizer = EmotionStabilizer(config)
    results = [stabilizer.update_fusion(agree) for _ in range(3)]
    results.append(stabilizer.update_fusion(nothing))
    _check("F fed from FusionResult (3 agreements, then no_prediction)", results,
           ["warming_up:-", "warming_up:-", "switched:Happy", "holding:Happy"])

    print("\nInvalid usage must be rejected:")
    _expect_error("ema_alpha 0", lambda: StabilizerConfig(ema_alpha=0.0))
    _expect_error("ema_alpha above 1", lambda: StabilizerConfig(ema_alpha=1.5))
    _expect_error("min_consecutive 0", lambda: StabilizerConfig(min_consecutive=0))
    _expect_error("negative max_missing_updates",
                  lambda: StabilizerConfig(max_missing_updates=-1))
    _expect_error("min_confidence above 1", lambda: StabilizerConfig(min_confidence=1.5))
    probe = EmotionStabilizer(config)
    _expect_error("missing class",
                  lambda: probe.update({k: v for k, v in uniform.items() if k != "Fear"}))
    _expect_error("NaN probability", lambda: probe.update({**happy, "Happy": float("nan")}))
    _expect_error("sum is not 1", lambda: probe.update({name: 0.1 for name in CANONICAL_EMOTIONS}))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()