"""Decide when the robot responds to the stabilized emotion.

Only the selection logic lives here. It does not print anything and does not
drive any hardware. Rules:

- Respond only when the stabilizer status is SWITCHED or STABLE.
- Never repeat a response for the emotion that was already answered.
- Keep at least min_updates_between_responses updates between two responses.
  An emotion change inside the cooldown is not lost: it is answered as soon as
  the cooldown is over, if that emotion is still the stable one.
- After the stabilizer reports EXPIRED, the memory of the last answered
  emotion is cleared, so a returning emotion can be answered again.
- silent_emotions are never answered (for example {"Neutral"}).

Updates are counted, not timed, so the behavior is deterministic. The default
cooldown is an untuned placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.emotions import CANONICAL_EMOTIONS
from src.fusion.smoothing import StabilizedEmotion, StabilizerStatus
from src.robot.responses import Response, get_response

SPEAKING_STATUSES = (StabilizerStatus.SWITCHED, StabilizerStatus.STABLE)


@dataclass(frozen=True)
class SelectorConfig:
    """Tunable settings (defaults are untuned placeholders)."""

    min_updates_between_responses: int = 3
    silent_emotions: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.min_updates_between_responses < 0:
            raise ValueError(
                "min_updates_between_responses must be 0 or more, "
                f"got {self.min_updates_between_responses}"
            )
        unknown = set(self.silent_emotions) - set(CANONICAL_EMOTIONS)
        if unknown:
            raise ValueError(f"silent_emotions contains unknown emotions: {sorted(unknown)}")


@dataclass(frozen=True)
class Selection:
    """Result of one update: a response to give (or None) and the reason."""

    response: Response | None
    emotion: str | None  # the stable emotion that was considered
    reason: str

    @property
    def spoken(self) -> bool:
        return self.response is not None


class ResponseSelector:
    """Stateful selector; call update() once per stabilizer update."""

    def __init__(self, config: SelectorConfig | None = None) -> None:
        self._config = config if config is not None else SelectorConfig()
        self.reset()

    def reset(self) -> None:
        """Forget all state."""
        self._updates = 0
        self._last_response_at: int | None = None
        self._last_answered_emotion: str | None = None

    def update(self, stabilized: StabilizedEmotion) -> Selection:
        """Consider one stabilizer snapshot and decide whether to respond."""
        self._updates += 1
        status = stabilized.status

        if status is StabilizerStatus.EXPIRED:
            self._last_answered_emotion = None
        if status not in SPEAKING_STATUSES:
            return Selection(None, None, f"no stable emotion to answer (status {status.value})")

        emotion = stabilized.emotion
        if emotion is None:
            raise ValueError(f"inconsistent snapshot: status {status.value} without an emotion")
        if emotion in self._config.silent_emotions:
            return Selection(None, emotion, f"{emotion} is configured as a silent emotion")
        if emotion == self._last_answered_emotion:
            return Selection(None, emotion, f"{emotion} was already answered")

        if self._last_response_at is not None:
            elapsed = self._updates - self._last_response_at
            needed = self._config.min_updates_between_responses
            if elapsed < needed:
                return Selection(
                    None,
                    emotion,
                    f"cooldown ({elapsed}/{needed} updates since the last response)",
                )

        self._last_response_at = self._updates
        self._last_answered_emotion = emotion
        return Selection(get_response(emotion), emotion, f"responding to stable {emotion}")


# ---------------------------------------------------------------------------
# Self-check with hand-written snapshots (no model, camera or hardware)
# ---------------------------------------------------------------------------


def _snap(status: StabilizerStatus, emotion: str | None) -> StabilizedEmotion:
    return StabilizedEmotion(
        status=status,
        emotion=emotion,
        confidence=None,
        smoothed_probabilities=None,
        candidate=None,
        candidate_streak=0,
        missing_updates=0,
        message="",
    )


def _tokens(selections: list[Selection]) -> list[str]:
    return [s.response.key if s.spoken else "-" for s in selections]


def _check(label: str, selections: list[Selection], expected: list[str]) -> None:
    tokens = _tokens(selections)
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
    from src.fusion.smoothing import EmotionStabilizer, StabilizerConfig

    warming = _snap(StabilizerStatus.WARMING_UP, None)
    no_data = _snap(StabilizerStatus.NO_DATA, None)
    expired = _snap(StabilizerStatus.EXPIRED, None)

    def switched(emotion: str) -> StabilizedEmotion:
        return _snap(StabilizerStatus.SWITCHED, emotion)

    def stable(emotion: str) -> StabilizedEmotion:
        return _snap(StabilizerStatus.STABLE, emotion)

    def holding(emotion: str) -> StabilizedEmotion:
        return _snap(StabilizerStatus.HOLDING, emotion)

    def run(snapshots: list[StabilizedEmotion], config: SelectorConfig) -> list[Selection]:
        selector = ResponseSelector(config)
        return [selector.update(snapshot) for snapshot in snapshots]

    config = SelectorConfig(min_updates_between_responses=3)
    print("Selector self-check (hand-written snapshots, no hardware involved)\n")

    selections = run([warming, switched("Happy"), stable("Happy"), stable("Happy")], config)
    _check("A responds once when Happy is adopted, then stays quiet", selections,
           ["-", "Happy", "-", "-"])
    assert "already answered" in selections[2].reason

    selections = run(
        [switched("Happy"), switched("Sad"), stable("Sad"), stable("Sad"), stable("Sad")], config
    )
    _check("B change inside the cooldown is answered once the cooldown ends", selections,
           ["Happy", "-", "-", "Sad", "-"])
    assert selections[1].reason.startswith("cooldown (1/3")
    assert selections[2].reason.startswith("cooldown (2/3")

    selections = run([holding("Happy"), holding("Happy"), no_data, warming, expired], config)
    _check("C holding, no_data, warming_up and expired never respond", selections, ["-"] * 5)

    selections = run(
        [switched("Happy"), holding("Happy"), holding("Happy"), expired, no_data, warming,
         switched("Happy")],
        config,
    )
    _check("D after expiry the same emotion is answered again", selections,
           ["Happy", "-", "-", "-", "-", "-", "Happy"])

    silent_config = SelectorConfig(min_updates_between_responses=3,
                                   silent_emotions=frozenset({"Neutral"}))
    selections = run([switched("Neutral"), stable("Neutral"), switched("Happy")], silent_config)
    _check("E a silent emotion is never answered", selections, ["-", "-", "Happy"])
    assert "silent" in selections[0].reason

    no_cooldown = SelectorConfig(min_updates_between_responses=0)
    selections = run([switched("Happy"), switched("Sad"), switched("Angry")], no_cooldown)
    _check("F cooldown 0 answers every change", selections, ["Happy", "Sad", "Angry"])

    selector = ResponseSelector(config)
    stabilizer = EmotionStabilizer(StabilizerConfig(
        ema_alpha=0.5, min_confidence=0.40, switch_margin=0.10,
        min_consecutive=3, max_missing_updates=2,
    ))
    happy = {"Angry": 0.05, "Fear": 0.05, "Happy": 0.80, "Sad": 0.05, "Neutral": 0.05}
    selections = [selector.update(stabilizer.update(happy)) for _ in range(4)]
    _check("G real stabilizer: 4 steady Happy observations", selections,
           ["-", "-", "Happy", "-"])
    assert selections[2].response.text == get_response("Happy").text

    print("\nInvalid usage must be rejected:")
    _expect_error("negative cooldown", lambda: SelectorConfig(min_updates_between_responses=-1))
    _expect_error("unknown silent emotion",
                  lambda: SelectorConfig(silent_emotions=frozenset({"Surprise"})))
    _expect_error("SWITCHED snapshot without an emotion",
                  lambda: ResponseSelector().update(_snap(StabilizerStatus.SWITCHED, None)))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()