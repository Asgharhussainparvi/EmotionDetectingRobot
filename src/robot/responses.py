"""Predefined robot responses for each emotion (software simulation only).

Design rules:
- An emotion prediction is an uncertain inference, so response texts use
  tentative language and never state how the person feels.
- No medical or mental-health advice; the robot only listens and offers help.
- Gestures are abstract names for a console or simulation. A real hardware
  driver must be added later behind an interface, with its own safety checks
  (limits, emergency stop, no motion without explicit enablement).
- When there is no decided emotion (None), a neutral fallback without any
  emotional wording is used.

Selection logic (when to speak, cooldown) is deliberately not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.common.emotions import CANONICAL_EMOTIONS


class SimulatedGesture(str, Enum):
    """Abstract gestures; no locomotion, and no hardware is driven from here."""

    NONE = "none"
    NOD = "nod"
    TILT_HEAD = "tilt_head"
    CALM_IDLE = "calm_idle"


@dataclass(frozen=True)
class Response:
    """One predefined robot response."""

    key: str  # an emotion name, or FALLBACK_KEY
    text: str
    gesture: SimulatedGesture


FALLBACK_KEY = "no_decision"

RESPONSES: dict[str, Response] = {
    "Angry": Response(
        "Angry",
        "This might be a frustrating moment. I'll keep things calm. "
        "Tell me if there is something I can help with.",
        SimulatedGesture.CALM_IDLE,
    ),
    "Fear": Response(
        "Fear",
        "If something feels worrying, take your time. "
        "I'm here, and you can tell me what would help.",
        SimulatedGesture.CALM_IDLE,
    ),
    "Happy": Response(
        "Happy",
        "That sounds like a good moment! I'm glad to be part of it.",
        SimulatedGesture.NOD,
    ),
    "Sad": Response(
        "Sad",
        "This might be a heavy moment. I'm here if you'd like to talk, "
        "and there is no rush.",
        SimulatedGesture.TILT_HEAD,
    ),
    "Neutral": Response(
        "Neutral",
        "Hello! I'm listening whenever you want to talk.",
        SimulatedGesture.NONE,
    ),
}

FALLBACK_RESPONSE = Response(FALLBACK_KEY, "I'm listening.", SimulatedGesture.NONE)

# Phrases that would state the person's feelings as fact. Used by the self-check.
FORBIDDEN_PHRASES: tuple[str, ...] = ("you are", "you seem", "you feel", "you look")


def get_response(emotion: str | None) -> Response:
    """Return the predefined response for a decided emotion.

    None means no decided emotion and returns the neutral fallback.
    Raises ValueError for names that are not canonical emotions.
    """
    if emotion is None:
        return FALLBACK_RESPONSE
    if emotion not in RESPONSES:
        raise ValueError(f"Unknown emotion '{emotion}'; expected one of {list(RESPONSES)}")
    return RESPONSES[emotion]


def _expect_error(label: str, build) -> None:
    try:
        build()
    except ValueError as exc:
        print(f"  rejected as expected ({label}): {exc}")
        return
    raise AssertionError(f"'{label}' should have been rejected")


def main() -> None:
    print("Response catalog self-check (no hardware involved)\n")

    assert set(RESPONSES) == set(CANONICAL_EMOTIONS), "catalog must cover every canonical emotion"
    for emotion in CANONICAL_EMOTIONS:
        response = RESPONSES[emotion]
        assert response.key == emotion
        assert response.text.strip(), f"empty text for {emotion}"
        print(f"{emotion:<8} gesture={response.gesture.value:<10} {response.text}")
    print(f"{'(none)':<8} gesture={FALLBACK_RESPONSE.gesture.value:<10} {FALLBACK_RESPONSE.text}")

    all_responses = [*RESPONSES.values(), FALLBACK_RESPONSE]
    for response in all_responses:
        lowered = response.text.lower()
        for phrase in FORBIDDEN_PHRASES:
            assert phrase not in lowered, (
                f"'{response.key}' states feelings as fact ('{phrase}'): {response.text}"
            )
    print("\nNo response states the person's feelings as fact.")

    assert get_response(None) is FALLBACK_RESPONSE
    assert get_response("Happy") is RESPONSES["Happy"]
    print("get_response(None) -> fallback, get_response('Happy') -> Happy response")

    print("\nInvalid usage must be rejected:")
    _expect_error("unknown emotion", lambda: get_response("Surprise"))
    _expect_error("wrong case", lambda: get_response("happy"))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()