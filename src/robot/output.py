"""Robot output interface and a console implementation (simulation only).

RobotOutput is the boundary between the software and any robot body. The
project code talks only to this interface, so a hardware driver (for example
on a Raspberry Pi) can be added later without changing the rest of the code.

Rules for any hardware implementation of RobotOutput:
- It must not move any actuator unless it was explicitly enabled by the user.
- It must respect joint/speed limits and implement stop() so that all motion
  halts immediately.
- It must have been tested with the hardware in a safe state first.

ConsoleRobot below drives no hardware at all. It only prints the response text
and the name of the simulated gesture.
"""

from __future__ import annotations

import sys
from typing import Protocol, TextIO, runtime_checkable

from src.robot.responses import Response, SimulatedGesture, get_response

STOP_MESSAGE = "[robot] stop (simulated): no hardware is connected"


@runtime_checkable
class RobotOutput(Protocol):
    """What the rest of the project needs from a robot body."""

    def perform(self, response: Response) -> None:
        """Say the response text and perform its gesture."""

    def stop(self) -> None:
        """Halt all motion and speech immediately."""


class ConsoleRobot:
    """Prints responses to a text stream; controls no hardware."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    def perform(self, response: Response) -> None:
        if not isinstance(response, Response):
            raise TypeError(f"expected a Response, got {type(response).__name__}")
        print(f'[robot] says: "{response.text}"', file=self._stream)
        if response.gesture is not SimulatedGesture.NONE:
            print(f"[robot] gesture (simulated): {response.gesture.value}", file=self._stream)

    def stop(self) -> None:
        print(STOP_MESSAGE, file=self._stream)


# ---------------------------------------------------------------------------
# Self-check (output is captured in memory; no hardware is involved)
# ---------------------------------------------------------------------------


def _show(title: str, text: str) -> None:
    print(title)
    for line in text.splitlines():
        print(f"  {line}")


def _expect_error(label: str, build) -> None:
    try:
        build()
    except TypeError as exc:
        print(f"  rejected as expected ({label}): {exc}")
        return
    raise AssertionError(f"'{label}' should have been rejected")


def main() -> None:
    import io

    from src.fusion.smoothing import EmotionStabilizer, StabilizerConfig
    from src.robot.selector import ResponseSelector

    print("Robot output self-check (simulation only, no hardware involved)\n")

    buffer = io.StringIO()
    robot = ConsoleRobot(stream=buffer)
    assert isinstance(robot, RobotOutput)

    happy = get_response("Happy")
    robot.perform(happy)
    expected = f'[robot] says: "{happy.text}"\n[robot] gesture (simulated): nod\n'
    assert buffer.getvalue() == expected
    _show("A Happy response (text and simulated gesture):", buffer.getvalue())

    buffer = io.StringIO()
    ConsoleRobot(stream=buffer).perform(get_response("Neutral"))
    assert "gesture" not in buffer.getvalue() and "says" in buffer.getvalue()
    _show("B Neutral response (gesture NONE prints no gesture line):", buffer.getvalue())

    buffer = io.StringIO()
    ConsoleRobot(stream=buffer).perform(get_response(None))
    assert buffer.getvalue() == '[robot] says: "I\'m listening."\n'
    _show("C fallback response for no decided emotion:", buffer.getvalue())

    buffer = io.StringIO()
    ConsoleRobot(stream=buffer).stop()
    assert buffer.getvalue() == STOP_MESSAGE + "\n"
    _show("D stop():", buffer.getvalue())

    buffer = io.StringIO()
    robot = ConsoleRobot(stream=buffer)
    selector = ResponseSelector()
    stabilizer = EmotionStabilizer(StabilizerConfig(
        ema_alpha=0.5, min_confidence=0.40, switch_margin=0.10,
        min_consecutive=3, max_missing_updates=2,
    ))
    steady_happy = {"Angry": 0.05, "Fear": 0.05, "Happy": 0.80, "Sad": 0.05, "Neutral": 0.05}
    spoken_at = []
    for update in range(1, 5):
        selection = selector.update(stabilizer.update(steady_happy))
        if selection.spoken:
            robot.perform(selection.response)
            spoken_at.append(update)
    assert spoken_at == [3] and buffer.getvalue() == expected
    _show("E stabilizer + selector + console robot (4 steady Happy updates, robot spoke "
          f"at update {spoken_at[0]}):", buffer.getvalue())

    class RecordingRobot:
        """A different implementation of the same interface (like a future driver)."""

        def __init__(self) -> None:
            self.responses: list[Response] = []
            self.stopped = False

        def perform(self, response: Response) -> None:
            self.responses.append(response)

        def stop(self) -> None:
            self.stopped = True

    recorder = RecordingRobot()
    assert isinstance(recorder, RobotOutput) and not isinstance(object(), RobotOutput)
    recorder.perform(happy)
    recorder.stop()
    assert recorder.responses == [happy] and recorder.stopped
    print("F another class can implement RobotOutput; an arbitrary object is not a RobotOutput")

    print("\nInvalid usage must be rejected:")
    _expect_error("plain string instead of a Response",
                  lambda: ConsoleRobot(stream=io.StringIO()).perform("hello"))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()