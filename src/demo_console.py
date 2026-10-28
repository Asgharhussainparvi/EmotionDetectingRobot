"""Console demo: real camera and microphone -> fusion -> smoothing -> robot.

Each round (press Enter) lasts --seconds. During a round the microphone is
recorded and, at the same time, webcam frames are classified; the facial
result for the round is the average probability vector over the frames in
which a face was found. The speech result comes from the recording (or from
--audio FILE instead of the microphone).

The report names the modalities that contributed to the final decision and
those that were excluded, with the reason. Missing sensors are reported, never
silently ignored. The simulated robot answers only when the stabilized emotion
justifies it (see src/robot/selector.py); no hardware is driven.

Privacy: frames and recordings stay in memory and are never saved.

Limitations: models were trained on FER2013 (faces) and RAVDESS (acted
speech). Behavior with this camera, microphone, lighting and your voice is
unknown. Fusion weights and thresholds are untuned placeholders. Emotion
predictions are uncertain inferences, not proof of how a person feels.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd

from src.common.emotions import CANONICAL_EMOTIONS
from src.common.prediction import ModalityPrediction
from src.facial.predict import DEFAULT_MODEL_DIR as FACIAL_MODEL_DIR
from src.facial.predict import FacialEmotionPredictor
from src.fusion.adapters import (
    SPEECH_MODALITY,
    facial_unavailable,
    from_facial,
    from_speech,
    speech_unavailable,
)
from src.fusion.fuse import DEFAULT_WEIGHTS, DisagreementPolicy, FusionConfig, FusionResult, fuse
from src.fusion.smoothing import EmotionStabilizer, StabilizedEmotion, StabilizerConfig
from src.robot.output import ConsoleRobot
from src.robot.selector import ResponseSelector, Selection, SelectorConfig
from src.speech.features import extract_features
from src.speech.microphone import prepare_waveform
from src.speech.predict import DEFAULT_MODEL_DIR as SPEECH_MODEL_DIR
from src.speech.predict import SpeechEmotionPredictor

DEFAULT_SECONDS = 4.0
MIN_SECONDS = 1.0
MAX_SECONDS = 10.0
QUIT_COMMANDS = {"q", "quit", "exit"}
CAMERA_RETRY_PAUSE_S = 0.05


@dataclass(frozen=True)
class _Probabilities:
    """Minimal object with a probabilities mapping (what the adapters accept)."""

    probabilities: dict[str, float]


def format_probabilities(probabilities: dict[str, float]) -> str:
    return "  ".join(f"{name} {probabilities[name]:.2f}" for name in CANONICAL_EMOTIONS)


def describe(prediction: ModalityPrediction, detail: str) -> str:
    if prediction.is_available:
        text = f"available    {prediction.emotion} {prediction.confidence:.2f}"
    else:
        text = f"{prediction.status.value:<12} {prediction.message}"
    return f"{text}   [{detail}]" if detail else text


def open_camera(index: int) -> "cv2.VideoCapture | None":
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        return None
    return capture


def capture_facial(
    capture: "cv2.VideoCapture", predictor: FacialEmotionPredictor, seconds: float
) -> tuple[ModalityPrediction, str]:
    """Classify webcam frames for `seconds` and average those that contain a face."""
    deadline = time.perf_counter() + seconds
    frames_read = 0
    vectors: list[list[float]] = []
    while time.perf_counter() < deadline:
        success, frame = capture.read()
        if not success:
            time.sleep(CAMERA_RETRY_PAUSE_S)
            continue
        frames_read += 1
        prediction = predictor.predict_frame(frame)
        if prediction is not None:
            vectors.append([prediction.probabilities[name] for name in CANONICAL_EMOTIONS])

    if frames_read == 0:
        return facial_unavailable("camera returned no frames"), "0 frames read"
    detail = f"{len(vectors)}/{frames_read} frames contained a face"
    if not vectors:
        return from_facial(None), detail
    mean = np.mean(np.array(vectors), axis=0)
    averaged = {name: float(mean[i]) for i, name in enumerate(CANONICAL_EMOTIONS)}
    return from_facial(_Probabilities(averaged)), detail


def start_recording(seconds: float, device: int | None) -> tuple[np.ndarray, int, str]:
    """Start a non-blocking mono recording at the device's default sample rate."""
    info = sd.query_devices(device, "input")
    native_rate = int(info["default_samplerate"])
    buffer = sd.rec(
        int(seconds * native_rate),
        samplerate=native_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    return buffer, native_rate, str(info["name"])


def speech_from_recording(
    buffer: np.ndarray, native_rate: int, predictor: SpeechEmotionPredictor
) -> ModalityPrediction:
    """Turn a finished recording into a speech prediction (quiet audio is NO_INPUT)."""
    config = predictor.feature_config
    try:
        waveform = prepare_waveform(buffer[:, 0].copy(), native_rate, config)
    except ValueError as exc:
        return ModalityPrediction.no_input(SPEECH_MODALITY, str(exc))
    return from_speech(predictor.predict_features(extract_features(waveform, config)))


def print_report(
    round_number: int,
    facial: ModalityPrediction,
    facial_detail: str,
    speech: ModalityPrediction,
    speech_detail: str,
    result: FusionResult,
    stabilized: StabilizedEmotion,
) -> None:
    print(f"\n=== Round {round_number} ===")
    print(f"facial : {describe(facial, facial_detail)}")
    print(f"speech : {describe(speech, speech_detail)}")

    emotion = "None" if result.emotion is None else result.emotion
    print(f"fused  : decision={result.decision.value}  emotion={emotion}")
    if result.probabilities is not None:
        print(f"         probabilities: {format_probabilities(result.probabilities)}")
        print(f"         best guess {result.best_guess} ({result.confidence:.2f}); "
              "scores are not calibrated confidences")
    contributing = "+".join(result.contributing) if result.contributing else "none"
    print(f"         contributing: {contributing}; modalities disagree: "
          f"{'yes' if result.modalities_disagree else 'no'}")
    for modality, reason in result.excluded.items():
        print(f"         excluded {modality}: {reason}")

    stable = "-" if stabilized.emotion is None else stabilized.emotion
    line = f"stable : {stabilized.status.value}  emotion={stable}"
    if stabilized.candidate is not None:
        line += f"  (candidate {stabilized.candidate}, streak {stabilized.candidate_streak})"
    print(line)
    if stabilized.message:
        print(f"         {stabilized.message}")


def report_robot(selection: Selection, robot: ConsoleRobot) -> None:
    """Show the robot's response, or say why it stays silent this round."""
    if selection.response is None:
        print(f"robot  : silent ({selection.reason})")
        return
    robot.perform(selection.response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Console demo of the emotion pipeline.")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                        help="Length of one round.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--device", type=int, default=None, help="Microphone device index.")
    parser.add_argument("--audio", type=Path, default=None,
                        help="Use this audio file as the speech input instead of the microphone.")
    parser.add_argument("--no-camera", action="store_true", help="Do not use the camera.")
    parser.add_argument("--no-mic", action="store_true",
                        help="Do not use the microphone (ignored when --audio is given).")
    parser.add_argument("--policy", choices=[p.value for p in DisagreementPolicy],
                        default=DisagreementPolicy.WEIGHTED_AVERAGE.value,
                        help="What to do when facial and speech disagree.")
    parser.add_argument("--min-consecutive", type=int,
                        default=StabilizerConfig().min_consecutive,
                        help="Rounds a new emotion must lead before it is adopted.")
    parser.add_argument("--response-cooldown", type=int,
                        default=SelectorConfig().min_updates_between_responses,
                        help="Minimum rounds between two robot responses.")
    parser.add_argument("--silent-neutral", action="store_true",
                        help="The robot does not answer a stable Neutral emotion.")
    parser.add_argument("--facial-model-dir", type=Path, default=FACIAL_MODEL_DIR)
    parser.add_argument("--speech-model-dir", type=Path, default=SPEECH_MODEL_DIR)
    args = parser.parse_args()

    if not MIN_SECONDS <= args.seconds <= MAX_SECONDS:
        parser.error(f"--seconds must be between {MIN_SECONDS} and {MAX_SECONDS}")
    if args.min_consecutive < 1:
        parser.error("--min-consecutive must be at least 1")
    if args.response_cooldown < 0:
        parser.error("--response-cooldown must be 0 or more")
    if args.audio is not None and not args.audio.is_file():
        parser.error(f"audio file not found: {args.audio}")
    if args.no_camera and args.no_mic and args.audio is None:
        parser.error("camera and microphone are both disabled; nothing to do")
    return args


def main() -> int:
    args = parse_args()
    use_camera = not args.no_camera
    use_mic = args.audio is None and not args.no_mic
    use_speech = args.audio is not None or use_mic

    facial_predictor: FacialEmotionPredictor | None = None
    speech_predictor: SpeechEmotionPredictor | None = None
    try:
        if use_camera:
            facial_predictor = FacialEmotionPredictor(args.facial_model_dir)
        if use_speech:
            speech_predictor = SpeechEmotionPredictor(args.speech_model_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    capture = None
    camera_problem = ""
    if use_camera:
        capture = open_camera(args.camera_index)
        if capture is None:
            camera_problem = f"cannot open camera index {args.camera_index}"
            print(f"WARNING: {camera_problem}; continuing without facial input.")

    audio_prediction: ModalityPrediction | None = None
    if args.audio is not None and speech_predictor is not None:
        try:
            audio_prediction = from_speech(speech_predictor.predict_file(args.audio))
        except Exception as exc:  # shown in the report as the reason
            audio_prediction = speech_unavailable(f"{type(exc).__name__}: {exc}")

    fusion_config = FusionConfig(disagreement_policy=DisagreementPolicy(args.policy))
    stabilizer = EmotionStabilizer(StabilizerConfig(min_consecutive=args.min_consecutive))
    selector = ResponseSelector(SelectorConfig(
        min_updates_between_responses=args.response_cooldown,
        silent_emotions=frozenset({"Neutral"}) if args.silent_neutral else frozenset(),
    ))
    robot = ConsoleRobot()

    print("Privacy: camera frames and microphone recordings stay in memory and are NOT saved.")
    print(f"Fusion weights {DEFAULT_WEIGHTS} (untuned placeholders), policy: {args.policy}")
    print(f"Stable emotion needs {args.min_consecutive} consecutive leading round(s); "
          f"robot cooldown {args.response_cooldown} round(s).")
    print("The robot is simulated (console only); no hardware is driven.")
    print("Predictions are uncertain inferences, not proof of how a person feels.")

    round_number = 0
    try:
        while True:
            try:
                answer = input("\nPress Enter to run a round (q + Enter to quit): ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if answer.strip().lower() in QUIT_COMMANDS:
                break
            round_number += 1

            recording: tuple[np.ndarray, int, str] | None = None
            speech: ModalityPrediction | None = None
            speech_detail = ""
            if use_mic and speech_predictor is not None:
                try:
                    recording = start_recording(args.seconds, args.device)
                    print(f"Recording {args.seconds:.1f} s ... speak now.")
                except Exception as exc:  # no device, permission, ...
                    speech = speech_unavailable(f"{type(exc).__name__}: {exc}")

            if not use_camera:
                facial = facial_unavailable("disabled with --no-camera")
                facial_detail = ""
            elif capture is None or facial_predictor is None:
                facial = facial_unavailable(camera_problem)
                facial_detail = ""
            else:
                try:
                    facial, facial_detail = capture_facial(
                        capture, facial_predictor, args.seconds
                    )
                except Exception as exc:  # shown in the report as the reason
                    facial = facial_unavailable(f"{type(exc).__name__}: {exc}")
                    facial_detail = ""

            if recording is not None and speech_predictor is not None:
                try:
                    sd.wait()
                    speech = speech_from_recording(recording[0], recording[1], speech_predictor)
                    speech_detail = f"microphone: {recording[2]}"
                except Exception as exc:  # shown in the report as the reason
                    speech = speech_unavailable(f"{type(exc).__name__}: {exc}")
            elif speech is None:
                if args.audio is not None and audio_prediction is not None:
                    speech = audio_prediction
                    speech_detail = f"file: {args.audio.name}"
                else:
                    speech = speech_unavailable("disabled with --no-mic")

            result = fuse([facial, speech], fusion_config)
            stabilized = stabilizer.update_fusion(result)
            print_report(
                round_number, facial, facial_detail, speech, speech_detail, result, stabilized
            )
            report_robot(selector.update(stabilized), robot)
    finally:
        sd.stop()
        if capture is not None:
            capture.release()
        robot.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())