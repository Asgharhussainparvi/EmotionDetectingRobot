"""Facial emotion inference for a single image or a live webcam stream.

Pipeline: BGR frame -> grayscale -> Haar-cascade face detection (largest
face) -> crop -> resize to 48x48 -> CNN -> probabilities over
CANONICAL_EMOTIONS.

Privacy: frames are processed in memory only. Nothing is written to disk
unless --output is given explicitly (image mode).

Note: the reported test accuracy was measured on FER2013 PrivateTest.
Webcam conditions (lighting, angle, camera quality) differ from FER2013, so
live performance is unknown and must be evaluated separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from src.common.emotions import CANONICAL_EMOTIONS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "facial"

IMAGE_SIDE = 48
CASCADE_FILE = "haarcascade_frontalface_default.xml"
DETECT_SCALE_FACTOR = 1.1
DETECT_MIN_NEIGHBORS = 5
DETECT_MIN_SIZE = (48, 48)

# Display only. Softmax scores are not calibrated probabilities.
LOW_CONFIDENCE_DISPLAY_THRESHOLD = 0.40

WINDOW_NAME = "Facial emotion (press q or Esc to quit)"
ESC_KEY = 27
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_COLOR = (0, 255, 0)
WARNING_COLOR = (0, 0, 255)
BAR_MAX_WIDTH = 100
BAR_HEIGHT = 12
BAR_ROW_SPACING = 20


@dataclass(frozen=True)
class FacialPrediction:
    """Result for one detected face."""

    emotion: str
    confidence: float
    probabilities: dict[str, float]
    face_box: tuple[int, int, int, int]  # x, y, width, height in frame pixels


class FacialEmotionPredictor:
    """Loads the trained CNN and a face detector; predicts from BGR frames."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        model_path = model_dir / "best.keras"
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._check_class_order(model_dir)
        self._model = keras.models.load_model(model_path)
        self._detector = self._load_detector()

    @staticmethod
    def _check_class_order(model_dir: Path) -> None:
        mapping_path = model_dir / "label_mapping.json"
        if not mapping_path.is_file():
            raise FileNotFoundError(f"Label mapping not found: {mapping_path}")
        with mapping_path.open("r", encoding="utf-8") as handle:
            saved_order = json.load(handle)["class_order"]
        if saved_order != list(CANONICAL_EMOTIONS):
            raise ValueError(
                f"Class order mismatch. Saved: {saved_order}, "
                f"current: {list(CANONICAL_EMOTIONS)}"
            )

    @staticmethod
    def _load_detector() -> "cv2.CascadeClassifier":
        if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
            raise RuntimeError(
                f"This OpenCV build ({cv2.__version__}) has no CascadeClassifier "
                "or bundled cascade data (cv2.data). A different face detector "
                "is needed; please report this message."
            )
        cascade_path = Path(cv2.data.haarcascades) / CASCADE_FILE
        if not cascade_path.is_file():
            raise FileNotFoundError(f"Face cascade file not found: {cascade_path}")
        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            raise RuntimeError(f"Failed to load face cascade: {cascade_path}")
        return detector

    def detect_largest_face(self, gray: np.ndarray) -> tuple[int, int, int, int] | None:
        """Return (x, y, w, h) of the largest detected face, or None."""
        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=DETECT_SCALE_FACTOR,
            minNeighbors=DETECT_MIN_NEIGHBORS,
            minSize=DETECT_MIN_SIZE,
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        return int(x), int(y), int(w), int(h)

    def predict_face(self, face_gray: np.ndarray) -> np.ndarray:
        """Return class probabilities for one grayscale face crop (any size)."""
        resized = cv2.resize(
            face_gray, (IMAGE_SIDE, IMAGE_SIDE), interpolation=cv2.INTER_AREA
        )
        batch = resized.reshape(1, IMAGE_SIDE, IMAGE_SIDE, 1).astype(np.uint8)
        return self._model(batch, training=False).numpy()[0]

    def predict_frame(self, frame_bgr: np.ndarray) -> FacialPrediction | None:
        """Detect the largest face in a BGR frame and classify its emotion."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        box = self.detect_largest_face(gray)
        if box is None:
            return None
        x, y, w, h = box
        probabilities = self.predict_face(gray[y : y + h, x : x + w])
        best = int(np.argmax(probabilities))
        return FacialPrediction(
            emotion=CANONICAL_EMOTIONS[best],
            confidence=float(probabilities[best]),
            probabilities={
                name: float(probabilities[i]) for i, name in enumerate(CANONICAL_EMOTIONS)
            },
            face_box=box,
        )


def annotate(frame: np.ndarray, prediction: FacialPrediction | None) -> None:
    """Draw the box, label and probability bars on the frame (in place)."""
    if prediction is None:
        cv2.putText(frame, "No face detected", (10, 30), FONT, 0.8, WARNING_COLOR, 2)
        return

    x, y, w, h = prediction.face_box
    cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)
    label = f"{prediction.emotion} {prediction.confidence:.2f}"
    if prediction.confidence < LOW_CONFIDENCE_DISPLAY_THRESHOLD:
        label += " (low confidence)"
    cv2.putText(frame, label, (x, max(y - 10, 20)), FONT, 0.7, BOX_COLOR, 2)

    for row, name in enumerate(CANONICAL_EMOTIONS):
        probability = prediction.probabilities[name]
        top = 10 + row * BAR_ROW_SPACING
        cv2.putText(frame, name, (10, top + BAR_HEIGHT), FONT, 0.45, BOX_COLOR, 1)
        bar_left = 85
        cv2.rectangle(
            frame,
            (bar_left, top),
            (bar_left + int(probability * BAR_MAX_WIDTH), top + BAR_HEIGHT),
            BOX_COLOR,
            -1,
        )


def run_image(predictor: FacialEmotionPredictor, image_path: Path, output: Path | None) -> int:
    if not image_path.is_file():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 1
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"ERROR: could not read image: {image_path}", file=sys.stderr)
        return 1

    prediction = predictor.predict_frame(frame)
    if prediction is None:
        print("No face detected.")
    else:
        print(f"Predicted emotion : {prediction.emotion}")
        print(f"Confidence        : {prediction.confidence:.4f}")
        print("Probabilities     :")
        for name in CANONICAL_EMOTIONS:
            print(f"  {name:<8} {prediction.probabilities[name]:.4f}")

    if output is not None:
        annotate(frame, prediction)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), frame):
            print(f"ERROR: could not write {output}", file=sys.stderr)
            return 1
        print(f"Saved annotated image: {output}")
    return 0


def run_webcam(predictor: FacialEmotionPredictor, camera_index: int) -> int:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        print(f"ERROR: cannot open camera index {camera_index}", file=sys.stderr)
        return 1
    print("Webcam started. Frames are processed in memory and are NOT saved.")
    print("Press q or Esc in the video window to quit.")
    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("ERROR: failed to read a frame from the camera", file=sys.stderr)
                return 1
            annotate(frame, predictor.predict_frame(frame))
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ESC_KEY):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Facial emotion inference.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=Path, help="Path to an image file.")
    mode.add_argument("--webcam", action="store_true", help="Use the live webcam.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=None,
                        help="Image mode only: save an annotated copy here.")
    args = parser.parse_args()

    try:
        predictor = FacialEmotionPredictor(args.model_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.webcam:
        return run_webcam(predictor, args.camera_index)
    return run_image(predictor, args.image, args.output)


if __name__ == "__main__":
    sys.exit(main())