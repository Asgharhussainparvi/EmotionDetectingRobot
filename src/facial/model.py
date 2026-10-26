"""CNN architecture for facial emotion recognition (FER2013, 5 classes).

Input : (48, 48, 1) grayscale images, raw pixel values in [0, 255].
Output: softmax probabilities in the order of CANONICAL_EMOTIONS.

Normalization (Rescaling) and data augmentation live inside the model so
training and inference share the same preprocessing. Keras augmentation
layers are active only when the model is called with training=True.
"""

from __future__ import annotations

import os

# Optional: hides TensorFlow's informational startup logs (not errors).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

from src.common.emotions import CANONICAL_EMOTIONS  # noqa: E402

IMAGE_SIDE = 48
INPUT_SHAPE = (IMAGE_SIDE, IMAGE_SIDE, 1)
NUM_CLASSES = len(CANONICAL_EMOTIONS)

BLOCK_FILTERS = (32, 64, 128)
DENSE_UNITS = 128
BLOCK_DROPOUT = 0.25
HEAD_DROPOUT = 0.5

AUG_ROTATION_FACTOR = 0.05  # fraction of a full turn, about +/- 18 degrees
AUG_ZOOM_FACTOR = 0.1
AUG_TRANSLATION_FACTOR = 0.1


def _conv_block(x, filters: int):
    """Two Conv-BatchNorm-ReLU layers followed by pooling and dropout."""
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    return layers.Dropout(BLOCK_DROPOUT)(x)


def build_model(seed: int = 42) -> keras.Model:
    """Build the (uncompiled) facial emotion CNN."""
    inputs = keras.Input(shape=INPUT_SHAPE, name="face_gray_48x48")

    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    x = layers.RandomFlip("horizontal", seed=seed)(x)
    x = layers.RandomRotation(AUG_ROTATION_FACTOR, seed=seed)(x)
    x = layers.RandomZoom(AUG_ZOOM_FACTOR, seed=seed)(x)
    x = layers.RandomTranslation(
        AUG_TRANSLATION_FACTOR, AUG_TRANSLATION_FACTOR, seed=seed
    )(x)

    for filters in BLOCK_FILTERS:
        x = _conv_block(x, filters)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(DENSE_UNITS, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(HEAD_DROPOUT)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="emotion_probs")(x)

    return keras.Model(inputs, outputs, name="facial_emotion_cnn")


def main() -> None:
    import numpy as np

    model = build_model()
    model.summary()

    # Structural check only. The model is UNTRAINED, so these outputs carry
    # no meaning as emotion predictions; we only verify shapes and behavior.
    rng = np.random.default_rng(0)
    dummy = rng.integers(0, 256, size=(4, IMAGE_SIDE, IMAGE_SIDE, 1), dtype=np.uint8)

    first = model(dummy, training=False).numpy()
    second = model(dummy, training=False).numpy()

    print("\n--- Structural checks (untrained model, dummy noise input) ---")
    print(f"Output shape               : {first.shape} (expected (4, {NUM_CLASSES}))")
    print(f"Row sums (expected ~1.0)   : {first.sum(axis=1)}")
    print(f"Deterministic at inference : {bool(np.allclose(first, second))}")
    print(f"Class order                : {CANONICAL_EMOTIONS}")


if __name__ == "__main__":
    main()