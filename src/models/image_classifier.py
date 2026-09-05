"""Optional ImageNet image classification with Keras ResNet50.

TensorFlow is deliberately imported lazily so the tabular pipeline remains
usable in environments that do not install the optional image dependencies.
"""

from pathlib import Path
from typing import Any

import numpy as np

try:  # Keep importing ``src.models`` possible without TensorFlow installed.
    import tensorflow as tf
except (ImportError, ModuleNotFoundError):  # pragma: no cover - environment dependent
    tf = None


def _require_tensorflow() -> Any:
    if tf is None:
        raise ImportError(
            "Image classification requires TensorFlow. Install it with "
            "`pip install tensorflow` (or install this project's requirements)."
        )
    return tf


def preprocess_image(image: Any, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load and preprocess an image for ImageNet ResNet50.

    ``image`` may be a filesystem path, a Pillow image, or a numpy array.
    The returned array has shape ``(1, height, width, 3)`` and float32 values.
    """
    tensorflow = _require_tensorflow()
    if len(target_size) != 2 or min(target_size) <= 0:
        raise ValueError("target_size must contain two positive integers")

    if isinstance(image, (str, Path)):
        image = tensorflow.keras.utils.load_img(image, target_size=target_size)
        array = tensorflow.keras.utils.img_to_array(image)
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError("image must be an HxWx3/4 array or an image path")
        if array.shape[-1] == 4:
            array = array[..., :3]
        array = tensorflow.image.resize(array, target_size).numpy()

    array = np.asarray(array, dtype=np.float32)
    array = tensorflow.keras.applications.resnet50.preprocess_input(array)
    return np.expand_dims(array, axis=0)


def load_resnet50(weights: str | None = "imagenet", include_top: bool = True):
    """Construct a Keras ResNet50 model, raising a helpful optional-dependency error."""
    tensorflow = _require_tensorflow()
    return tensorflow.keras.applications.ResNet50(
        weights=weights, include_top=include_top
    )


def predict_image(model, image: Any, top_k: int = 5) -> list[tuple[str, float]]:
    """Return decoded ImageNet predictions as ``(label, probability)`` pairs."""
    _require_tensorflow()
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    scores = np.asarray(model.predict(preprocess_image(image), verbose=0))
    if scores.ndim != 2 or scores.shape[0] != 1:
        raise ValueError("model must return a single batch of class probabilities")
    decoded = tf.keras.applications.resnet50.decode_predictions(scores, top=top_k)[0]
    return [(label, float(probability)) for _, label, probability in decoded]
