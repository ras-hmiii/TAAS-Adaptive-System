"""Grad-CAM explanations for Keras image classifiers."""

from pathlib import Path
from typing import Any

import numpy as np

from src.models.image_classifier import _require_tensorflow, preprocess_image


def _last_convolutional_layer(model):
    for layer in reversed(model.layers):
        shape = getattr(layer.output, "shape", ())
        if len(shape) == 4:
            return layer
    raise ValueError("Could not find a 4D convolutional layer; provide last_conv_layer_name")


def make_gradcam_heatmap(
    image: Any,
    model,
    last_conv_layer_name: str | None = None,
    pred_index: int | None = None,
) -> np.ndarray:
    """Create a normalized (0..1) 2D Grad-CAM heatmap for one image."""
    tensorflow = _require_tensorflow()
    inputs = preprocess_image(image)
    layer = (
        model.get_layer(last_conv_layer_name)
        if last_conv_layer_name
        else _last_convolutional_layer(model)
    )
    grad_model = tensorflow.keras.models.Model(model.inputs, [layer.output, model.output])
    with tensorflow.GradientTape() as tape:
        convolution_output, predictions = grad_model(inputs)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        if predictions.shape[-1] == 1:
            class_channel = predictions[:, 0]
        else:
            index = int(tensorflow.argmax(predictions[0])) if pred_index is None else pred_index
            if index < 0 or index >= predictions.shape[-1]:
                raise ValueError("pred_index is outside the model's output range")
            class_channel = predictions[:, index]
    gradients = tape.gradient(class_channel, convolution_output)
    pooled_gradients = tensorflow.reduce_mean(gradients, axis=(1, 2))
    heatmap = tensorflow.reduce_sum(
        convolution_output[0] * pooled_gradients[0], axis=-1
    )
    heatmap = tensorflow.maximum(heatmap, 0)
    maximum = tensorflow.reduce_max(heatmap)
    heatmap = tensorflow.where(maximum > 0, heatmap / maximum, heatmap)
    return heatmap.numpy().astype(np.float32)


def _as_rgb_array(image: Any) -> np.ndarray:
    if isinstance(image, (str, Path)):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Image overlays require Pillow. Install it with `pip install Pillow`.") from exc
        image = Image.open(image)
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError("image must be an HxWx3/4 array or an image path")
    return array[..., :3]


def overlay_heatmap(image: Any, heatmap: np.ndarray, alpha: float = 0.4):
    """Blend a heatmap over an image and return a Pillow RGB image."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Image overlays require Pillow. Install it with `pip install Pillow`.") from exc
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    base = _as_rgb_array(image)
    if np.asarray(heatmap).ndim != 2:
        raise ValueError("heatmap must be a 2D array")
    base_image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
    resized = Image.fromarray(
        (np.clip(np.asarray(heatmap, dtype=np.float32), 0, 1) * 255).astype(np.uint8),
        mode="L",
    ).resize(base_image.size, Image.Resampling.BILINEAR)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    # A compact blue-to-red colormap avoids a matplotlib dependency.
    color = np.stack((values, np.minimum(values * 2, 2 - values), 1 - values), axis=-1)
    blended = (1 - alpha * values[..., None]) * np.asarray(base_image, dtype=np.float32)
    blended += alpha * values[..., None] * (color * 255)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


overlay_gradcam = overlay_heatmap
