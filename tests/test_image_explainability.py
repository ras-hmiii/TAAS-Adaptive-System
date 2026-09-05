import importlib.util

import numpy as np
import pytest

from src.explainability.gradcam import overlay_heatmap
from src.models.image_classifier import preprocess_image


@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is not None,
    reason="TensorFlow is installed",
)
def test_missing_tensorflow_error_is_actionable():
    with pytest.raises(ImportError, match="requires TensorFlow"):
        preprocess_image(np.zeros((8, 8, 3), dtype=np.uint8))


def test_overlay_heatmap_validates_inputs_without_tensorflow():
    pytest.importorskip("PIL")
    with pytest.raises(ValueError, match="2D"):
        overlay_heatmap(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 1)))


@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None,
    reason="TensorFlow is an optional image-classification dependency",
)
def test_preprocess_image_shape():
    result = preprocess_image(np.zeros((32, 32, 3), dtype=np.uint8))
    assert result.shape == (1, 224, 224, 3)
    assert result.dtype == np.float32
