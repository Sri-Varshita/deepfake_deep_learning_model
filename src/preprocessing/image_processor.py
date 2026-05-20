"""Image preprocessing functions."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input as keras_efficientnet_preprocess


DEFAULT_IMAGE_SIZE = (128, 128)


def efficientnet_preprocess(image: Image.Image | np.ndarray) -> np.ndarray:
    """Resize and preprocess an image using the EfficientNet pipeline."""
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert("RGB"))
    else:
        img_array = np.asarray(image)

    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.ndim == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    img_resized = cv2.resize(img_array, DEFAULT_IMAGE_SIZE)
    return keras_efficientnet_preprocess(img_resized.astype(np.float32))


def preprocess_image(image: Image.Image, method: str = "efficientnet") -> Optional[np.ndarray]:
    """Convert PIL image to EfficientNet-preprocessed numpy array.

    The method argument is retained for backward compatibility, but the
    application always uses the EfficientNet pipeline now.
    """
    try:
        return efficientnet_preprocess(image)
    except Exception as e:
        import streamlit as st
        st.error(f"❌ Error preprocessing image: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None
