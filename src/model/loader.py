"""Model loading utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from tensorflow import keras

from .architecture import attention_block, build_effatt_model


IMAGE_SIZE = 128
MODEL_REPO_ID = "CemRoot/deepfake-detection-model"
MODEL_FILENAME = "best_model_effatt.h5"


def _rescale_gap(tensors):
    return tensors[0] / tensors[1]


@dataclass
class FallbackDeepfakeModel:
    """Simple local fallback so the UI still works if the remote model is unavailable."""

    is_fallback: bool = True
    fallback_reason: str = ""

    def predict(self, inputs, verbose=0):
        batch = np.asarray(inputs, dtype=np.float32)
        if batch.ndim == 3:
            batch = batch[np.newaxis, ...]

        if batch.size == 0:
            return np.array([[0.5, 0.5]], dtype=np.float32)

        if batch.max() > 1.5:
            batch = batch / 255.0

        gray = batch.mean(axis=-1)
        gradient_x = np.abs(np.diff(gray, axis=2)).mean(axis=(1, 2)) if gray.shape[2] > 1 else np.zeros(batch.shape[0], dtype=np.float32)
        gradient_y = np.abs(np.diff(gray, axis=1)).mean(axis=(1, 2)) if gray.shape[1] > 1 else np.zeros(batch.shape[0], dtype=np.float32)
        texture = (gradient_x + gradient_y) / 2.0
        color_std = batch.std(axis=(1, 2, 3))
        brightness_delta = np.abs(batch.mean(axis=(1, 2, 3)) - 0.5)

        fake_score = 0.35 + 0.85 * texture + 0.25 * (1.0 - color_std) + 0.15 * brightness_delta
        fake_score = np.clip(fake_score, 0.02, 0.98)
        real_score = 1.0 - fake_score
        return np.stack([fake_score, real_score], axis=-1).astype(np.float32)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model from Hugging Face with a local fallback.

    Returns:
        Loaded Keras model, or a lightweight fallback predictor if the remote
        model cannot be downloaded or deserialized.
    """
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model",
        )

        try:
            model = keras.models.load_model(
                model_path,
                custom_objects={
                    "attention_block": attention_block,
                    "RescaleGAP": _rescale_gap,
                },
                compile=False,
            )
            model.is_fallback = False
            return model
        except Exception:
            try:
                model = build_effatt_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
                model.load_weights(model_path)
                model.is_fallback = False
                return model
            except Exception as weight_error:
                return FallbackDeepfakeModel(fallback_reason=f"Model weights could not be loaded: {weight_error}")

    except Exception as download_error:
        return FallbackDeepfakeModel(fallback_reason=f"Model download failed: {download_error}")
