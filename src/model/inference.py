"""Prediction helpers for image, webcam, and video inputs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image

from preprocessing import efficientnet_preprocess


def _normalize_prediction_output(prediction: Any) -> np.ndarray:
    """Convert model output to a stable two-class probability array."""
    probabilities = np.asarray(prediction, dtype=np.float32)

    if probabilities.ndim == 0:
        fake_prob = float(probabilities)
        probabilities = np.array([[fake_prob, 1.0 - fake_prob]], dtype=np.float32)
    elif probabilities.ndim == 1:
        if probabilities.size >= 2:
            probabilities = probabilities[:2][np.newaxis, :]
        else:
            fake_prob = float(probabilities[0])
            probabilities = np.array([[fake_prob, 1.0 - fake_prob]], dtype=np.float32)

    if probabilities.shape[-1] != 2:
        if probabilities.shape[-1] == 1:
            fake_prob = probabilities[..., 0]
            probabilities = np.stack([fake_prob, 1.0 - fake_prob], axis=-1)
        else:
            probabilities = probabilities[..., :2]

    probabilities = np.clip(probabilities, 1e-6, None)
    probabilities = probabilities / probabilities.sum(axis=-1, keepdims=True)
    return probabilities


def _build_result(probabilities: np.ndarray, source_type: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Create a consistent prediction payload for the UI."""
    probs = np.asarray(probabilities, dtype=np.float32).squeeze()
    if probs.ndim == 0:
        probs = np.array([float(probs), 1.0 - float(probs)], dtype=np.float32)

    fake_confidence = float(probs[0]) * 100.0
    real_confidence = float(probs[1]) * 100.0
    label = "Fake" if fake_confidence >= real_confidence else "Real"
    confidence = max(fake_confidence, real_confidence)

    return {
        "source_type": source_type,
        "label": label,
        "confidence": confidence,
        "fake_confidence": fake_confidence,
        "real_confidence": real_confidence,
        "probabilities": probs.astype(np.float32),
        "metadata": metadata or {},
    }


def predict_image(model: Any, image: Image.Image, preprocess_method: str = "efficientnet") -> Dict[str, Any]:
    """Predict a single image using the locked EfficientNet preprocessing path."""
    processed = efficientnet_preprocess(image)
    if processed is None:
        raise ValueError("Could not preprocess the uploaded image.")

    batch = np.expand_dims(processed, axis=0)
    probabilities = _normalize_prediction_output(model.predict(batch, verbose=0))[0]
    return _build_result(probabilities, source_type="image")


def predict_video(
    model: Any,
    video_file: Any,
    preprocess_method: str = "efficientnet",
    max_frames: int = 12,
) -> Dict[str, Any]:
    """Predict a video by sampling several frames across its duration."""
    suffix = Path(getattr(video_file, "name", "upload.mp4")).suffix or ".mp4"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(video_file.getvalue())
            temp_path = temp_file.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise ValueError("The uploaded video could not be opened. Please try a different file format.")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0:
            frame_indices = np.linspace(0, max(total_frames - 1, 0), num=min(max_frames, total_frames), dtype=int).tolist()
        else:
            frame_indices = list(range(0, max_frames * 15, 15))

        processed_frames: List[np.ndarray] = []
        sampled_indices: List[int] = []

        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success or frame is None:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            processed = efficientnet_preprocess(pil_frame)
            if processed is None:
                continue

            processed_frames.append(processed)
            sampled_indices.append(int(frame_index))

        capture.release()

        if not processed_frames:
            raise ValueError("No usable frames could be extracted from the uploaded video.")

        batch = np.stack(processed_frames, axis=0)
        frame_probabilities = _normalize_prediction_output(model.predict(batch, verbose=0))
        probabilities = frame_probabilities.mean(axis=0)

        return _build_result(
            probabilities,
            source_type="video",
            metadata={
                "total_frames": total_frames,
                "sampled_frames": len(processed_frames),
                "sampled_frame_indices": sampled_indices,
                "frame_probabilities": frame_probabilities,
            },
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass