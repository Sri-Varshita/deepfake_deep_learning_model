"""
Enhanced Inference Pipeline - Implements Strict Analysis Pipeline

Step 1: Input validation
Step 2: Preprocessing
Step 3: Facial consistency analysis
Step 4: Artifact detection
Step 5: Temporal analysis (for videos)
Step 6: Multi-model verification
Step 7: Decision engine with confidence rules
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import cv2
from PIL import Image
import streamlit as st

from preprocessing import preprocess_image
from analysis import (
    InputValidator,
    FacialConsistencyAnalyzer,
    ArtifactDetector,
    EnsembleDetector,
    DecisionEngine
)


# Initialize analysis modules (cached)
@st.cache_resource
def initialize_analysis_pipeline() -> Dict[str, Any]:
    """Initialize all analysis modules."""
    return {
        "input_validator": InputValidator(),
        "facial_analyzer": FacialConsistencyAnalyzer(),
        "artifact_detector": ArtifactDetector(),
        "ensemble": EnsembleDetector(),
        "decision_engine": DecisionEngine()
    }


def analyze_image_complete(
    model: Any,
    image: Image.Image,
    preprocess_method: str = "efficientnet"
) -> Dict[str, Any]:
    """
    Complete image analysis following strict pipeline.

    Step 1: Input validation
    Step 2: Preprocessing
    Step 3: Facial consistency analysis
    Step 4: Artifact detection
    Step 6: Multi-model verification (if ensemble available)
    Step 7: Advanced decision engine

    Args:
        model: Primary deepfake detection model
        image: PIL Image to analyze
        preprocess_method: Locked preprocessing method (EfficientNet)

    Returns:
        Comprehensive analysis result
    """

    pipeline = initialize_analysis_pipeline()
    validator = pipeline["input_validator"]
    facial_analyzer = pipeline["facial_analyzer"]
    artifact_detector = pipeline["artifact_detector"]
    decision_engine = pipeline["decision_engine"]

    # ===== STEP 1: INPUT VALIDATION =====
    st.write("🔍 Step 1: Validating input quality...")
    quality_result = validator.validate_image(image)

    # Convert to numpy array for analysis
    img_array = np.array(image.convert("RGB"))

    # ===== STEP 2: PREPROCESSING =====
    st.write("🔄 Step 2: Preprocessing image...")
    processed = preprocess_image(image, method=preprocess_method)

    if processed is None:
        return {
            "error": "Could not preprocess the image",
            "label": "Error",
            "confidence": 0.0
        }

    # ===== STEP 3: FACIAL CONSISTENCY ANALYSIS =====
    st.write("👤 Step 3: Analyzing facial consistency...")
    facial_result = facial_analyzer.analyze_image(img_array)

    # ===== STEP 4: ARTIFACT DETECTION =====
    st.write("🔎 Step 4: Detecting artifacts...")
    artifact_result = artifact_detector.detect_artifacts(img_array)

    # ===== STEP 5: MODEL PREDICTION =====
    st.write("🤖 Step 5: Running model prediction...")
    batch = np.expand_dims(processed, axis=0)
    model_output = model.predict(batch, verbose=0)
    model_result = _build_model_result(model_output)

    # ===== STEP 6: MULTI-MODEL VERIFICATION (Optional) =====
    ensemble_info = None
    if hasattr(model, '_is_ensemble') and model._is_ensemble:
        st.write("🔀 Step 6: Ensemble verification...")
        # Register the primary model if ensemble available
        # This would be done during model loading
        ensemble_result = pipeline["ensemble"].predict_ensemble(batch)
        ensemble_info = ensemble_result

    # ===== STEP 7: ADVANCED DECISION ENGINE =====
    st.write("⚖️ Step 7: Making final decision...")
    final_decision = decision_engine.make_decision(
        model_prediction=model_result,
        input_quality=quality_result,
        facial_consistency=facial_result,
        artifact_detection=artifact_result,
        ensemble_info=ensemble_info
    )

    # Combine all results
    complete_result = {
        "final_prediction": final_decision,
        "model_prediction": model_result,
        "input_quality": quality_result,
        "facial_analysis": facial_result,
        "artifact_analysis": artifact_result,
        "ensemble_analysis": ensemble_info,
        "analysis_pipeline": "Strict Analysis Pipeline (7 Steps)"
    }

    return complete_result


def analyze_video_complete(
    model: Any,
    video_file: Any,
    preprocess_method: str = "efficientnet",
    max_frames: int = 12
) -> Dict[str, Any]:
    """
    Complete video analysis following strict pipeline.

    Args:
        model: Primary deepfake detection model
        video_file: Uploaded video file
        preprocess_method: Locked preprocessing method (EfficientNet)
        max_frames: Maximum frames to sample

    Returns:
        Comprehensive video analysis result
    """

    pipeline = initialize_analysis_pipeline()
    validator = pipeline["input_validator"]
    facial_analyzer = pipeline["facial_analyzer"]
    artifact_detector = pipeline["artifact_detector"]
    decision_engine = pipeline["decision_engine"]

    # ===== STEP 1: INPUT VALIDATION (Video) =====
    st.write("🔍 Step 1: Validating video quality...")
    suffix = Path(getattr(video_file, "name", "upload.mp4")).suffix or ".mp4"

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(video_file.getvalue())
            temp_path = temp_file.name

        quality_result = validator.validate_video(temp_path, sample_frames=5)

        # ===== STEP 2: PREPROCESSING (Extract Frames) =====
        st.write("🔄 Step 2: Extracting and preprocessing frames...")
        capture = cv2.VideoCapture(temp_path)

        if not capture.isOpened():
            return {
                "error": "Could not open video file",
                "label": "Error",
                "confidence": 0.0
            }

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, max(total_frames - 1, 0), num=min(max_frames, total_frames), dtype=int).tolist()

        frames = []
        processed_frames = []

        for frame_idx in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = capture.read()
            if ret and frame is not None:
                frames.append(frame)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)
                processed = preprocess_image(pil_frame, method=preprocess_method)
                if processed is not None:
                    processed_frames.append(processed)

        capture.release()

        if not frames:
            return {
                "error": "No usable frames extracted",
                "label": "Error",
                "confidence": 0.0
            }

        # ===== STEP 3: FACIAL CONSISTENCY ANALYSIS (Temporal) =====
        st.write("👤 Step 3: Analyzing facial consistency across frames...")
        temporal_result = facial_analyzer.analyze_video_frames(frames)

        # ===== STEP 4: ARTIFACT DETECTION (Frame-to-Frame) =====
        st.write("🔎 Step 4: Detecting frame artifacts...")
        frame_anomalies = artifact_detector.detect_frame_anomalies(frames)

        # ===== STEP 5: MODEL PREDICTION (Video Average) =====
        st.write("🤖 Step 5: Running model predictions...")
        if processed_frames:
            batch = np.stack(processed_frames, axis=0)
            model_output = model.predict(batch, verbose=0)
            # Average across frames
            frame_probs = _normalize_prediction_output(model_output)
            avg_probs = frame_probs.mean(axis=0)
            model_result = _build_model_result(np.array([avg_probs]))
        else:
            model_result = {
                "label": "Error",
                "confidence": 0.0,
                "fake_confidence": 50.0,
                "real_confidence": 50.0
            }

        # ===== STEP 6: MULTI-MODEL VERIFICATION (Optional) =====
        ensemble_info = None

        # ===== STEP 7: ADVANCED DECISION ENGINE =====
        st.write("⚖️ Step 7: Making final decision...")

        # Adjust facial consistency for temporal analysis
        temporal_consistency = temporal_result.get("temporal_consistency", 0.5)
        facial_consistency_for_decision = {
            "consistency_score": temporal_consistency,
            "issues": temporal_result.get("issues", []),
            "facial_artifacts": frame_anomalies.get("anomalies", [])
        }

        final_decision = decision_engine.make_decision(
            model_prediction=model_result,
            input_quality=quality_result,
            facial_consistency=facial_consistency_for_decision,
            artifact_detection={
                "artifact_score": frame_anomalies.get("anomaly_score", 0.5),
                "has_artifacts": len(frame_anomalies.get("anomalies", [])) > 0,
                "artifacts": frame_anomalies.get("anomalies", [])
            },
            ensemble_info=ensemble_info
        )

        # Combine all results
        complete_result = {
            "final_prediction": final_decision,
            "model_prediction": model_result,
            "input_quality": quality_result,
            "temporal_analysis": temporal_result,
            "frame_anomalies": frame_anomalies,
            "video_metadata": {
                "total_frames": total_frames,
                "analyzed_frames": len(frames),
                "frame_indices": frame_indices
            },
            "analysis_pipeline": "Strict Video Analysis Pipeline (7 Steps)"
        }

        return complete_result

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _normalize_prediction_output(prediction: Any) -> np.ndarray:
    """Convert model output to stable 2-class probability array."""
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
    elif probabilities.ndim == 2:
        if probabilities.shape[-1] != 2:
            if probabilities.shape[-1] == 1:
                fake_prob = probabilities[:, 0]
                probabilities = np.stack([fake_prob, 1.0 - fake_prob], axis=-1)
            else:
                probabilities = probabilities[:, :2]

    probabilities = np.clip(probabilities, 1e-6, None)
    probabilities = probabilities / probabilities.sum(axis=-1, keepdims=True)
    return probabilities


def _build_model_result(probabilities: np.ndarray) -> Dict[str, Any]:
    """Build model result dictionary."""
    probs = np.asarray(probabilities, dtype=np.float32).squeeze()
    if probs.ndim == 0:
        probs = np.array([float(probs), 1.0 - float(probs)], dtype=np.float32)

    fake_confidence = float(probs[0]) * 100.0
    real_confidence = float(probs[1]) * 100.0
    label = "Fake" if fake_confidence >= real_confidence else "Real"
    confidence = max(fake_confidence, real_confidence)

    return {
        "label": label,
        "confidence": confidence,
        "fake_confidence": fake_confidence,
        "real_confidence": real_confidence,
        "probabilities": probs.astype(np.float32)
    }
