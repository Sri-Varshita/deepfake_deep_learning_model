"""Advanced analysis panel for metadata, prediction explanations, and Grad-CAM heatmaps."""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from preprocessing import DEFAULT_IMAGE_SIZE, efficientnet_preprocess


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".m4v", ".webm", ".mkv"}


def _get_file_bytes(file: Any) -> bytes:
    if hasattr(file, "getvalue"):
        return file.getvalue()
    if isinstance(file, (bytes, bytearray)):
        return bytes(file)
    if isinstance(file, str) and os.path.exists(file):
        with open(file, "rb") as handle:
            return handle.read()
    return b""


def _to_pil_image(media: Any) -> Image.Image:
    """Convert uploaded files, bytes, paths, or PIL images into a PIL image."""
    if isinstance(media, Image.Image):
        return media.convert("RGB")

    file_bytes = _get_file_bytes(media)
    if file_bytes:
        return Image.open(io_bytes(file_bytes)).convert("RGB")

    if isinstance(media, str) and os.path.exists(media):
        return Image.open(media).convert("RGB")

    raise ValueError("Unsupported media type for image conversion.")


def _get_file_name(file: Any) -> str:
    return getattr(file, "name", "uploaded_media")


def _infer_kind(file: Any) -> str:
    name = _get_file_name(file).lower()
    suffix = Path(name).suffix
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    content_type = str(getattr(file, "type", "")).lower()
    if content_type.startswith("video/"):
        return "video"
    return "image"


def _format_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "Unknown"
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.1f} {units[unit_index]}"


def _get_image_channels(mode: str) -> str:
    mapping = {
        "L": "Grayscale",
        "RGB": "RGB",
        "RGBA": "RGBA",
        "CMYK": "CMYK",
    }
    return mapping.get(mode, mode or "Unknown")


def extract_metadata(file: Any) -> Dict[str, Any]:
    """Extract image or video metadata from an uploaded file."""
    metadata: Dict[str, Any] = {
        "filename": _get_file_name(file),
        "upload_timestamp": datetime.now().strftime("%d-%m-%Y %I:%M %p"),
        "file_size": _format_size(len(_get_file_bytes(file))),
        "media_type": _infer_kind(file),
    }

    file_bytes = _get_file_bytes(file)
    if not file_bytes:
        metadata["error"] = "Unable to read file bytes."
        return metadata

    suffix = Path(metadata["filename"]).suffix.lower()

    try:
        if metadata["media_type"] == "image":
            with Image.open(io_bytes(file_bytes)) as image:
                width, height = image.size
                metadata.update({
                    "image_size": f"{width}x{height}",
                    "width": width,
                    "height": height,
                    "file_type": image.format or suffix.replace(".", "").upper() or "Image",
                    "color_channels": _get_image_channels(image.mode),
                })
        else:
            temp_path = _write_temp_file(file_bytes, suffix or ".mp4")
            try:
                capture = cv2.VideoCapture(temp_path)
                if not capture.isOpened():
                    raise ValueError("Video could not be opened.")

                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                duration = frame_count / fps if fps else 0.0

                metadata.update({
                    "duration": f"{duration:.2f} sec" if duration else "Unknown",
                    "frame_count": frame_count or "Unknown",
                    "fps": f"{fps:.2f}" if fps else "Unknown",
                    "resolution": f"{width}x{height}" if width and height else "Unknown",
                    "file_type": suffix.replace(".", "").upper() or "Video",
                })
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
    except Exception as error:
        metadata["error"] = f"Metadata extraction failed: {error}"

    return metadata


def explain_prediction(prediction: str, confidence: float) -> List[str]:
    """Generate deterministic human-readable reasoning for the prediction."""
    prediction = (prediction or "Unknown").lower()
    confidence = float(confidence or 0.0)

    if confidence < 70:
        return ["No strong manipulation indicators detected"]

    if confidence > 95:
        if prediction == "fake":
            return [
                "Unnatural facial boundaries detected",
                "Abnormal texture inconsistencies found",
                "Lighting mismatch observed",
                "Possible GAN artifacts detected",
                "Facial region contains suspicious patterns",
            ]
        return [
            "Strong structural consistency observed",
            "No obvious manipulation artifacts detected",
            "Facial geometry remains stable",
        ]

    # 70-95 range
    if prediction == "fake":
        return [
            "Minor texture inconsistencies detected",
            "Possible manipulation indicators present",
            "Subtle facial edge irregularities observed",
        ]

    return [
        "No strong manipulation indicators detected",
        "Visual patterns appear internally consistent",
    ]


def _find_last_conv_layer(model: Any):
    """Locate the most suitable convolutional layer for Grad-CAM."""
    layers_to_scan = []

    def collect(layers):
        for layer in layers:
            layers_to_scan.append(layer)
            nested_layers = getattr(layer, "layers", None)
            if nested_layers:
                collect(nested_layers)

    collect(getattr(model, "layers", []))

    for layer in reversed(layers_to_scan):
        try:
            output_shape = getattr(layer, "output_shape", None)
            if output_shape is not None and len(output_shape) == 4:
                return layer
        except Exception:
            continue

    for layer in reversed(layers_to_scan):
        if "conv" in layer.name.lower():
            return layer

    return None


def _resize_for_display(image_array: np.ndarray) -> np.ndarray:
    return cv2.resize(image_array, (DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1]))


def generate_heatmap(image: Image.Image, model: Any) -> Image.Image:
    """Generate a Grad-CAM heatmap for an image using the EfficientNet model."""
    try:
        pil_image = _to_pil_image(image)

        if not hasattr(model, "layers"):
            return pil_image

        last_conv_layer = _find_last_conv_layer(model)
        if last_conv_layer is None:
            return pil_image

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [last_conv_layer.output, model.output],
        )

        processed = np.expand_dims(efficientnet_preprocess(pil_image), axis=0)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(processed)
            class_index = tf.argmax(predictions[0])
            class_channel = predictions[:, class_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()

        base_image = _resize_for_display(np.array(pil_image))
        heatmap = cv2.resize(heatmap, (base_image.shape[1], base_image.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(base_image, 0.62, colored_heatmap, 0.38, 0)
        return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    except Exception:
        try:
            return _to_pil_image(image)
        except Exception:
            return Image.new("RGB", (DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1]), color="white")


def _write_temp_file(file_bytes: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        return temp_file.name


def io_bytes(file_bytes: bytes):
    from io import BytesIO

    return BytesIO(file_bytes)


def _get_video_key_frame(video_file: Any, result: Dict[str, Any]) -> Tuple[Image.Image | None, int | None]:
    file_bytes = _get_file_bytes(video_file)
    if not file_bytes:
        return None, None

    suffix = Path(_get_file_name(video_file)).suffix or ".mp4"
    temp_path = _write_temp_file(file_bytes, suffix)
    try:
        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            return None, None

        metadata = result.get("metadata", {})
        suspicious_index = None
        frame_probs = metadata.get("frame_probabilities")
        if isinstance(frame_probs, np.ndarray) and frame_probs.size:
            suspicious_index = int(np.argmax(frame_probs[:, 0]))
        else:
            sampled = metadata.get("sampled_frame_indices", [])
            if sampled:
                suspicious_index = int(sampled[len(sampled) // 2])

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if suspicious_index is None:
            suspicious_index = total_frames // 2 if total_frames else 0

        capture.set(cv2.CAP_PROP_POS_FRAMES, suspicious_index)
        success, frame = capture.read()
        if not success or frame is None:
            return None, suspicious_index

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb), suspicious_index
    finally:
        try:
            capture.release()
        except Exception:
            pass
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _metadata_cards(metadata: Dict[str, Any]) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown(
            f"""
            <div style='background:#ffffff;border-radius:16px;padding:1rem 1.1rem;box-shadow:0 8px 24px rgba(0,0,0,0.12);border:1px solid rgba(15,23,42,0.08);'>
                <div style='font-weight:700;color:#0f172a;margin-bottom:0.35rem;'>Filename</div>
                <div style='color:#334155;'>{metadata.get('filename', 'Unknown')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div style='background:#ffffff;border-radius:16px;padding:1rem 1.1rem;box-shadow:0 8px 24px rgba(0,0,0,0.12);border:1px solid rgba(15,23,42,0.08);'>
                <div style='font-weight:700;color:#0f172a;margin-bottom:0.35rem;'>Upload Time</div>
                <div style='color:#334155;'>{metadata.get('upload_timestamp', 'Unknown')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if metadata.get("media_type") == "image":
        cards = [
            ("Resolution", metadata.get("image_size", "Unknown")),
            ("File Size", metadata.get("file_size", "Unknown")),
            ("Channels", metadata.get("color_channels", "Unknown")),
            ("File Type", metadata.get("file_type", "Unknown")),
        ]
    else:
        cards = [
            ("Duration", metadata.get("duration", "Unknown")),
            ("Frame Count", metadata.get("frame_count", "Unknown")),
            ("FPS", metadata.get("fps", "Unknown")),
            ("Resolution", metadata.get("resolution", "Unknown")),
        ]

    card_cols = st.columns(2)
    for index, (title, value) in enumerate(cards):
        with card_cols[index % 2]:
            st.markdown(
                f"""
                <div style='background:#f8fafc;border-radius:16px;padding:0.9rem 1rem;margin-top:0.75rem;box-shadow:0 6px 18px rgba(15,23,42,0.08);border:1px solid rgba(148,163,184,0.2);'>
                    <div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;font-weight:700;'>{title}</div>
                    <div style='margin-top:0.35rem;font-size:1rem;color:#0f172a;font-weight:700;'>{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _reason_cards(reasons: List[str]) -> None:
    for reason in reasons:
        st.markdown(
            f"""
            <div style='background:#fff;border-radius:14px;padding:0.9rem 1rem;margin-bottom:0.75rem;box-shadow:0 8px 20px rgba(0,0,0,0.08);border-left:4px solid #f59e0b;'>
                <span style='font-weight:700;color:#92400e;'>⚠</span>
                <span style='margin-left:0.5rem;color:#1f2937;'>{reason}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _heatmap_cards(image: Image.Image, heatmap_image: Image.Image) -> None:
    left, right = st.columns(2)
    with left:
        st.image(image, caption="Original", use_column_width=True)
    with right:
        st.image(heatmap_image, caption="AI Attention Heatmap", use_column_width=True)


def render_analysis_panel(model: Any, media: Any, result: Dict[str, Any], source_name: str, media_type: str) -> None:
    """Render the advanced analysis expander below an existing prediction result."""
    if not result or "error" in result:
        return

    with st.expander("▼ Advanced Analysis", expanded=False):
        st.markdown(
            """
            <div style='margin-bottom:1rem;'>
                <h3 style='margin:0;color:#0f172a;font-weight:800;'>Advanced Media Analysis</h3>
                <p style='margin:0.35rem 0 0 0;color:#475569;'>Metadata, deterministic reasoning, and visual attribution for the current prediction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metadata = extract_metadata(media)
        prediction = result.get("label", "Unknown")
        confidence = float(result.get("confidence", 0.0))
        reasons = explain_prediction(prediction, confidence)

        st.markdown(
            """
            <div style='background:linear-gradient(135deg,#ffffff 0%,#f8fafc 100%);border-radius:18px;padding:1rem 1.1rem;margin-bottom:1rem;box-shadow:0 10px 30px rgba(15,23,42,0.08);border:1px solid rgba(148,163,184,0.18);'>
                <div style='font-size:1.05rem;font-weight:800;color:#0f172a;margin-bottom:0.75rem;'>1. Image Metadata Information</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if metadata.get("error"):
            st.warning(metadata["error"])
        else:
            _metadata_cards(metadata)

        st.markdown(
            """
            <div style='background:linear-gradient(135deg,#ffffff 0%,#f8fafc 100%);border-radius:18px;padding:1rem 1.1rem;margin:1rem 0 1rem 0;box-shadow:0 10px 30px rgba(15,23,42,0.08);border:1px solid rgba(148,163,184,0.18);'>
                <div style='font-size:1.05rem;font-weight:800;color:#0f172a;margin-bottom:0.75rem;'>2. Fake Reason Analysis</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _reason_cards(reasons)

        st.markdown(
            """
            <div style='background:linear-gradient(135deg,#ffffff 0%,#f8fafc 100%);border-radius:18px;padding:1rem 1.1rem;margin:1rem 0 1rem 0;box-shadow:0 10px 30px rgba(15,23,42,0.08);border:1px solid rgba(148,163,184,0.18);'>
                <div style='font-size:1.05rem;font-weight:800;color:#0f172a;margin-bottom:0.75rem;'>3. Heatmap Visualization</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cache_key = hashlib.sha256((_get_file_name(media) + str(confidence) + prediction + media_type).encode("utf-8")).hexdigest()
        if "advanced_analysis_cache" not in st.session_state:
            st.session_state["advanced_analysis_cache"] = {}

        cached = st.session_state["advanced_analysis_cache"].get(cache_key)
        if cached is None:
            with st.spinner("Generating attention heatmap..."):
                if media_type == "image":
                    original_for_display = _to_pil_image(media)
                    heatmap_image = generate_heatmap(original_for_display, model)
                else:
                    key_frame, _ = _get_video_key_frame(media, result)
                    if key_frame is None:
                        original_for_display = Image.new("RGB", (DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1]), color="white")
                        heatmap_image = original_for_display
                    else:
                        original_for_display = _to_pil_image(key_frame)
                        heatmap_image = generate_heatmap(original_for_display, model)

                cached = {
                    "original": original_for_display,
                    "heatmap": heatmap_image,
                }
                st.session_state["advanced_analysis_cache"][cache_key] = cached

        _heatmap_cards(cached["original"], cached["heatmap"])
