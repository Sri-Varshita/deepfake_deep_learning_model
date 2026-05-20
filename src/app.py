"""
DeepShield AI - Deepfake Detection System
Main Streamlit Application

Developed by: Emin Cem Koyluoglu
"""

from __future__ import annotations

from PIL import Image
import streamlit as st

from model import load_model, predict_image, predict_video
from ui import apply_custom_css, render_analysis_panel, render_footer, render_header, render_sidebar


DEFAULT_PREPROCESS_METHOD = "efficientnet"


st.set_page_config(
    page_title="DeepShield AI - Deepfake Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _render_prediction(result, source_name: str, show_debug: bool = False):
    """Render a consistent result panel for image, webcam, and video inputs."""
    fake_confidence = result["fake_confidence"]
    real_confidence = result["real_confidence"]
    confidence = result["confidence"]

    if result["label"] == "Fake":
        st.error(f"### 🚨 DETECTION RESULT: FAKE (AI-Generated) - {source_name}")
        st.markdown(
            "The input media exhibits characteristics consistent with **synthetic generation** by artificial intelligence systems."
        )
    else:
        st.success(f"### ✅ DETECTION RESULT: AUTHENTIC - {source_name}")
        st.markdown("The input media exhibits characteristics consistent with **genuine photographic content**.")

    st.markdown("---")
    st.markdown("**📈 Classification Confidence Scores:**")

    st.markdown("🚨 **Synthetic (AI-Generated):**")
    st.progress(fake_confidence / 100)
    st.write(f"**{fake_confidence:.2f}%**")

    st.markdown("")

    st.markdown("✅ **Authentic (Genuine):**")
    st.progress(real_confidence / 100)
    st.write(f"**{real_confidence:.2f}%**")

    st.markdown("---")
    st.info(f"**Classification Output:** {result['label']} (Confidence: {confidence:.2f}%)")

    if show_debug:
        st.markdown("**🔍 Model Output Diagnostics:**")
        st.write(f"- Probability Distribution: {result['probabilities']}")
        if result.get("metadata"):
            st.json(result["metadata"])


def _render_image_tab(model, show_debug: bool):
    uploaded_file = st.file_uploader(
        "Select an image file...",
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG, JPEG, or PNG format image for deepfake analysis",
        key="image_uploader",
    )

    if uploaded_file is None:
        st.info("👆 Please upload an image file and initiate analysis to view detection results.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    if show_debug:
        st.markdown("**🔍 Image Metadata:**")
        st.write(f"- File Format: {image.format}")
        st.write(f"- Color Mode: {image.mode}")
        st.write(f"- Dimensions: {image.size}")

    if st.button("🔍 Analyze Image", key="analyze_image", use_container_width=True):
        with st.spinner("🔄 Analyzing image..."):
            try:
                result = predict_image(model, image, DEFAULT_PREPROCESS_METHOD)
                _render_prediction(result, "Image", show_debug=show_debug)
                render_analysis_panel(model, uploaded_file, result, "Image", "image")
            except Exception as error:
                st.error(f"❌ Could not process the uploaded image: {error}")


def _render_webcam_tab(model, show_debug: bool):
    captured_image = st.camera_input(
        "Take a webcam snapshot for analysis",
        help="Use your device camera to capture a face or scene for deepfake analysis",
    )

    if captured_image is None:
        st.info("👆 Capture an image from your webcam to run detection.")
        return

    image = Image.open(captured_image).convert("RGB")
    st.image(image, caption="Webcam Snapshot", use_column_width=True)

    if st.button("🔍 Analyze Webcam Snapshot", key="analyze_webcam", use_container_width=True):
        with st.spinner("🔄 Analyzing webcam snapshot..."):
            try:
                result = predict_image(model, image, DEFAULT_PREPROCESS_METHOD)
                _render_prediction(result, "Webcam Snapshot", show_debug=show_debug)
                render_analysis_panel(model, captured_image, result, "Webcam Snapshot", "image")
            except Exception as error:
                st.error(f"❌ Could not process the webcam image: {error}")


def _render_video_tab(model, show_debug: bool):
    uploaded_video = st.file_uploader(
        "Select a video file...",
        type=["mp4", "mov", "avi", "m4v", "webm"],
        help="Upload a short video clip for frame-based deepfake analysis",
        key="video_uploader",
    )

    if uploaded_video is None:
        st.info("👆 Upload a video file to run frame-based detection.")
        return

    st.video(uploaded_video)

    if st.button("🔍 Analyze Video", key="analyze_video", use_container_width=True):
        with st.spinner("🔄 Extracting frames and analyzing video..."):
            try:
                result = predict_video(model, uploaded_video, DEFAULT_PREPROCESS_METHOD)
                _render_prediction(result, "Video", show_debug=show_debug)

                metadata = result.get("metadata", {})
                if metadata:
                    st.markdown("**🎞️ Video Sampling Summary:**")
                    st.write(f"- Total Frames: {metadata.get('total_frames', 'Unknown')}")
                    st.write(f"- Sampled Frames: {metadata.get('sampled_frames', 'Unknown')}")
                    sampled_indices = metadata.get("sampled_frame_indices", [])
                    if sampled_indices:
                        st.write(f"- Sampled Frame Indices: {sampled_indices}")

                render_analysis_panel(model, uploaded_video, result, "Video", "video")
            except Exception as error:
                st.error(f"❌ Could not process the uploaded video: {error}")


def main():
    """Main application entry point."""
    apply_custom_css()
    render_header()

    show_debug = render_sidebar()

    with st.spinner("🔄 Loading AI model..."):
        model = load_model()

    if getattr(model, "is_fallback", False):
        st.warning(
            "The remote Hugging Face model could not be loaded in this session, so the app is running with a local fallback predictor. "
            "The interface remains functional, but the predictions are only a demo-grade approximation until the model downloads successfully."
        )
        if getattr(model, "fallback_reason", "") and show_debug:
            st.caption(f"Fallback reason: {model.fallback_reason}")

    st.markdown("### 📥 Input Modes")
    image_tab, webcam_tab, video_tab = st.tabs(["Image Upload", "Webcam", "Video Upload"])

    with image_tab:
        _render_image_tab(model, show_debug)

    with webcam_tab:
        _render_webcam_tab(model, show_debug)

    with video_tab:
        _render_video_tab(model, show_debug)

    render_footer()


if __name__ == "__main__":
    main()
