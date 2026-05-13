"""Reusable UI components."""

import streamlit as st


def render_header():
    """Render the application header."""
    st.markdown('<h1 class="main-header">🛡️ DeepShield AI - Deepfake Detection System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Advanced AI-Generated Image Detection Using EfficientNetB7 with Attention Mechanism</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")


def render_sidebar():
    """Render sidebar with settings and return selected options."""
    with st.sidebar:
        # Sidebar Header
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.1) 100%);
                    padding: 2rem 1.5rem; margin: -1rem -1rem 2rem -1rem;
                    border-radius: 0 0 20px 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
            <div style='text-align: center;'>
                <div style='background: rgba(255,255,255,0.2); width: 60px; height: 60px;
                           border-radius: 50%; margin: 0 auto 1rem auto; display: flex;
                           align-items: center; justify-content: center;
                           box-shadow: 0 4px 10px rgba(0,0,0,0.2);'>
                    <span style='font-size: 2rem;'>⚙️</span>
                </div>
                <h2 style='color: #ffffff; font-size: 1.8rem; font-weight: 800; margin: 0;
                           text-shadow: 0 2px 4px rgba(0,0,0,0.3); letter-spacing: 0.5px;
                           -webkit-font-smoothing: antialiased;'>
                    SETTINGS
                </h2>
                <p style='color: rgba(255,255,255,0.9); font-size: 0.9rem; margin: 0.5rem 0 0 0;
                          font-weight: 500;'>
                    Configure Detection Parameters
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Preprocessing Method Section
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.1) 100%);
                    padding: 1.25rem; border-radius: 15px; margin-bottom: 1.5rem;
                    border: 2px solid rgba(255,255,255,0.2);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);'>
            <div style='display: flex; align-items: center; margin-bottom: 0.75rem;'>
                <div style='background: rgba(255,255,255,0.2); width: 35px; height: 35px;
                           border-radius: 8px; display: flex; align-items: center; justify-content: center;
                           margin-right: 0.75rem;'>
                    <span style='font-size: 1.2rem;'>📊</span>
                </div>
                <p style='color: #ffffff; font-weight: 700; margin: 0; font-size: 1.1rem;
                          text-shadow: 0 1px 2px rgba(0,0,0,0.2);'>
                    Preprocessing Method
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("""
        **ℹ️ Overview of Preprocessing Methods**

        Input images undergo preprocessing transformations before model inference. The selected method significantly impacts detection accuracy.
        """)

        preprocessing_method = st.radio(
            "Select preprocessing method",
            options=[
                "✅ Training Match (Recommended)",
                "Simple [0,1] Normalization",
                "EfficientNet ImageNet"
            ],
            index=0,
            label_visibility="collapsed"
        )

        # Detailed explanation based on selection
        if "Training Match" in preprocessing_method:
            st.success("""
            **✅ Training Match (Recommended)**

            **Method Description:**
            - Employs identical preprocessing pipeline used during model training
            - BGR color format (OpenCV standard)
            - No normalization applied (raw pixel values: 0-255)

            **Rationale:**
            - Ensures consistency with training methodology
            - Provides optimal detection accuracy
            - Maintains full compatibility with trained model weights

            **✨ Recommended for conference demonstrations**
            """)
        elif "Simple" in preprocessing_method:
            st.warning("""
            **⚠️ Simple [0,1] Normalization**

            **Method Description:**
            - Processes images in RGB color format
            - Applies pixel-wise division by 255 (normalized range: 0-1)

            **Application Scenarios:**
            - Comparative analysis and ablation studies
            - Investigating normalization impact on detection performance

            **⚠️ Note:** Detection accuracy may be reduced, as this method differs from the training preprocessing pipeline.
            """)
        else:
            st.warning("""
            **📚 EfficientNet ImageNet Preprocessing**

            **Method Description:**
            - Applies ImageNet standardization protocol
            - Performs mean subtraction and variance scaling
            - Utilizes EfficientNet default preprocessing pipeline

            **Application Scenarios:**
            - Transfer learning applications with EfficientNet architecture
            - Compatibility with ImageNet pre-trained weights

            **⚠️ Note:** This model was trained using custom preprocessing. Performance may be suboptimal with this configuration.
            """)

        st.markdown("---")

        # Map selection to method string
        if "Training Match" in preprocessing_method:
            preprocess_method = "training_match"
        elif "Simple" in preprocessing_method:
            preprocess_method = "simple_norm"
        else:
            preprocess_method = "efficientnet"

        # Debug Mode Section
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.1) 100%);
                    padding: 1.25rem; border-radius: 15px; margin-bottom: 1.5rem; margin-top: 1rem;
                    border: 2px solid rgba(255,255,255,0.2);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);'>
            <div style='display: flex; align-items: center; margin-bottom: 0.75rem;'>
                <div style='background: rgba(255,255,255,0.2); width: 35px; height: 35px;
                           border-radius: 8px; display: flex; align-items: center; justify-content: center;
                           margin-right: 0.75rem;'>
                    <span style='font-size: 1.2rem;'>🔧</span>
                </div>
                <p style='color: #ffffff; font-weight: 700; margin: 0; font-size: 1.1rem;
                          text-shadow: 0 1px 2px rgba(0,0,0,0.2);'>
                    Debug Mode
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("""
        **ℹ️ Debug Mode Overview**

        Displays detailed technical diagnostics for advanced analysis and system validation.
        """)

        show_debug = st.checkbox("Enable debug information", value=False)

        if show_debug:
            st.success("""
            **🔍 Debug Mode Enabled**

            **Diagnostic Information Displayed:**
            - 📷 Input image metadata (format, dimensions, color mode)
            - 🔄 Preprocessing pipeline details (tensor shape, data type, value range)
            - 📊 Raw model output (probability distributions)
            - 📈 Statistical metrics (mean, standard deviation)

            **Target Users:**
            - 👨‍💻 System developers
            - 🔬 Research personnel
            - 🎓 Technical demonstration audiences

            **Note:** Additional diagnostic information will appear in the analysis output section.
            """)
        else:
            st.info("""
            **💡 Standard Mode Active**

            Displays detection results only. Technical diagnostics are hidden.

            **To enable diagnostics:** Activate the checkbox above.
            """)

        # Pro Tip Section
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(255,215,0,0.15) 0%, rgba(255,193,7,0.1) 100%);
                    padding: 1.25rem; border-radius: 15px; margin-top: 2rem;
                    border: 2px solid rgba(255,215,0,0.3);
                    box-shadow: 0 4px 12px rgba(255,215,0,0.2);'>
            <div style='display: flex; align-items: flex-start;'>
                <div style='background: rgba(255,215,0,0.3); width: 40px; height: 40px;
                           border-radius: 50%; display: flex; align-items: center; justify-content: center;
                           margin-right: 1rem; flex-shrink: 0;'>
                    <span style='font-size: 1.4rem;'>💡</span>
                </div>
                <div>
                    <h4 style='color: #ffd700; font-weight: 700; margin: 0 0 0.5rem 0; font-size: 1rem;
                               text-shadow: 0 1px 2px rgba(0,0,0,0.3);'>
                        Recommended Configuration
                    </h4>
                    <p style='color: rgba(255,255,255,0.95); font-size: 0.85rem; margin: 0; line-height: 1.5;'>
                        For conference demonstrations, the <strong style='color: #ffffff;'>Training Match</strong> preprocessing method is recommended to ensure optimal detection accuracy.
                        Standard mode (debug disabled) provides a streamlined presentation interface.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    return preprocess_method, show_debug


def render_footer():
    """Render the application footer."""
    st.markdown("---")
    st.markdown("### 📋 About This System")

    # Feature Cards
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div style='background: #ffffff;
                    padding: 2rem; border-radius: 15px;
                    border: 2px solid #3182ce;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
            <h4 style='color: #2c5282; margin-bottom: 1.5rem; font-size: 1.3rem; font-weight: 700;'>
                🔬 Technical Architecture
            </h4>
            <ul style='list-style: none; padding: 0; margin: 0;'>
                <li style='padding: 0.75rem 0; border-bottom: 2px solid #e2e8f0; color: #2d3748; font-size: 1rem;'>
                    <strong style='color: #1a202c;'>EfficientNetB7</strong> convolutional backbone
                </li>
                <li style='padding: 0.75rem 0; border-bottom: 2px solid #e2e8f0; color: #2d3748; font-size: 1rem;'>
                    <strong style='color: #1a202c;'>Custom attention mechanism</strong> enhancing feature discrimination
                </li>
                <li style='padding: 0.75rem 0; border-bottom: 2px solid #e2e8f0; color: #2d3748; font-size: 1rem;'>
                    <strong style='color: #1a202c;'>Large-scale training dataset</strong> comprising authentic and synthetic images
                </li>
                <li style='padding: 0.75rem 0; color: #2d3748; font-size: 1rem;'>
                    <strong style='color: #1a202c;'>Robust performance</strong> across diverse deepfake generation methods
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style='background: #ffffff;
                    padding: 2rem; border-radius: 15px;
                    border: 2px solid #805ad5;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
            <h4 style='color: #553c9a; margin-bottom: 1.5rem; font-size: 1.3rem; font-weight: 700;'>
                ✨ Detection Capabilities
            </h4>
            <ul style='list-style: none; padding: 0; margin: 0;'>
                <li style='padding: 0.75rem 0; border-bottom: 2px solid #e2e8f0; color: #2d3748; font-size: 1rem;'>
                    Synthetic images generated by <strong style='color: #1a202c;'>GANs</strong> (Generative Adversarial Networks)
                </li>
                <li style='padding: 0.75rem 0; border-bottom: 2px solid #e2e8f0; color: #2d3748; font-size: 1rem;'>
                    <strong style='color: #1a202c;'>Diffusion-based models</strong> (Stable Diffusion, DALL-E, Midjourney)
                </li>
                <li style='padding: 0.75rem 0; border-bottom: 2px solid #e2e8f0; color: #2d3748; font-size: 1rem;'>
                    Alternative <strong style='color: #1a202c;'>generative AI architectures</strong>
                </li>
                <li style='padding: 0.75rem 0; color: #2d3748; font-size: 1rem;'>
                    <strong style='color: #1a202c;'>Resilient</strong> to post-processing and compression artifacts
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    return
