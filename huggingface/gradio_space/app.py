"""
DeepShield AI - Deepfake Detection System
Gradio Interface for Hugging Face Spaces

Author: Emin Cem Koyluoglu
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import layers, regularizers
import tensorflow as tf

# Disable GPU (for CPU-only deployment)
tf.config.set_visible_devices([], 'GPU')

# Model configuration
MODEL_REPO = "CemRoot/deepfake-detection-model"
MODEL_FILE = "best_model_effatt.h5"
IMAGE_SIZE = 128


def attention_block(features, depth):
    """Attention mechanism for enhanced feature extraction"""
    attn = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(layers.Dropout(0.5)(features))
    attn = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(attn)
    attn = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(attn)
    attn = layers.Conv2D(1, (1, 1), padding='valid', activation='sigmoid')(attn)

    up = layers.Conv2D(depth, (1, 1), padding='same', activation='linear', use_bias=False)
    up_w = np.ones((1, 1, 1, depth), dtype=np.float32)
    up.build((None, None, None, 1))
    up.set_weights([up_w])
    up.trainable = True

    attn = up(attn)
    masked = layers.Multiply()([attn, features])

    gap_feat = layers.GlobalAveragePooling2D()(masked)
    gap_mask = layers.GlobalAveragePooling2D()(attn)
    gap = layers.Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_feat, gap_mask])
    return gap


def build_effatt_model(input_shape=(128, 128, 3)):
    """Build EfficientNetB7 with Attention mechanism"""
    base_model = EfficientNetB7(include_top=False, weights=None, input_shape=input_shape)
    base_model.trainable = False

    features = base_model.output
    bn_features = layers.BatchNormalization()(features)
    pt_depth = base_model.output_shape[-1]
    gap = attention_block(bn_features, pt_depth)

    x = layers.Dropout(0.5)(gap)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    return keras.Model(inputs=base_model.input, outputs=outputs)


def load_model():
    """Load the deepfake detection model from local file or Hugging Face Hub"""
    print("📥 Loading model...")

    try:
        # Try loading from local file first (if uploaded to Space)
        import os
        local_model_path = "best_model_effatt.h5"

        if os.path.exists(local_model_path):
            print(f"✅ Found local model file: {local_model_path}")
            model_path = local_model_path
        else:
            # Fallback: Download from Hugging Face Model Hub
            print("📥 Downloading model from Hugging Face Hub...")
            model_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                repo_type="model"
            )
            print(f"✅ Model downloaded: {model_path}")

        # Try loading full model first (with custom objects)
        try:
            def RescaleGAP(tensors):
                return tensors[0] / tensors[1]

            model = keras.models.load_model(
                model_path,
                custom_objects={
                    'RescaleGAP': RescaleGAP,
                    'attention_block': attention_block
                },
                compile=False
            )
            print("✅ Model loaded successfully (full model)!")
            return model
        except Exception as full_load_error:
            print(f"⚠️ Full model load failed: {full_load_error}")
            print("🔄 Attempting to rebuild architecture and load weights...")

            # Rebuild architecture and load weights
            try:
                model = build_effatt_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
                model.load_weights(model_path)
                print("✅ Model loaded successfully (weights only)!")
                return model
            except Exception as weights_error:
                print(f"❌ Weights loading also failed: {weights_error}")
                return None

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None


# Load model at startup
print("🚀 Starting DeepFake Detection System...")
model = load_model()


def preprocess_image(image):
    """
    Preprocess image for model inference using the locked EfficientNet pipeline.

    Args:
        image: PIL Image or numpy array

    Returns:
        Preprocessed numpy array ready for model input
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Resize to model input size
    img_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    # EfficientNet preprocessing is the only supported path.
    from tensorflow.keras.applications.efficientnet import preprocess_input
    arr = preprocess_input(img_resized.astype(np.float32))

    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)

    return arr


def predict(image):
    """
    Predict if image is real or fake

    Args:
        image: Input image (PIL Image or numpy array)

    Returns:
        Dictionary with class probabilities for Gradio Label component
    """
    if model is None:
        return {"Error": 1.0}

    try:
        # Preprocess image
        processed_image = preprocess_image(image)

        # Make prediction
        predictions = model.predict(processed_image, verbose=0)

        fake_prob = float(predictions[0][0])
        real_prob = float(predictions[0][1])

        # Return probabilities for Gradio Label (will show as colored bars)
        result = {
            "🚨 FAKE (AI-Generated)": fake_prob,
            "✅ AUTHENTIC (Real)": real_prob
        }

        return result

    except Exception as e:
        return {"Error": 1.0}


# Custom CSS for colored progress bars
custom_css = """
.label-item:first-child .label-content {
    background: linear-gradient(90deg, #ff4444 0%, #cc0000 100%) !important;
}
.label-item:last-child .label-content {
    background: linear-gradient(90deg, #00cc00 0%, #009900 100%) !important;
}
.label-item .label-content {
    color: white !important;
    font-weight: bold !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}
"""

# Create Gradio interface
with gr.Blocks(
    title="DeepShield AI - Deepfake Detection System",
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue",
    ),
    css=custom_css
) as demo:

    # Header
    gr.Markdown("""
    # 🛡️ DeepShield AI - Deepfake Detection System

    **Advanced AI-Generated Image Detection Using EfficientNetB7 with Attention Mechanism**

    Upload an image to detect if it's authentic or AI-generated.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # Input image
            input_image = gr.Image(
                label="Upload Image",
                type="pil",
                height=400
            )

            # Analyze button
            analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")

        with gr.Column(scale=1):
            # Output results with colored progress bars
            output_result = gr.Label(
                label="Detection Results",
                num_top_classes=2,
                show_label=True
            )

    # Information section
    gr.Markdown("""
    ---

    ## 🔬 Technology Stack

    - **Architecture:** EfficientNetB7 + Custom Attention Mechanism
    - **Framework:** TensorFlow 2.15
    - **Model Size:** ~780MB
    - **Input Size:** 128×128 pixels

    ---

    ## 📈 Detection Capabilities

    ✅ **GANs:** StyleGAN, ProGAN, BigGAN, CycleGAN, StarGAN

    ✅ **Diffusion Models:** Stable Diffusion, DALL-E, Midjourney, Imagen

    ✅ **Other:** Neural style transfer, VAE-based generators, image-to-image translation

    ---

    ## 👨‍💻 Author

    **Emin Cem Koyluoglu**

    DeepShield AI - Deepfake Detection System

    ---

    ## 🔗 Links

    - **Model Repository:** [CemRoot/deepfake-detection-model](https://huggingface.co/CemRoot/deepfake-detection-model)
    - **GitHub:** [CemRoot/deepfake-detection-streamlit](https://github.com/CemRoot/deepfake-detection-streamlit)

    ---

    **License:** MIT | **Framework:** Gradio + TensorFlow
    """)

    # Connect button to prediction function
    analyze_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=output_result
    )

    # Example images (optional - can add examples here)
    # gr.Examples(
    #     examples=[
    #         ["example1.jpg"],
    #         ["example2.jpg"],
    #     ],
    #     inputs=[input_image],
    # )

# Launch the app
if __name__ == "__main__":
    demo.launch()
