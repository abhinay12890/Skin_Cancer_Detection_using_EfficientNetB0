import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import requests
from PIL import Image
from io import BytesIO

# Prevent model reload on every interaction
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("skin_cancer_detector.keras", compile=False)

model = load_my_model()

@st.cache_resource
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")

# ---------------- GRADCAM ---------------- #

def get_gradcam_heatmap(model, img_tensor):
    # Pulling layer directly from full graph
    base_model = model.get_layer("efficientnetb0")
    last_conv_layer = base_model.get_layer(find_last_conv_layer(base_model))

    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )

    img_tensor = tf.cast(img_tensor, tf.float32)
    inputs = tf.keras.applications.efficientnet.preprocess_input(img_tensor)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs, training=False)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

# ---------------- Overlay ---------------- #

def overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
    jet_heatmap = cv2.resize(jet_heatmap, (image.shape[1], image.shape[0]))

    # Superimpose the heatmap onto original image
    superimposed_img = image * (1 - alpha) + jet_heatmap * alpha
    return np.uint8(np.clip(superimposed_img, 0, 255))

# ---------------- UI ---------------- #

st.title("AI-Powered Skin Cancer Detection with Explainable Grad-CAM")

st.markdown(
    """
    Upload a dermoscopic image to receive an AI-assisted prediction.
    
    ⚠️ **Disclaimer:** This tool is for educational purposes only and is not a medical diagnosis.
    """
)

uploaded_file = st.file_uploader("Upload a dermoscopic image", type=["jpg", "png", "jpeg"])
image_url = st.text_input("OR paste an image URL")

image = None

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

elif image_url:
    try:
        response = requests.get(image_url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        image = np.array(img)
    except Exception:
        st.error("Could not load image from URL.")

# Only run analysis if an image exists
if image is not None:
    # -------- Preprocess -------- #
    resized = cv2.resize(image, (224, 224))
    img_tensor = np.expand_dims(resized, axis=0)
    img_tensor = tf.cast(img_tensor, tf.float32)

    with st.spinner("Analyzing lesion..."):
        # Preprocess for EfficientNet specifically
        processed_input = tf.keras.applications.efficientnet.preprocess_input(img_tensor)
        pred = model.predict(processed_input, verbose=0)[0][0]
        
        heatmap = get_gradcam_heatmap(model, img_tensor)
        overlay = overlay_heatmap(image, heatmap)

        # -------- Results -------- #
        st.divider()
        st.subheader("AI Analysis Result")

        label = "Cancer" if pred > 0.61 else "Normal"

        if label == "Cancer":
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

        confidence = float(pred) if pred > 0.61 else 1 - float(pred)
        st.progress(confidence)
        st.caption(f"Raw Model Output Score: **{pred:.3f}**")

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(overlay, caption="Grad-CAM (Heatmap)", use_container_width=True)
