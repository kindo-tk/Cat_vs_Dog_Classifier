import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess


IMG_SIZE = 224
MODEL_PATH = "models/best_overall.keras"

# Loading the model
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    return None

# preprocessing
def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    return eff_preprocess(arr)

def predict(model, img):
    prob = float(model.predict(preprocess_image(img), verbose=0)[0][0])
    label = "Dog" if prob > 0.5 else "Cat"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence * 100

# -------------------------- UI ---------------------------
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")


hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .uploadedFile {text-align: center;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center; font-size:38px;'>Cat vs Dog Classifier</h1><br>",
    unsafe_allow_html=True
)

with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.error("Model not found! Please place best_overall.keras in this folder.")
    st.stop()

uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)

    col1, col2 = st.columns([1, 1])

    # uploaded image
    with col1:
        st.markdown("<div style='text-align:center; font-size:14px; color:gray;'>Uploaded Image</div>", 
                    unsafe_allow_html=True)
        st.image(img, width=280)

        predict_btn = st.button("Predict", use_container_width=True)

    # Prediction output
    with col2:
        if predict_btn:
            label, confidence = predict(model, img)
            color = "#0B3D91" if label == "Dog" else "#8B0000"

            st.markdown(
                f"""
                <div style="text-align:center; margin-top:35%;">
                    <span style="font-size:32px; font-weight:bold; color:{color};">
                        {label}
                    </span><br>
                    <span style="font-size:16px; color:gray;">
                        Confidence: {confidence:.2f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

# ---------------------- Footer ---------------------------
st.markdown(
    """
    <div style='text-align:center; margin-top:60px; font-size:14px; color:gray;'>
        Made with ❤️ by 
        <a href="https://www.linkedin.com/in/tufan-kundu-577945221/" target="_blank" style="text-decoration:none; font-weight:bold; color:#0B3D91;">
            TK
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
