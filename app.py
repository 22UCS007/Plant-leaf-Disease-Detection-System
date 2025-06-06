import os
os.environ["PORT"] = "10000"
import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
import time
from PIL import Image

# ✅ Page configuration (must come BEFORE any Streamlit UI code)
st.set_page_config(page_title="Potato Plant Disease Detection", layout="wide")

# === Google Drive Model File ID ===
# GOOGLE_DRIVE_FILE_ID = "1wLcoL20u9BKG72bcSrUMCuI4fHu3CF3l"
MODEL_PATH = "trained_plant_disease_model.keras"

# ✅ Download model only if not exists
# def download_model():
#     if not os.path.exists(MODEL_PATH):
#         with st.spinner("🔽 Downloading AI model... Please wait."):
#             gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)
#         st.success("✅ Model downloaded successfully!")

# download_model()

# ✅ Cache the loaded model for performance
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ✅ Prediction function (uses cached model)
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# === Sidebar Navigation ===
st.sidebar.title("🌿 Potato Plant Disease System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("🔎 Select Page", ["Home", "Disease Recognition"])

# === CSS Styling ===
st.markdown("""
    <style>
        .main-container {display: flex; justify-content: center; align-items: center; flex-direction: column;}
        .image-container {display: flex; justify-content: center;align-items: center;width: 100%;max-height: 400px; overflow: hidden;}
        .stButton>button {border-radius: 10px; background-color: #4CAF50; color: white; font-size: 18px; padding: 10px; border: none; cursor: pointer;}
        .stButton>button:hover {background-color: #45a049; cursor: pointer;}
        @keyframes blink {50% {opacity: 0.5;}}
        .running-text {font-size: 20px; color: red; animation: blink 1.5s infinite; text-align: center;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# === Show Logo Image ===
try:
    img = Image.open("potato_AI.jpg")
    img = img.resize((800, 200))
    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
except Exception as e:
    st.warning(f"⚠️ Error loading image: {e}")

# === Pages ===
if app_mode == "Home":
    st.markdown("<h1 style='text-align: center; color: green;'>🌿Potato Plant Disease Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>A Sustainable Approach to Smart Agriculture</h3>", unsafe_allow_html=True)
    st.markdown("""
        <ul style='font-size:18px;'>
            <li>Detects Potato plant diseases using AI models.</li>
            <li>Helps farmers take early preventive measures.</li>
            <li>Improves crop health and agricultural yield.</li>
        </ul>
        <p class='running-text'><b>🔍 Detect diseases quickly & save your potato plant! 🚀</b></p>
    """, unsafe_allow_html=True)

elif app_mode == "Disease Recognition":
    st.header("🌱 Potato Plant Disease Detection System")
    
    test_image = st.file_uploader("📤 Upload a potato plant image:", type=["jpg", "jpeg", "png"])

    if st.button("Show Image"):
        if test_image:
            st.image(test_image, use_container_width=True)
        else:
            st.warning("⚠️ Please upload an image first.")

    if st.button("Predict Disease"):
        if test_image:
            with st.spinner("🧠 Analyzing image..."):
                time.sleep(2)  # Simulate delay
                result_index = model_prediction(test_image)
                class_name = ["Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy"]
                st.success(f"✅ Model Prediction: {class_name[result_index]}")
                st.snow()
        else:
            st.warning("⚠️ Please upload an image first.")

st.markdown("</div>", unsafe_allow_html=True)
