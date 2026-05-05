import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Potato Leaf Disease Detection", page_icon="🥔", layout="centered")

# ---------------------------------------------------------
# Custom CSS for Banners matching the UI
# ---------------------------------------------------------
st.markdown("""
    <style>
    .main-banner {
        background: linear-gradient(to right, #388E3C, #81C784);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-banner {
        background-color: #2E7D32;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-top: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Header Banner
# ---------------------------------------------------------
st.markdown("""
    <div class="main-banner">
        <h1 style="margin:0; font-size: 2.5em;">🥔 🌿 Potato Leaf Disease Detection 🔗</h1>
        <p style="margin:5px 0 0 0; font-size: 1.1em;">Upload a potato leaf image to detect plant health</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model Loading & Processing
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('potato_disease_model.h5')

model = load_model()

# Class names exactly as they appear in the Kaggle dataset directories
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def process_and_predict(image):
    # Resize image to match the input shape of the CNN
    img = image.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, 0) # Create a batch dimension
    
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    
    return predicted_class, confidence

# ---------------------------------------------------------
# File Uploader UI
# ---------------------------------------------------------
uploaded_file = st.file_uploader("📥 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Create two columns layout matching the screenshot
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.image(image, use_column_width=True, caption="Uploaded Image")
        
    # Get predictions
    predicted_class, confidence = process_and_predict(image)
    
    # Clean up the class name for display (e.g., 'Potato___Early_blight' -> 'Early Blight')
    display_name = predicted_class.split('___')[-1].replace('_', ' ').title()
    
    with col2:
        st.markdown("### 📊 Confidence Level")
        
        # Display progress bar
        st.progress(int(confidence) if confidence <= 100 else 100)
        st.write(f"**Confidence: {confidence}%**")
        
        # Display confidence status box
        if confidence >= 80:
            st.success("🔥 High Confidence")
        elif confidence >= 50:
            st.warning("⚠️ Medium Confidence")
        else:
            st.error("❗ Low Confidence")

    # Bottom Prediction Banner
    if "Healthy" in display_name:
        st.markdown(f"""
            <div class="prediction-banner" style="background-color: #388E3C;">
                <h3 style="margin:0;">✅ Prediction: {display_name} 🌿</h3>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="prediction-banner">
                <h3 style="margin:0;">⚠️ Prediction: {display_name} 🍂 ⚠️ {display_name} 🍂</h3>
            </div>
        """, unsafe_allow_html=True)