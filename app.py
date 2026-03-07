import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 AgriVision AI")
st.subheader("AI Powered Fruit Recognition")

st.write("Upload a fruit image and the AI will identify it.")

# ----------------------------------
# Load Model
# ----------------------------------
@st.cache_resource
def load_ai_model():
    model = load_model("agrivision_model.keras")
    return model

model = load_ai_model()

# Get input size automatically from model
input_shape = model.input_shape
img_height = input_shape[1]
img_width = input_shape[2]

# ----------------------------------
# Fruit Class Labels
# ----------------------------------
# Update these names according to your dataset
class_names = [
    "Apple",
    "Banana",
    "Grapes",
    "Mango",
    "Orange"
]

# ----------------------------------
# File Upload
# ----------------------------------
uploaded_file = st.file_uploader(
    "Upload Fruit Image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------------
# Prediction
# ----------------------------------
if uploaded_file is not None:

    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize image according to model
    img = image.resize((img_width, img_height))

    # Convert to array
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("🔍 AI analyzing fruit..."):
        prediction = model.predict(img_array)

    pred_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    # Safe class detection
    if pred_index < len(class_names):
        predicted_class = class_names[pred_index]
    else:
        predicted_class = f"Class {pred_index}"

    # ----------------------------------
    # Result Display
    # ----------------------------------
    st.success("Prediction Complete")

    st.markdown(f"### 🍎 Fruit : **{predicted_class}**")
    st.markdown(f"### 📊 Confidence : **{confidence:.2f}%**")

    st.progress(int(confidence))

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption("AgriVision AI • Final Year AI Project")