import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import json

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 AgriVision AI")
st.write("Upload a fruit image and AI will identify the fruit.")

# --------------------------------
# Load AI Model
# --------------------------------
@st.cache_resource
def load_ai_model():
    model = load_model("agrivision_model.keras")
    return model

model = load_ai_model()

# --------------------------------
# Load Class Names
# --------------------------------
@st.cache_resource
def load_class_names():
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}

class_names = load_class_names()

# --------------------------------
# File Upload
# --------------------------------
uploaded_file = st.file_uploader(
    "Upload Fruit Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------
# Prediction Section
# --------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=400)

    # Resize image to match training size
    img = image.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("AI analyzing fruit..."):
        prediction = model.predict(img_array)

    pred_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    predicted_class = class_names.get(pred_index, "Unknown Fruit")

    st.success("Prediction Complete")

    st.markdown(f"### 🍎 Fruit : **{predicted_class}**")
    st.markdown(f"### 📊 Confidence : **{confidence:.2f}%**")

    st.progress(int(confidence))

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.caption("AgriVision AI • AI Fruit Recognition System")