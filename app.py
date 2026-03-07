import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Page Config
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌱",
    layout="centered"
)

# Title
st.title("🌱 AgriVision AI")
st.subheader("AI Powered Fruit Classification")
st.write("Upload a fruit image and let AI identify it!")

# Load Model
@st.cache_resource
def load_my_model():
    model = load_model("agrivision_model.keras")
    return model

model = load_my_model()

# Class Names (edit based on your dataset)
class_names = [
    "Apple",
    "Banana",
    "Grapes",
    "Mango",
    "Orange"
]

# File Upload
uploaded_file = st.file_uploader(
    "📤 Upload Fruit Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.reshape(img_array, (1, 224, 224, 3))

    # Prediction
    with st.spinner("🤖 AI is analyzing the fruit..."):
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    # Result
    st.success("✅ Prediction Complete")

    st.markdown(f"### 🍎 Fruit Name: **{predicted_class}**")
    st.markdown(f"### 📊 Confidence: **{confidence:.2f}%**")

    st.progress(int(confidence))

# Footer
st.markdown("---")
st.caption("AgriVision AI | Deep Learning Based Fruit Classification")