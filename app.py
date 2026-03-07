import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# -------------------------------
# Page Settings
# -------------------------------
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 AgriVision AI")
st.subheader("AI Based Fruit Identification System")

st.write("Upload a fruit image and the AI will identify the fruit.")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_my_model():
    model = load_model("agrivision_model.keras")
    return model

model = load_my_model()

# -------------------------------
# Fruit Classes (edit if needed)
# -------------------------------
class_names = [
    "Apple",
    "Banana",
    "Grapes",
    "Mango",
    "Orange"
]

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Fruit Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize image (safe size)
    img = image.resize((150,150))

    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("AI analyzing fruit..."):
        prediction = model.predict(img_array)

    predicted_index = int(np.argmax(prediction))

    # Safe class selection
    if predicted_index < len(class_names):
        predicted_class = class_names[predicted_index]
    else:
        predicted_class = "Unknown Fruit"

    confidence = float(np.max(prediction) * 100)

    st.success("Prediction Completed")

    st.markdown(f"### 🍎 Fruit : **{predicted_class}**")
    st.markdown(f"### 📊 Confidence : **{confidence:.2f}%**")

    st.progress(int(confidence))

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("AgriVision AI | Final Year AI Project")