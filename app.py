import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌱"
)

st.title("🌱 AgriVision AI")
st.write("Upload a fruit image to classify")

# Load model
@st.cache_resource
def load_my_model():
    model = load_model("agrivision_model.keras")
    return model

model = load_my_model()

# Get model input size automatically
input_shape = model.input_shape
img_height = input_shape[1]
img_width = input_shape[2]

# Class names (edit based on dataset)
class_names = [
    "Apple",
    "Banana",
    "Grapes",
    "Mango",
    "Orange"
]

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","png","jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Resize based on model
    img = image.resize((img_width, img_height))

    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing... 🤖"):
        prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success("Prediction Complete")

    st.write("### 🍎 Fruit:", predicted_class)
    st.write("### 📊 Confidence:", f"{confidence:.2f}%")