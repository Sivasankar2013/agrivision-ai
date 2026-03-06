import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("agrivision_model.keras")

# Class labels
class_names = sorted(os.listdir("dataset/Training"))

# Fruit information database
fruit_info = {
    "Apple": {
        "Ripeness": "Ripe",
        "Hybrid": "Fuji",
        "Storage": "7-10 days",
        "Nutrients": "Vitamin C"
    },
    "Banana": {
        "Ripeness": "Ripe",
        "Hybrid": "Cavendish",
        "Storage": "3-5 days",
        "Nutrients": "Potassium"
    },
    "Orange": {
        "Ripeness": "Ripe",
        "Hybrid": "Navel",
        "Storage": "10-14 days",
        "Nutrients": "Vitamin C"
    }
}

st.title("🍎 AgriVision AI - Fruit Analyzer")

st.write("Upload a fruit image to analyze")

uploaded_file = st.file_uploader("Choose a fruit image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((100,100))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    predicted_label = class_names[predicted_index]

    st.subheader("🔍 Prediction Result")

    st.write("**Fruit:**", predicted_label)
    st.write("**Confidence:** {:.2f}%".format(confidence))

    # Show fruit info
    if predicted_label in fruit_info:

        info = fruit_info[predicted_label]

        st.subheader("📊 Fruit Details")

        st.write("**Ripeness:**", info["Ripeness"])
        st.write("**Hybrid:**", info["Hybrid"])
        st.write("**Storage:**", info["Storage"])
        st.write("**Nutrients:**", info["Nutrients"])
