import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# Load Model
model = load_model("agrivision_model.keras")

# Class names
class_names = [
"Apple",
"Banana",
"Orange",
"Mango",
"Grapes"
]

st.title("🌱 AgriVision AI")
st.write("Upload a fruit image to classify")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((224,224))
    image = np.array(image)/255.0
    image = np.expand_dims(image,axis=0)

    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")