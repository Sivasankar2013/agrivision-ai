import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import json

# --------------------------------
# Page Settings
# --------------------------------
st.set_page_config(page_title="AgriVision AI", page_icon="🍎")

st.title("🍎 AgriVision AI")
st.write("Upload a fruit image and AI will identify the fruit")

# --------------------------------
# Load Model
# --------------------------------
@st.cache_resource
def load_ai_model():
    model = load_model("agrivision_model.keras")
    return model

model = load_ai_model()

# --------------------------------
# Load Class Names
# --------------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse dictionary (index → class name)
class_names = {v: k for k, v in class_indices.items()}

# --------------------------------
# File Upload
# --------------------------------
uploaded_file = st.file_uploader(
    "Upload Fruit Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------
# Prediction
# --------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize image (same as training)
    img = image.resize((100, 100))

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
st.caption("AgriVision AI • AI Fruit Recognition")