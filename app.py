import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('histopathology_classifier_model.h5')  

# Defining class names
class_names = {
    0: "TUMOR",
    1: "STROMA",
    2: "COMPLEX",
    3: "LYMPHO",
    4: "DEBRIS",
    5: "MUCOSA",
    6: "ADIPOSE",
    7: "EMPTY"
}

# Setting Streamlit app title
st.title("🧬 Histopathology Image Classifier")
st.write("Upload a histopathology image (64x64 RGB) and get the tissue type prediction.")

# Uploading image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Reading the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_resized = cv2.resize(image, (64, 64))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing the image
    image_array = image_resized / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 64, 64, 3)

    # Predicting
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]

    # Showing result
    st.markdown(f"### 🧾 Prediction: `{class_names[predicted_class]}`")
    st.markdown(f"#### 🔍 Confidence: `{confidence:.2f}`")

    # Showing full class probabilities
    st.subheader("📊 Class Probabilities")
    for idx, prob in enumerate(predictions[0]):
        st.write(f"{class_names[idx]}: {prob:.4f}")
