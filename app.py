import streamlit as st
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms
from face_recognition import preprocessing, FaceFeaturesExtractor

# Load the trained model
MODEL_PATH = 'model/face_recogniser.pkl'
model = joblib.load(MODEL_PATH)

# Define image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])
    return transform(image)

# Streamlit app
st.title("Face Recognition App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Extract features
    features_extractor = model.features_extractor
    _, embedding = features_extractor(preprocessed_image)

    if embedding is None:
        st.write("No face detected in the image.")
    else:
        # Predict the class
        prediction = model.classifier.predict([embedding.flatten()])
        class_name = model.idx_to_class[prediction[0]]

        st.write(f"Predicted class: {class_name}")
