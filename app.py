import io
import joblib
from PIL import Image
import streamlit as st
from face_recognition import preprocessing

# Load the face recognizer model
face_recogniser = joblib.load('model/face_recogniser.pkl')
preprocess = preprocessing.ExifOrientationNormalize()

# Streamlit interface
st.title("Face Recognition Application")

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Checkbox to include all predictions
include_predictions = st.checkbox("Include all predictions", value=False)

# If an image is uploaded
if uploaded_file is not None:
    # Read the image
    img = Image.open(io.BytesIO(uploaded_file.read()))
    
    # Preprocess the image
    img = preprocess(img)
    
    # Convert image to RGB (stripping alpha channel if exists)
    img = img.convert('RGB')
    
    # Perform face recognition
    faces = face_recogniser(img)
    
    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Display results
    st.subheader("Recognition Results")
    if faces:
        for idx, face in enumerate(faces):
            st.markdown(f"### Face {idx + 1}")
            st.markdown(f"**Top Prediction:** {face.top_prediction.label} (Confidence: {face.top_prediction.confidence:.2f})")
            st.markdown(f"**Bounding Box:** Left: {face.bb.left}, Top: {face.bb.top}, Right: {face.bb.right}, Bottom: {face.bb.bottom}")
            
            if include_predictions:
                st.markdown("**All Predictions:**")
                for pred in face.all_predictions:
                    st.markdown(f"- {pred.label}: {pred.confidence:.2f}")
    else:
        st.warning("No faces detected.")
