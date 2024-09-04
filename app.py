import io
import joblib
import pandas as pd
from PIL import Image
import streamlit as st
from face_recognition import preprocessing

# Load the face recognizer model
face_recogniser = joblib.load('model/face_recogniser.pkl')
preprocess = preprocessing.ExifOrientationNormalize()

# Initialize session state to store multiple columns of data
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

# Streamlit interface
st.title("Face Recognition Application")

# Date and name of the lecture
lecture_date = st.date_input("Select the date of the lecture")
lecture_name = st.text_input("Enter the name of the lecture")

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
    
    # Prepare data for the new column
    recognized_names = []

    # Display results
    st.subheader("Recognition Results")
    if faces:
        for idx, face in enumerate(faces):
            st.markdown(f"### Face {idx + 1}")
            st.markdown(f"**Top Prediction:** {face.top_prediction.label} (Confidence: {face.top_prediction.confidence:.2f})")
            st.markdown(f"**Bounding Box:** Left: {face.bb.left}, Top: {face.bb.top}, Right: {face.bb.right}, Bottom: {face.bb.bottom}")
            
            # Add name to recognized names list
            recognized_names.append(face.top_prediction.label)
            
            if include_predictions:
                st.markdown("**All Predictions:**")
                for pred in face.all_predictions:
                    st.markdown(f"- {pred.label}: {pred.confidence:.2f}")
    else:
        st.warning("No faces detected.")

    # Add data to session state dataframe
    if recognized_names:
        column_name = f"{lecture_name} ({lecture_date})"
        st.session_state.df[column_name] = pd.Series(recognized_names)
        st.success(f"Added data for {lecture_name} on {lecture_date}")

# Button to add another column
if st.button("Add Another Lecture"):
    st.experimental_rerun()

# Button to download the CSV file
if not st.session_state.df.empty:
    csv = st.session_state.df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name='attendance.csv', mime='text/csv')
