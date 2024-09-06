import streamlit as st
import requests
import json
import os
from PIL import Image
import numpy as np

# Function to load and preprocess image files
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Main app code
def main():
    st.title("Image Upload and Model Training")

    st.write("Upload images to train the model in the background.")

    # Upload image section
    uploaded_files = st.file_uploader("Choose images", type=["jpg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        st.write("Uploaded Images:")
        image_array = []
        
        for uploaded_file in uploaded_files:
            img = load_image(uploaded_file)
            st.image(img, caption=f"Uploaded {uploaded_file.name}", use_column_width=True)
            
            # Preprocess images (convert to array, reshape, etc.)
            img = np.array(img.resize((64, 64)))  # Resizing for simplicity
            image_array.append(img.tolist())  # Convert to list for JSON serialization
        
        if st.button("Start Training"):
            # Send images to backend API for training
            response = requests.post('http://backend_server_address/train', json={"images": image_array})
            
            if response.status_code == 200:
                job_id = response.json().get('job_id')
                st.write(f"Training started! Job ID: {job_id}")
                st.write("Please wait for the model to be trained. Check the status below:")
            else:
                st.write("Failed to start training. Please try again.")
        
        # Check training status section
        job_id_input = st.text_input("Enter your Job ID to check status:")
        
        if st.button("Check Status"):
            if job_id_input:
                status_response = requests.get(f'http://backend_server_address/status/{job_id_input}')
                status = status_response.json().get('status')
                st.write(f"Current Status: {status}")
                
                if status == 'COMPLETED':
                    model_link = status_response.json().get('download_link')
                    st.write(f"Download your model here: {model_link}")
            else:
                st.write("Please enter a valid Job ID.")

if _name_ == "_main_":
    main()
