import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

st.title("Object recognizer using CIFAR-10 Dataset")
st.write("Made by Yashas Jain")
st.write('This model can detect Automobile, Cat, Plane, Bird, Dog, Deer, Frog, Horse, Ship, Truck.')
liss = ['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to 32x32x3 array
    resized_image = image.resize((32, 32))
    image_array = np.array(resized_image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    if image_array.shape == (1, 32, 32, 3):  # Check the shape with batch dimension
        
        st.write("The image you uploaded is of a.......")
        
        # Load the deep learning model
        loaded_model = load_model('model/face_recogniser.pkl')

        # Normalize pixel values (assuming your model expects normalized input)
        image_array = image_array / 255.0

        # Predict with the loaded model
        result = loaded_model.predict(image_array)[0]
        maxx  = 0
        indexx = 0
        
        for i in range(len(result)):
            if maxx < result[i]:
                maxx = result[i]
                indexx = i

        st.header(liss[indexx])
    else:
        st.write(f"The image array shape is: {image_array.shape}. It may not be a color image. Please try some other image.")

#st.bottom("For internships related to machine learning and data science kindly email me at yashasjain247@gmail.com or contact me via Whatsapp")


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Contact me through email - yashasjain247@gmail.com</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
