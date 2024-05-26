import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('brain_tumor.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((150,150))  # Adjust size as per your model requirement
    img_array = np.array(image)
    img_array=img_array.reshape(1,150,150,3)  # Normalize if required
    return img_array

# Function to predict the image
def predict(image):
    processed_image = preprocess_image(image)
    x = model.predict(processed_image)
    ind=x.argmax()
    if(ind==0):
        return ("glioma_tumor")
    elif(ind==1):
        return("meningioma_tumor")
    elif(ind==2):
        return("no_tumor")
    elif(ind==3):
        return("pituitary_tumor")
    


st.title("Image Classification App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.text("Predicting...")
    prediction = predict(image)
    st.write(f'{prediction}')
