import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('brain_tumor.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((150,150))  
    img_array = np.array(image)
    img_array=img_array.reshape(1,150,150,3)  
    return img_array

# Function to predict the image
def predict(image):
    processed_image = preprocess_image(image)
    x = model.predict(processed_image)
    ind=x.argmax()
    if(ind==0):
        return ("This might be glioma tumor")
    elif(ind==1):
        return("This might be meningioma tumor")
    elif(ind==2):
        return("There isn't any tumor")
    elif(ind==3):
        return("This might be pituitary tumor")
    


st.title('Brain tumor detector')
pages = ['Home', 'Predictor']
selected_page = st.sidebar.radio('Select a page', pages)

if selected_page == 'Home':
    st.header('Welcome to the Brain tumor detector!')
    st.write('This app allows you to detect the type of tumor from report image.')
elif selected_page == 'Predictor':
    st.header('Image Predictor')
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('')

        if st.button('Predict'):
            label = predict(image)
            st.write(f'{label}')


