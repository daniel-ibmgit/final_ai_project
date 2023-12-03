import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from keras.models import load_model


st.title('Handwritten Digit Recognition')

st.write("""
            This app predicts a digit based on the **MNIST** handwritten digit dataset.
            """)
st.write('---')

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (28,28)
    image = Image.open(image_data)
    image = image.resize(size)
    image = np.asarray(image)
    img = image[:,:,1]
    img = img.reshape(1,28,28,1)
    img = img/255.0
    model = load_model(model)
    prediction = model.predict(img)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(file, 'mnist.h5')
    class_names = ['0','1','2','3','4','5','6','7','8','9']
    string = "This image most likely is: " + class_names[np.argmax(prediction)]
    st.success(string)
    
st.write('---')
st.write('**Developed by:**')
st.write('Delphine Niyogushimirwa **and** Daniel Byiringiro')