# final_ai_project

[Demo Video](https://youtu.be/ao0hQpP-03A)

# Convolutional Neural Network for MNIST Digit Classification

This repository contains a Convolutional Neural Network (CNN) implemented using Keras for classifying handwritten digits from the MNIST dataset.

## Dataset

The model is trained and tested on the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0 through 9).

## Preprocessing

The dataset is preprocessed as follows:
- The images are normalized to have pixel values in the range [0, 1].
- The labels are one-hot encoded using the `to_categorical` function from Keras.

## Model Architecture

The CNN model is constructed as follows:

1. **Convolutional Layers:**
   - The first convolutional layer has 32 filters with a 3x3 kernel and ReLU activation.
   - MaxPooling is applied with a 2x2 pool size.
   - The second convolutional layer has 64 filters with a 3x3 kernel and ReLU activation.
   - MaxPooling is applied again with a 2x2 pool size.
   - The third convolutional layer has 128 filters with a 3x3 kernel and ReLU activation.

2. **Flatten Layer:**
   - Flattens the output from the convolutional layers into a 1D array.

3. **Dense Layers:**
   - A dense layer with 128 units and ReLU activation.
   - The final dense layer with 10 units (equal to the number of classes) and softmax activation for multi-class classification.

## Model Compilation

The model is compiled using categorical crossentropy as the loss function, the Adam optimizer, and accuracy as the evaluation metric.

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## Saved Model

`mnist.h5` contains the saved model which is used in deployed using streamlit and used for predictions.
`webapp.py` contains code for deploying the model
`demo video` is hosted on youtube and the video link is here.
