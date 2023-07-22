import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the pre-trained Keras model
model = keras.models.load_model('keras_model.h5')

# Define the class labels
class_labels = ['ready_to_pickle_strawberry','not_pickle_strawberry','Bad_weevils_impacted']

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match model input shape
    image = np.array(image)  # Convert image to numpy array
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    class_index = np.argmax(prediction)
    class_label = class_labels[class_index]
    confidence = prediction[0, class_index]
    return class_label, confidence

# Create the Streamlit web application
def main():
    st.title("Strawberry Rapiness Classification")
    st.write("Upload an image")

    # Create file uploader for user to upload image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make predictions
        st.header("Prediction Result")
        class_label, confidence = predict(image)
        st.header(f"Prediction: {class_label}")
        st.header(f"Confidence: {confidence:.2f}")

# Run the application
if __name__ == '__main__':
    main()