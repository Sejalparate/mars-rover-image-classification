from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Load model
model_path = r'C:\Users\sejbp\Coding\GITHUB\Mars-Rover-Image-Classification\Model_training_dataset_split\final_model.h5'
model = tf.keras.models.load_model(model_path, compile = False)

# Load label information from the text file
label_info_file = r'C:\Users\sejbp\Coding\GITHUB\Mars-Rover-Image-Classification\Mars Surface and Curiosity Image dataset\main_dataset\Labels Information.txt'

# Create a dictionary to map class indices to label names
class_indices = {}
with open(label_info_file, 'r') as file:
    for index, line in enumerate(file):
        label = line.strip()
        class_indices[index] = label

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((180, 180))  
    image_array = np.array(image)  
    image_array = np.expand_dims(image_array, axis=0)  
    image_array = image_array.astype('float32') / 255.0 
    return image_array

# Function to predict the class of an image
def predict_image_class(model, image_array, class_indices):
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name.split(' ', 1)[1].strip()

# Streamlit app
st.title("Mars Rover Image Classification")

uploaded_image = st.file_uploader("Upload an image obtained from a Mars rover to classify it.", type = ["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)  # Open image for display
    resized_image = image.resize((224, 224))
    st.image(resized_image, caption = 'Uploaded Image')

if st.button('Classify'):
    if uploaded_image is not None:
        image_array = preprocess_image(image)
        prediction = predict_image_class(model, image_array, class_indices)
        st.success(f'Prediction: {prediction}')
    else:
        st.error("Error: Please upload an image first.")