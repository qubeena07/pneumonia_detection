import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your pneumonia detection model
model = tf.keras.models.load_model('chest_xray_1.h5')
# model = tf.keras.models.load_model('chest_Xray2.h5')


# Function to preprocess the image for prediction
def preprocess_image(image):
    # Convert the image to grayscale
    image = image.convert('L')  # 'L' mode for grayscale in PIL
    # Resize to 150x150 as expected by the model
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0  # Normalize pixel values between 0 and 1
    # Add channel dimension for grayscale (150, 150, 1)
    img_array = np.expand_dims(img_array, axis=-1)
    # Add batch dimension (1, 150, 150, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions using the model
def make_prediction(image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    return prediction

# Page Configuration
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🩺",
    layout="wide"
)
# Header Section
col1, col2 = st.columns([2, 1])  # Adjust column ratios as needed

with col1:
    st.markdown("""
    <style>
    
        .header {
            background-color: white;
            padding: 0 6px;
            text-align: left;
            color: #588CD3;
            font-size: 45px;
            border-radius: 5px;
            font-weight:bold;
        }
        .text-container {
            max-width: 50%;  /* Set a maximum width for the text */
            line-height: 1.6;  /* Adjust line spacing for better readability */
        }
    </style>
    
    <div class="header">Pneumonia Detection System</div>
    <p class="text-container">This app is designed to detect pneumonia by analyzing chest X-ray images, providing quick and accurate diagnostic results to assist in early detection and treatment.</p>

    """, unsafe_allow_html=True)

with col2:
    st.image("pneumonia_Image.png", width=100, use_container_width=True)  # Replace with your image path


# # Header Section
# st.markdown("""
# <style>
#     .header {
#         background-color: #0056b3;
#         padding: 12px;
#         text-align: center;
#         color: white;
#         font-size: 30px;
#         border-radius: 5px;
#     }
# </style>
# <div class="header">Pneumonia Detection System</div>
# """, unsafe_allow_html=True)
# st.write("")

# Sidebar for Navigation
# st.sidebar.title("Navigation")
# st.sidebar.markdown("""
# - **Project Overview**
# - **Upload X-ray Image**
# - **Prediction Results**
# """)
# st.sidebar.write("Designed for hospital and laboratory use.")

# Main Layout
# st.title("Pneumonia Detection System")
st.write("Upload a chest X-ray image to analyze and detect signs of pneumonia.")

# Project Overview
with st.expander("System Description"):
    st.markdown("""
    This application is designed for **hospital and laboratory settings** to assist medical professionals in diagnosing **pneumonia** from chest X-ray images. 
    It leverages advanced deep learning algorithms to provide accurate predictions.
    """)

# Upload Section
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

# Display uploaded image and results
if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(Image.open(uploaded_file), caption="Uploaded X-ray", use_container_width=True)

    with col2:
        st.subheader("Prediction Results")
        # Predict when button is clicked
        if st.button("Analyze X-ray"):
            image = Image.open(uploaded_file)
            prediction = make_prediction(image)

            # Display prediction score
            st.write(f"**Confidence Score:** {prediction[0][0]:.2f}")

            # Display interpretation
            if prediction[0][0] > 0.5:
                st.success("The model predicts: **Normal**")
            else:
                st.error("The model predicts: **Pneumonia**")
        else:
            st.info("Click 'Analyze X-ray' to begin.")

# Footer Section
st.markdown("""
<hr style="border:1px solid #e0e0e0;">
<div style="text-align:center;">
    <small>For diagnostic assistance only. Please consult a healthcare professional for final diagnosis.</small>
</div>
""", unsafe_allow_html=True)
