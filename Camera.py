import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
from io import BytesIO

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

# Select samples based on the sampled indices
X = mnist.data
y = mnist.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the KNN classifier
knn_classifier = KNeighborsClassifier()

# Train the KNN classifier
knn_classifier.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = knn_classifier.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

def predict_number(image, model):
    # Convert image to grayscale and resize to 28x28
    image = image.convert('L').resize((28, 28))
    # Convert image to numpy array
    img_array = np.array(image)
    # Flatten the image array
    img_flat = img_array.flatten()
    # Normalize pixel values
    img_scaled = img_flat / 255.0
    # Reshape the array to match the input shape of the model
    img_reshaped = img_scaled.reshape(1, -1)
    # Make prediction
    prediction = model.predict(img_reshaped)
    return prediction[0]

# Streamlit app
def main():
    st.title('Digit Recognition App')
    st.write('Upload an image of a digit (0-9) or capture using webcam for prediction.')

    # Option to upload image or capture using webcam
    option = st.selectbox("Choose input option:", ("Upload Image", "Use Webcam"))

    if option == "Upload Image":
        # Upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            # Convert the uploaded file to bytes
            file_bytes = uploaded_file.read()
            # Convert the bytes data to a PIL Image object
            image = Image.open(BytesIO(file_bytes))
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            # Make prediction
            prediction = predict_number(image, knn_classifier)
            # Display prediction result in big size with bold
            st.markdown(f"<p style='font-size:18px;'><b>Prediction: {prediction}</b></p>", unsafe_allow_html=True)

    elif option == "Use Webcam":
        st.title("Webcam Input Example")
        frame =  st.camera_input("Take a picture")

        if frame is not None:
            # Convert frame to bytes
            frame_bytes = frame.read()
            # Convert frame bytes to PIL image
            pil_image = Image.open(BytesIO(frame_bytes))
            # Make prediction
            prediction = predict_number(pil_image, knn_classifier)
            # Display prediction result in big size with bold
            st.markdown(f"<p style='font-size:18px;'><b>Prediction: {prediction}</b></p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
