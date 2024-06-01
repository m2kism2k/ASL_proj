from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Function to perform inference
def predict_image(model_path, image_file):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Preprocess the image
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image

    # Perform prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    return chr(64 + predicted_class)  # Assuming class 0 corresponds to 'A', 1 to 'B', and so on
