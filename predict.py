import tensorflow as tf
import numpy as np
import os
from PIL import Image

MODEL_PATH = 'models/potato_disease_model.keras'
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
IMAGE_SIZE = (224, 224)

_model = None

def get_model(model_path=MODEL_PATH):
    global _model
    if _model is None:
        custom_objects = {'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input}
        _model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return _model

def predict_image(image_input, model_path=MODEL_PATH):
    model = get_model(model_path)
    
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB').resize(IMAGE_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('RGB').resize(IMAGE_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
    else:
        img_array = image_input
        
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0])) * 100
    
    return predicted_class, confidence

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        pred_class, conf = predict_image(test_path)
        print(f"\nPrediction: {pred_class}")
        print(f"Confidence: {conf:.2f}%")
    else:
        print("Usage: python predict.py <path_to_image>")
