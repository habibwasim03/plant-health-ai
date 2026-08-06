import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.saving.register_keras_serializable(package="Custom")
def preprocess_input(x):
    return tf.keras.applications.mobilenet_v2.preprocess_input(x)

def get_data_augmentation_module():
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.Lambda(preprocess_input),
    ], name="data_augmentation_module")
    
    return data_augmentation
