import tensorflow as tf
from tensorflow.keras import layers, models
from utils.augmentation import get_data_augmentation_module

def build_model(input_shape=(224, 224, 3), num_classes=3):
    data_augmentation = get_data_augmentation_module()
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="mobilenetv2_plant_disease")
    return model
