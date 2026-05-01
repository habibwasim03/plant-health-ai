import tensorflow as tf
import os

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
DATA_DIR = 'data'

def create_data_pipeline(data_dir, batch_size, image_size):
    print("--- Loading Training Data ---")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    print("\n--- Loading Validation Data ---")
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    class_names = train_dataset.class_names
    print(f"\nDiscovered Class Names: {class_names}")
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, class_names

if __name__ == "__main__":
    from utils.augmentation import get_data_augmentation_module
    
    train_ds, val_ds, classes = create_data_pipeline(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)
    print("\n[SUCCESS] DATA PIPELINE INITIALIZED SUCCESSFULLY.")
    
    print("\n--- Initializing Data Augmentation Module ---")
    data_augmentation_layer = get_data_augmentation_module()
    
    data_augmentation_layer.build(input_shape=(None, 224, 224, 3))
    data_augmentation_layer.summary()
    print("\n[SUCCESS] PREPROCESSING & AUGMENTATION MODULE READY.")
