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
    from models.model import build_model
    
    train_ds, val_ds, classes = create_data_pipeline(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)
    print("\n[SUCCESS] DATA PIPELINE INITIALIZED SUCCESSFULLY.")
    
    print("\n--- Building MobileNetV2 Transfer Learning Model ---")
    model = build_model(input_shape=(224, 224, 3), num_classes=len(classes))
    model.summary()
    print("\n[SUCCESS] MODEL ARCHITECTURE BUILT SUCCESSFULLY.")
    
    EPOCHS = 10
    
    print("\n--- Compiling Model ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n--- Starting Initial Training (Step 6: Top Layers for {EPOCHS} Epochs) ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    print("\n[SUCCESS] INITIAL TRAINING OF TOP LAYERS COMPLETED SUCCESSFULLY.")
    
    # Step 7: Fine-Tuning
    print("\n--- Step 7: Unfreezing Deep Base Layers for Fine-Tuning ---")
    base_model = model.get_layer("mobilenetv2_1.00_224")
    base_model.trainable = True

    # Freeze the early layers (0 to 100) and keep top layers (100+) trainable
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print("\n--- Re-Compiling Model with Low Learning Rate (1e-5) ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    FINE_TUNE_EPOCHS = 10
    total_epochs = EPOCHS + FINE_TUNE_EPOCHS

    print(f"\n--- Starting Fine-Tuning (Epochs {EPOCHS + 1} to {total_epochs}) ---")
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1] + 1
    )
    print("\n[SUCCESS] FINE-TUNING COMPLETED SUCCESSFULLY.")
    
    os.makedirs('models', exist_ok=True)
    model_save_path = os.path.join('models', 'potato_disease_model.keras')
    model.save(model_save_path)
    print(f"\n[SUCCESS] FINAL FINE-TUNED MODEL SAVED TO '{model_save_path}'.")
