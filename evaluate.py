import tensorflow as tf
import numpy as np
import os
from train import create_data_pipeline, DATA_DIR, BATCH_SIZE, IMAGE_SIZE

def evaluate_model(model_path='models/potato_disease_model.keras'):
    print(f"--- Loading Saved Model from '{model_path}' ---")
    model = tf.keras.models.load_model(model_path)
    
    print("\n--- Loading Validation Data for Evaluation ---")
    _, val_ds, class_names = create_data_pipeline(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)
    
    print("\n--- Evaluating Model Performance ---")
    loss, accuracy = model.evaluate(val_ds)
    print(f"\nOverall Validation Loss: {loss:.4f}")
    print(f"Overall Validation Accuracy: {accuracy * 100:.2f}%")
    
    print("\n--- Generating Confusion Matrix ---")
    y_true = []
    y_pred = []
    
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    cm = tf.math.confusion_matrix(y_true, y_pred).numpy()
    
    print("\nConfusion Matrix:")
    print(f"Order of classes: {class_names}")
    print(cm)
    
    print("\nPer-Class Accuracy breakdown:")
    for idx, name in enumerate(class_names):
        total_class = np.sum(cm[idx])
        correct_class = cm[idx][idx]
        acc = (correct_class / total_class) * 100 if total_class > 0 else 0
        print(f" - {name}: {correct_class}/{total_class} correct ({acc:.2f}%)")
        
    print("\n[SUCCESS] MODEL EVALUATION COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    evaluate_model()
