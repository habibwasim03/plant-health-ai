# Plant Disease Detection

This project is a deep learning system that classifies plant leaf images into disease categories using **transfer learning (MobileNetV2)**.

## Project Structure
- `data/`: Contains the PlantVillage dataset organized by class (excluded from git).
- `setup_data.py`: Script to extract specific classes (e.g., Potato) from the massive PlantVillage zip file.
- `train.py`: Initializes the Data Loading Pipeline using TensorFlow, handling batching and resizing automatically.

## Current Progress
- **Step 1 (Data Setup)**: Downloaded and extracted the PlantVillage dataset (Potato subset).
- **Step 2 (Data Pipeline)**: Created `train.py` to efficiently load the images, split into training/validation sets, and resize to 224x224.

: Data Augmentation & Model Building completed 

