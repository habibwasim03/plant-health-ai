import zipfile
import os
import shutil

zip_path = 'archive.zip'
target_dir = 'data'

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

print(f"Extracting Potato classes from {zip_path} to {target_dir}...")

extracted_count = 0
with zipfile.ZipFile(zip_path, 'r') as z:
    for file_info in z.infolist():
        if 'Potato' in file_info.filename and not file_info.is_dir() and not file_info.filename.startswith('plantvillage/'):
            parts = file_info.filename.split('/')
            
            class_name = next((part for part in parts if 'Potato' in part), None)
            
            if class_name:
                filename = parts[-1]
                class_dir = os.path.join(target_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                with z.open(file_info.filename) as source, open(os.path.join(class_dir, filename), "wb") as target:
                    shutil.copyfileobj(source, target)
                    extracted_count += 1

print(f"Extraction complete! Total images extracted: {extracted_count}")
for class_name in os.listdir(target_dir):
    class_path = os.path.join(target_dir, class_name)
    if os.path.isdir(class_path):
        print(f" - {class_name}: {len(os.listdir(class_path))} images")
