from ultralytics import YOLO
import os
import json
import shutil
from tqdm import tqdm # Used for a progress bar during conversion

# --- 1. DEFINE YOUR DATASET PATH ---
# This path looks correct based on your log.
# ===================================================================
RAW_DATA_PATH = r'C:\Users\Utkarsh\OneDrive\Desktop\new2\archive (2)\raw_data'
# ===================================================================

# Define paths based on the raw_data path
TRAIN_IMG_PATH = os.path.join(RAW_DATA_PATH, 'public_training_set_release_2.0', 'images')
TRAIN_ANNO_PATH = os.path.join(RAW_DATA_PATH, 'public_training_set_release_2.0', 'annotations.json')
TRAIN_LABEL_DIR = os.path.join(RAW_DATA_PATH, 'public_training_set_release_2.0', 'labels') # New labels folder

VAL_IMG_PATH = os.path.join(RAW_DATA_PATH, 'public_validation_set_2.0', 'images')
VAL_ANNO_PATH = os.path.join(RAW_DATA_PATH, 'public_validation_set_2.0', 'annotations.json')
VAL_LABEL_DIR = os.path.join(RAW_DATA_PATH, 'public_validation_set_2.0', 'labels') # New labels folder

TEST_IMG_PATH = os.path.join(RAW_DATA_PATH, 'public_test_release_2.0', 'images')

YAML_FILE_NAME = 'food_dataset.yaml'

# --- 2. NEW FUNCTION: Convert COCO to YOLO format ---
def convert_coco_to_yolo(json_path, output_label_dir):
    """
    Converts COCO annotations (from annotations.json) into YOLO format
    (.txt files) and saves them in the output_label_dir.
    """
    print(f"\nStarting COCO to YOLO conversion for: {json_path}")
    
    # Ensure output label directory exists
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Load COCO JSON
    try:
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return 0, {}

    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])
    
    if not images or not annotations or not categories:
        print(f"Error: JSON file {json_path} is missing 'images', 'annotations', or 'categories' key.")
        return 0, {}

    # Create mapping for image_id to filename and size
    image_map = {}
    for img in images:
        image_map[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
        
    # Create mapping for category_id to class_index (0-based)
    categories.sort(key=lambda x: x['id'])
    cat_map = {cat['id']: i for i, cat in enumerate(categories)}
    class_names = [cat['name'] for cat in categories]
        
    # Dictionary to hold annotations per image
    annotations_per_image = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in annotations_per_image:
            annotations_per_image[image_id] = []
        annotations_per_image[image_id].append(ann)
            
    # Process each image and its annotations
    converted_count = 0
    for image_id, ann_list in tqdm(annotations_per_image.items(), desc="Converting labels"):
        if image_id not in image_map:
            print(f"Warning: Skipping annotations for missing image_id {image_id}")
            continue
            
        img_data = image_map[image_id]
        img_w = img_data['width']
        img_h = img_data['height']
        
        # Get base filename
        base_filename = os.path.splitext(img_data['file_name'])[0]
        label_path = os.path.join(output_label_dir, f"{base_filename}.txt")
        
        with open(label_path, 'w') as label_file:
            for ann in ann_list:
                cat_id = ann['category_id']
                if cat_id not in cat_map:
                    print(f"Warning: Skipping unknown category_id {cat_id}")
                    continue
                class_index = cat_map[cat_id]
                
                # Get COCO bbox [x, y, w, h]
                x, y, w, h = ann['bbox']
                
                # Convert to YOLO format [x_center_norm, y_center_norm, w_norm, h_norm]
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # Write to file
                label_file.write(f"{class_index} {x_center} {y_center} {w_norm} {h_norm}\n")
        converted_count += 1

    print(f"Conversion complete. {converted_count} label files created in: {output_label_dir}")
    return len(class_names), class_names

# --- 3. Check Paths & Run Conversion ---
if not os.path.exists(TRAIN_ANNO_PATH):
    print("="*50)
    print(f"ERROR: Could not find annotations.json at: {TRAIN_ANNO_PATH}")
    print("Please check your `RAW_DATA_PATH` variable at the top of this script.")
    print("="*50)
    exit()

# Run conversion for training set
num_classes, class_names = convert_coco_to_yolo(TRAIN_ANNO_PATH, TRAIN_LABEL_DIR)
if num_classes == 0:
    print("Error: Training data conversion failed.")
    exit()

# Run conversion for validation set
val_num_classes, _ = convert_coco_to_yolo(VAL_ANNO_PATH, VAL_LABEL_DIR)
if val_num_classes == 0:
    print("Error: Validation data conversion failed.")
    exit()

if num_classes != val_num_classes:
    print(f"Warning: Train set has {num_classes} classes, Val set has {val_num_classes} classes.")

print(f"\nFound {num_classes} classes.")
    
# --- 4. Auto-Generate the .yaml File ---
print("Generating dataset YAML file...")
# This YAML now correctly points to the image folders.
# YOLO will automatically find the 'labels' folders nearby.
yaml_content = f"""
train: {os.path.abspath(TRAIN_IMG_PATH)}
val: {os.path.abspath(VAL_IMG_PATH)}
test: {os.path.abspath(TEST_IMG_PATH)}

# Classes
nc: {num_classes}
names: {class_names}
"""

# Write the YAML file
with open(YAML_FILE_NAME, 'w') as f:
    f.write(yaml_content)

print(f"Successfully created '{YAML_FILE_NAME}'")

# --- 5. Define Model and Training Parameters ---
MODEL_NAME = 'yolov8n.pt'  # Detection model
EPOCHS = 10                # Matching your log (was 75)
IMAGE_SIZE = 640
PROJECT_NAME = 'food_detection_model'

# --- 6. Load and Train the Model ---
model = YOLO(MODEL_NAME)

print(f"\nStarting DETECTION training using '{YAML_FILE_NAME}'")
print("INFO: Training on CPU. This will be very slow, but it will work.")
print("INFO: Your log showed 'optimizer=auto'. This is normal.")
print("      It chose a low learning rate (2e-05) because you are on CPU.")
print("      This is fine, just letting you know.")

results = model.train(
    data=YAML_FILE_NAME,    # Path to our newly generated YAML file
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=8,
    name=PROJECT_NAME,
    device='cpu'            # Explicitly setting to CPU to match your setup
)

print("\n--- Training Complete! ---")
print(f"Model and results saved to: {results.save_dir}")

# --- 7. Validate the Model ---
print("\n--- Validating Model Performance ---")
metrics = model.val()
print(f"Validation mAP50-95: {metrics.box.map}")
print(f"Validation mAP50: {metrics.box.map50}")

# --- 8. Run a Prediction! ---
print("\n--- Running a Test Prediction ---")
best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
best_model = YOLO(best_model_path)

test_image_url = 'https://tastesbetterfromscratch.com/wp-content/uploads/2023/06/Gourmet-Burger-1.jpg'

print(f"Predictions for: {test_image_url}\n")
pred_results = best_model(test_image_url)

for result in pred_results:
    boxes = result.boxes
    names = result.names

    if len(boxes) == 0:
        print("No objects detected.")
        continue

    for box in boxes:
        class_id = int(box.cls[0])
        class_name = names[class_id]
        confidence = box.conf[0]
        
        print(f"Detected: {class_name} (Confidence: {confidence*100:.2f}%)")