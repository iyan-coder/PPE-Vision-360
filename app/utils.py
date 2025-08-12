import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# Load YOLOv8 Detection Model
print("Loading Detection Model...")
detection_model = YOLO(r'D:\PPE-Vision-360\models\ppe_detection_best.pt')
print("Detection Model Loaded!")

# Load Classifiers using tf.saved_model.load
print("Loading Gloves Classifier...")
gloves_classifier = tf.saved_model.load(r'D:\PPE-Vision-360\models\best_glove_classifier_static').signatures['serving_default']
print("Gloves Classifier Loaded!")

print("Loading Helmet Classifier...")
helmet_classifier = tf.saved_model.load(r'D:\PPE-Vision-360\models\best_helmet_classifier_static').signatures['serving_default']
print("Helmet Classifier Loaded!")

print("Loading Vest Classifier...")
vest_classifier = tf.saved_model.load(r'D:\PPE-Vision-360\models\best_vest_classifier_static').signatures['serving_default']
print("Vest Classifier Loaded!")

print("Loading Shoes Classifier...")
shoes_classifier = tf.saved_model.load(r'D:\PPE-Vision-360\models\best_shoe_classifier_static').signatures['serving_default']
print("Shoes Classifier Loaded!")

# Class ID Mapping
class_map = {
    0: 'Helmet',
    1: 'Gloves',
    2: 'Vest',
    3: 'Shoes'
}

# Preprocess Crop
def preprocess_crop(crop):
    print(f"Original Crop Shape: {crop.shape}")
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        print("Converted Grayscale to RGB")
    elif crop.shape[2] == 1:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        print("Converted Single Channel to RGB")
    elif crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
        print("Removed Alpha Channel to BGR")

    crop = cv2.resize(crop, (224, 224))
    print(f"🔍 Resized Crop to: {crop.shape}")
    crop = crop.astype('float32') / 255.0
    crop = np.expand_dims(crop, axis=0)
    print(f"Final Preprocessed Crop Shape: {crop.shape}")
    return tf.convert_to_tensor(crop)

# Detect + Classify PPE
def detect_and_classify(image_path):
    print(f"Processing Image: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print("Failed to load image!")
        return {"error": "Image could not be loaded."}

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print("Running YOLOv8 Detection...")
    detections = detection_model(image)[0]
    print(f"Detection Results: {len(detections.boxes)} items found.")

    compliance_result = {}
    expected_items = {'Helmet', 'Gloves', 'Vest', 'Shoes'}

    for idx, det in enumerate(detections.boxes):
        class_id = int(det.cls[0])
        class_name = class_map.get(class_id, None)
        print(f"Processing Detection {idx+1}: Class ID={class_id}, Name={class_name}")

        if class_name is None:
            print("Skipped unknown class.")
            continue

        x1, y1, x2, y2 = map(int, det.xyxy[0])
        print(f"Cropping Coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        crop = image[y1:y2, x1:x2]
        crop_tensor = preprocess_crop(crop)

        # Run Inference
        print(f"Running Classifier for {class_name}...")
        if class_name == 'Helmet':
            raw_output = helmet_classifier(crop_tensor)
        elif class_name == 'Gloves':
            raw_output = gloves_classifier(crop_tensor)
        elif class_name == 'Vest':
            raw_output = vest_classifier(crop_tensor)
        elif class_name == 'Shoes':
            raw_output = shoes_classifier(crop_tensor)
        else:
            print(f"Unknown class: {class_name}")
            continue

        # Get prediction from output
        output_tensor = list(raw_output.values())[0]
        prediction = output_tensor[0][0].numpy()
        compliance = 'Compliant' if prediction >= 0.5 else 'Non-Compliant'
        compliance_result[class_name] = compliance
        print(f"{class_name}: {compliance} (Prediction Score: {prediction:.4f})")

    # Handle Missing Items
    for item in expected_items:
        if item not in compliance_result:
            compliance_result[item] = 'Not Detected (Non-Compliant)'
            print(f"{item}: Not Detected (Marked Non-Compliant)")

    overall_status = 'Fully Compliant' if all(v == 'Compliant' for v in compliance_result.values()) else 'Non-Compliant'
    print(f"🏁 Overall Compliance Status: {overall_status}")

    return {
        'item_results': compliance_result,
        'overall_status': overall_status
    }
