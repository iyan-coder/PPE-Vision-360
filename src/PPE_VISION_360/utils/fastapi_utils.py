import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception
import sys

# -------------------------------------------------------
# Load YOLOv8 detection model and all PPE classifiers
# Wrapped in try/except for robust error handling
# -------------------------------------------------------
try:
    # Load YOLOv8 detection model
    logger.info("Loading YOLOv8 Detection Model...")
    detection_model = YOLO(r'D:\PPE-Vision-360\models\ppe_detection_best.pt')
    logger.info("YOLOv8 Detection Model Loaded Successfully!")

    # Load PPE Classifiers (Helmet, Gloves, Vest, Shoes)
    logger.info("Loading Gloves Classifier...")
    gloves_classifier = tf.saved_model.load(
        r'D:\PPE-Vision-360\models\best_glove_classifier_static'
    ).signatures['serving_default']
    logger.info("Gloves Classifier Loaded!")

    logger.info("Loading Helmet Classifier...")
    helmet_classifier = tf.saved_model.load(
        r'D:\PPE-Vision-360\models\best_helmet_classifier_static'
    ).signatures['serving_default']
    logger.info("Helmet Classifier Loaded!")

    logger.info("Loading Vest Classifier...")
    vest_classifier = tf.saved_model.load(
        r'D:\PPE-Vision-360\models\best_vest_classifier_static'
    ).signatures['serving_default']
    logger.info("Vest Classifier Loaded!")

    logger.info("Loading Shoes Classifier...")
    shoes_classifier = tf.saved_model.load(
        r'D:\PPE-Vision-360\models\best_shoe_classifier_static'
    ).signatures['serving_default']
    logger.info("Shoes Classifier Loaded!")

except Exception as e:
    logger.error("❌ Error while loading models", exc_info=True)
    raise PpeVision360Exception(e, sys)

# -------------------------------------------------------
# Class ID Mapping from YOLO detection
# -------------------------------------------------------
class_map = {
    0: 'Helmet',
    1: 'Gloves',
    2: 'Vest',
    3: 'Shoes'
}

# -------------------------------------------------------
# Preprocess Crop for Classifier
# Ensures correct shape, size, and channels
# -------------------------------------------------------
def preprocess_crop(crop):
    try:
        logger.debug(f"Original Crop Shape: {crop.shape}")

        # Convert grayscale to RGB
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            logger.debug("Converted Grayscale to RGB")

        # Handle single-channel images
        elif crop.shape[2] == 1:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            logger.debug("Converted Single Channel to RGB")

        # Handle images with alpha channel
        elif crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
            logger.debug("Removed Alpha Channel")

        # Resize to classifier input size
        crop = cv2.resize(crop, (224, 224))
        logger.debug(f"Resized Crop: {crop.shape}")

        # Normalize pixel values
        crop = crop.astype('float32') / 255.0

        # Add batch dimension
        crop = np.expand_dims(crop, axis=0)
        logger.debug(f"Final Preprocessed Crop Shape: {crop.shape}")

        return tf.convert_to_tensor(crop)

    except Exception as e:
        logger.error("Error during crop preprocessing", exc_info=True)
        raise PpeVision360Exception(e, sys)

# -------------------------------------------------------
# Detect + Classify PPE Items
# Runs YOLOv8 detection, then classifier validation
# -------------------------------------------------------
def detect_and_classify(image_path):
    try:
        logger.info(f"Processing Image: {image_path}")

        # Load image in OpenCV
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            logger.error("Failed to load image")
            return {"error": "Image could not be loaded."}

        # Convert to RGB for YOLO
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Run YOLOv8 detection
        logger.info("Running YOLOv8 Detection...")
        detections = detection_model(image)[0]
        logger.info(f"Detection Results: {len(detections.boxes)} items found.")

        compliance_result = {}
        expected_items = {'Helmet', 'Gloves', 'Vest', 'Shoes'}

        # Loop over detections
        for idx, det in enumerate(detections.boxes):
            class_id = int(det.cls[0])
            class_name = class_map.get(class_id, None)
            logger.debug(f"Detection {idx+1}: Class ID={class_id}, Name={class_name}")

            if class_name is None:
                logger.warning("Skipped unknown class.")
                continue

            # Crop detected PPE region
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            logger.debug(f"Cropping: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            crop = image[y1:y2, x1:x2]
            crop_tensor = preprocess_crop(crop)

            # Run appropriate classifier
            logger.info(f"Running {class_name} Classifier...")
            if class_name == 'Helmet':
                raw_output = helmet_classifier(crop_tensor)
            elif class_name == 'Gloves':
                raw_output = gloves_classifier(crop_tensor)
            elif class_name == 'Vest':
                raw_output = vest_classifier(crop_tensor)
            elif class_name == 'Shoes':
                raw_output = shoes_classifier(crop_tensor)
            else:
                logger.warning(f"Unknown class skipped: {class_name}")
                continue

            # Extract prediction
            output_tensor = list(raw_output.values())[0]
            prediction = output_tensor[0][0].numpy()
            compliance = 'Compliant' if prediction >= 0.5 else 'Non-Compliant'
            compliance_result[class_name] = compliance
            logger.info(f"{class_name}: {compliance} (Score: {prediction:.4f})")

        # Handle missing PPE items
        for item in expected_items:
            if item not in compliance_result:
                compliance_result[item] = 'Not Detected (Non-Compliant)'
                logger.warning(f"{item}: Not Detected (Marked Non-Compliant)")

        # Determine overall status
        overall_status = (
            'Fully Compliant' 
            if all(v == 'Compliant' for v in compliance_result.values()) 
            else 'Non-Compliant'
        )
        logger.info(f"Overall Compliance Status: {overall_status}")

        return {
            'item_results': compliance_result,
            'overall_status': overall_status
        }

    except Exception as e:
        logger.error("Error during detection/classification pipeline", exc_info=True)
        raise PpeVision360Exception(e, sys)
