import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception
import streamlit as st
import gdown
import zipfile

# -------------------------------------------------------
# Helper to download and unzip from Google Drive
# -------------------------------------------------------
def download_from_drive(file_id, out_path, unzip=False):
    """Download file/folder from Google Drive and optionally unzip."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not os.path.exists(out_path):
        tmp_path = out_path + ".zip" if unzip else out_path
        logger.info(f"Downloading {file_id} to {tmp_path}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", tmp_path, quiet=False)
        if unzip:
            logger.info(f"Unzipping {tmp_path} to {out_path}...")
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                zip_ref.extractall(out_path)
            os.remove(tmp_path)  # remove zip after extraction
    else:
        logger.info(f"{out_path} already exists, skipping download.")
    return out_path

# -------------------------------------------------------
# Drive file IDs for models
# -------------------------------------------------------
FILE_IDS = {
    "yolo": "1bnrPYsBq2oaPb7c14QVSs3vRli6x8AQk",
    "gloves": "1qtQannlG65EhI6k5SLNlBWrHvDNY4aOl",
    "helmet": "1ifqgjuGV_LVw-K8uDjhoBBch4YEjcuJy",
    "vest": "1WuPcvOfr7spZx7TcfBZw5tPvI9zritPf",
    "shoes": "1103pHSDWEEs2wXwSAV4wCFy8prm9c5yR"
}

# -------------------------------------------------------
# Cached function to load YOLO + PPE classifiers from Drive
# -------------------------------------------------------
@st.cache_resource(show_spinner=True)
def get_models_from_drive(file_ids):
    try:
        # YOLOv8 Detection Model
        yolo_path = download_from_drive(file_ids['yolo'], "models/ppe_detection_best.pt")
        detection_model = YOLO(yolo_path)

        # PPE Classifiers
        gloves_path = download_from_drive(file_ids['gloves'], "models/best_glove_classifier_static", unzip=True)
        helmet_path = download_from_drive(file_ids['helmet'], "models/best_helmet_classifier_static", unzip=True)
        vest_path = download_from_drive(file_ids['vest'], "models/best_vest_classifier_static", unzip=True)
        shoes_path = download_from_drive(file_ids['shoes'], "models/best_shoe_classifier_static", unzip=True)

        gloves_classifier = tf.saved_model.load(gloves_path).signatures['serving_default']
        helmet_classifier = tf.saved_model.load(helmet_path).signatures['serving_default']
        vest_classifier = tf.saved_model.load(vest_path).signatures['serving_default']
        shoes_classifier = tf.saved_model.load(shoes_path).signatures['serving_default']

        logger.info("All models loaded and cached successfully!")
        return detection_model, gloves_classifier, helmet_classifier, vest_classifier, shoes_classifier

    except Exception as e:
        logger.error("Error loading models from Drive", exc_info=True)
        raise PpeVision360Exception(e, sys)

# -------------------------------------------------------
# Class ID Mapping
# -------------------------------------------------------
class_map = {0: 'Helmet', 1: 'Gloves', 2: 'Vest', 3: 'Shoes'}

# -------------------------------------------------------
# Preprocess crop for PPE classifier
# -------------------------------------------------------
def preprocess_crop(crop):
    try:
        if crop.ndim == 2:  # grayscale
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.shape[2] == 1:  # single channel
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.shape[2] == 4:  # alpha channel
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)

        crop = cv2.resize(crop, (224, 224))
        crop = crop.astype('float32') / 255.0
        crop = np.expand_dims(crop, axis=0)
        return tf.convert_to_tensor(crop)

    except Exception as e:
        logger.error("Error during crop preprocessing", exc_info=True)
        raise PpeVision360Exception(e, sys)

# -------------------------------------------------------
# Detect + Classify PPE Items
# -------------------------------------------------------
def detect_and_classify(image_path, detection_model, gloves_classifier,
                        helmet_classifier, vest_classifier, shoes_classifier):
    try:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            logger.error("Failed to load image")
            return {"error": "Image could not be loaded."}

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detections = detection_model(image)[0]

        compliance_result = {}
        expected_items = {'Helmet', 'Gloves', 'Vest', 'Shoes'}

        for idx, det in enumerate(detections.boxes):
            class_id = int(det.cls[0])
            class_name = class_map.get(class_id, None)
            if class_name is None:
                continue

            x1, y1, x2, y2 = map(int, det.xyxy[0])
            crop = image[y1:y2, x1:x2]
            crop_tensor = preprocess_crop(crop)

            if class_name == 'Helmet':
                raw_output = helmet_classifier(crop_tensor)
            elif class_name == 'Gloves':
                raw_output = gloves_classifier(crop_tensor)
            elif class_name == 'Vest':
                raw_output = vest_classifier(crop_tensor)
            elif class_name == 'Shoes':
                raw_output = shoes_classifier(crop_tensor)
            else:
                continue

            output_tensor = list(raw_output.values())[0]
            prediction = output_tensor[0][0].numpy()
            compliance_result[class_name] = 'Compliant' if prediction >= 0.5 else 'Non-Compliant'

        # Handle missing items
        for item in expected_items:
            if item not in compliance_result:
                compliance_result[item] = 'Not Detected (Non-Compliant)'

        overall_status = 'Fully Compliant' if all(v == 'Compliant' for v in compliance_result.values()) else 'Non-Compliant'
        return {'item_results': compliance_result, 'overall_status': overall_status}

    except Exception as e:
        logger.error("Error during detection/classification pipeline", exc_info=True)
        raise PpeVision360Exception(e, sys)

# -------------------------------------------------------
# Load all models once (cached)
# -------------------------------------------------------
detection_model, gloves_classifier, helmet_classifier, vest_classifier, shoes_classifier = get_models_from_drive(FILE_IDS)
