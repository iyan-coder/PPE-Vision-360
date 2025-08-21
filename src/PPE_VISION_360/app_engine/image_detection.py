import sys
import os
from PIL import Image
import numpy as np
import streamlit as st
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception

# Import PPE detection utils
from src.PPE_VISION_360.utils.fastapi_utils import (
    detect_and_classify,
    detection_model,
    gloves_classifier,
    helmet_classifier,
    vest_classifier,
    shoes_classifier
)

class ImageComplianceChecker:
    def __init__(self):
        try:
            logger.info("ImageComplianceChecker initialized successfully (local mode).")
        except Exception as e:
            logger.error("Error during ImageComplianceChecker initialization", exc_info=True)
            raise PpeVision360Exception(e, sys)

    def check_image(self, uploaded_file):
        """
        Process an uploaded image using local PPE detection/classification models.
        Reuses detect_and_classify() pipeline from utils.
        """
        try:
            # Save temporarily in memory to pass to OpenCV
            temp_path = f"temp_{uploaded_file.name}"
            image = Image.open(uploaded_file).convert("RGB")
            image.save(temp_path)

            logger.info("Running PPE detection/classification...")
            result = detect_and_classify(
                temp_path,
                detection_model=detection_model,
                gloves_classifier=gloves_classifier,
                helmet_classifier=helmet_classifier,
                vest_classifier=vest_classifier,
                shoes_classifier=shoes_classifier
            )

            logger.info("PPE detection/classification completed successfully.")
            return result

        except Exception as e:
            logger.error("Exception occurred while checking image compliance", exc_info=True)
            raise PpeVision360Exception(e, sys)

