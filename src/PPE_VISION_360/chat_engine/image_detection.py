import requests
from PIL import Image
import io
import sys
import streamlit as st
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception


class ImageComplianceChecker:
    def __init__(self, api_url="http://127.0.0.1:8000/check_compliance"):
        try:
            # Store API URL (default: local FastAPI service)
            self.api_url = api_url
            logger.info("ImageComplianceChecker initialized successfully.")
        except Exception as e:
            # Log and raise custom exception if init fails
            logger.error("Error during ImageComplianceChecker initialization", exc_info=True)
            raise PpeVision360Exception(e, sys)

    def check_image(self, uploaded_file):
        """
        Send an uploaded image to the compliance API and return the response
        """
        try:
            # Open the uploaded file as a PIL image
            image = Image.open(uploaded_file)

            # Convert image into in-memory byte stream (PNG format)
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)  # Reset pointer to start of stream

            # Prepare file payload for POST request
            files = {'file': (uploaded_file.name, img_bytes, 'image/png')}

            logger.info("Sending image to compliance API...")

            # Send image to compliance API
            response = requests.post(self.api_url, files=files)

            # If API request is successful
            if response.status_code == 200:
                logger.info("Compliance API response received successfully.")
                return response.json()
            else:
                # Log and show error if API fails
                logger.error(f"Compliance API Error {response.status_code}: {response.text}")
                st.error(f"API Failed: {response.status_code}")
                return None

        except Exception as e:
            # Catch any error (e.g., network, PIL, JSON parsing)
            logger.error("Exception occurred while checking image compliance", exc_info=True)
            st.error(f"API Exception: {e}")
            raise PpeVision360Exception(e, sys)
