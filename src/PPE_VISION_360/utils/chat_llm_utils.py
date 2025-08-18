import os
import sys
import requests
from dotenv import load_dotenv  # To load API keys from .env file
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception

# Load environment variables from .env file
load_dotenv()

class GroqClient:
    def __init__(self, api_key=None):
        try:
            # Use provided API key or fallback to environment variable
            self.api_key = api_key or os.getenv("GROQ_API_KEY")

            # Log info about initialization
            logger.info("GroqClient initialized successfully.")

        except Exception as e:
            # Log and raise custom exception if initialization fails
            logger.error("Error during GroqClient initialization", exc_info=True)
            raise PpeVision360Exception(e, sys)

    def query(self, messages):
        """
        Send a chat request to the Groq API with given messages
        """
        try:
            # Ensure API key is set, else return error
            if not self.api_key:
                logger.error("GROQ_API_KEY is not set in environment.")
                return "Error: GROQ_API_KEY not set"

            # Define HTTP request headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            # Define request payload (model + messages)
            payload = {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": messages
            }

            logger.info("Sending request to Groq API...")

            # Make POST request to Groq API with timeout
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            # Parse response JSON
            data = response.json()

            # Check if response was successful
            if response.status_code == 200:
                logger.info("Groq API response received successfully.")
                return data.get("choices", [{}])[0].get("message", {}).get("content", "No content")
            else:
                logger.error(f"Groq API Error {response.status_code}: {data}")
                return f"API Error {response.status_code}: {data}"

        except Exception as e:
            # Catch any exception during request/processing
            logger.error("Exception occurred while querying Groq API", exc_info=True)
            raise PpeVision360Exception(e, sys)
