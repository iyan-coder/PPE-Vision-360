import tensorflow as tf
import streamlit as st
import sys
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception

class BERTClassifier:
    def __init__(self, model, tokenizer, class_names=None):
        """
        Initialize BERTClassifier with model, tokenizer, and optional class names.
        """
        try:
            self.model = model
            self.tokenizer = tokenizer
            self.class_names = class_names or [
                "Emergency_Response",
                "Hazard_Reporting",
                "PPE_Compliance",
                "PPE_NonCompliance",
                "Safety_Procedure"
            ]

            # Log successful initialization
            logger.info("BERTClassifier initialized successfully with classes: %s", self.class_names)

        except Exception as e:
            # Log error with traceback and raise custom exception
            logger.error("Error initializing BERTClassifier", exc_info=True)
            raise PpeVision360Exception(e, sys)

    def classify(self, text: str):
        """
        Classify the input text into one of the predefined categories using BERT.
        Args:
            text (str): The input text to classify
        Returns:
            predicted_label (str): The predicted category
            confidence (float): The confidence score
        """
        try:
            # Tokenize the input text for BERT
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="tf"
            )
            logger.info("Text tokenization completed successfully")

            # Perform forward pass through the model
            outputs = self.model(**inputs)
            logits = outputs.logits
            logger.info("Model forward pass completed")

            # Convert logits to probabilities
            probs = tf.nn.softmax(logits, axis=-1)

            # Get the index of the highest probability class
            pred_class_idx = tf.argmax(probs, axis=1).numpy()[0]

            # Map index to actual label
            predicted_label = self.class_names[pred_class_idx]

            # Extract confidence score for predicted label
            confidence = probs[0, pred_class_idx].numpy()

            # Log the prediction result
            logger.info("Prediction completed: %s with confidence %.4f", predicted_label, confidence)

            return predicted_label, confidence

        except Exception as e:
            # Log error with traceback and raise custom exception
            logger.error("Error during text classification in BERTClassifier", exc_info=True)
            raise PpeVision360Exception(e, sys)
