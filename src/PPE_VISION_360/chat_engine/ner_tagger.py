import sys
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception

class NERTagger:
    def __init__(self, nlp_model):
        """
        Initialize the NERTagger class with a given NLP model.
        
        Args:
            nlp_model: A loaded spaCy or HuggingFace NLP model capable of NER.
        """
        try:
            self.nlp_model = nlp_model  # Store model for later use
            logger.info("NERTagger initialized successfully with provided NLP model.")
        except Exception as e:
            # Log the error and raise custom exception if init fails
            logger.error("Error initializing NERTagger.", exc_info=True)
            raise PpeVision360Exception(e, sys)

    def detect_entities(self, text: str):
        """
        Detect named entities in the provided text.
        
        Args:
            text (str): The input text where entities should be extracted.
        
        Returns:
            list: A list of tuples containing (entity_text, entity_label).
        """
        try:
            # Run NER model on the input text
            doc = self.nlp_model(text)

            # Extract entities as (text, label) pairs
            entities = [(ent.text, ent.label_) for ent in doc.ents] if doc.ents else []

            # Log detected entities
            logger.info(f"Detected entities: {entities}")

            return entities

        except Exception as e:
            # Log the error and raise custom exception if entity detection fails
            logger.error("Error during entity detection in NERTagger.", exc_info=True)
            raise PpeVision360Exception(e, sys)
