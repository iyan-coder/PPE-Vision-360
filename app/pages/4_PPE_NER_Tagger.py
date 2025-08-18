import os
import sys
import streamlit as st
from src.PPE_VISION_360.chat_engine.model_loader import get_ner

# -------------------------------
# Ensure project root is on path
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from src.PPE_VISION_360.chat_engine.ner_tagger import NERTagger
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception


# Set Streamlit page title
st.title("🧾 PPE NER Tagger")

try:
    
    # Load the custom NER model from saved path
    nlp_ner = get_ner(r"D:\PPE-Vision-360\models\ppe_ner_model")
    
    # Wrap loaded model inside NERTagger utility
    ner_bot = NERTagger(nlp_ner)

    # Create text input area for user to enter raw text
    text = st.text_area("Enter your text here...")

    # If button is clicked and input is not empty
    if st.button("🔍 Detect PPE Items") and text.strip():
        with st.spinner("Running NER..."):
            # Detect entities from text using NER model
            entities = ner_bot.detect_entities(text)
            
            # If entities are found, display them
            if entities:
                st.success("✅ Entities Detected:")
                for ent_text, ent_label in entities:
                    st.markdown(f"- **{ent_text}** → {ent_label}")
            else:
                # Show message if no PPE-related entities were detected
                st.info("No PPE items detected.")

# Catch all exceptions, log the error, and raise custom exception
except Exception as e:
    logger.error("Error in PPE NER Tagger module", exc_info=True)
    raise PpeVision360Exception(e, sys)
