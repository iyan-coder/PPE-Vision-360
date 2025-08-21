import os
import sys
import streamlit as st
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception
# -----------------------------
# Ensure src path is visible
# -----------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.PPE_VISION_360.logger.logger import logger
logger.info(f"Project root added to sys.path: {project_root}")

from src.PPE_VISION_360.app_engine.model_loader import get_bert_drive
from src.PPE_VISION_360.app_engine.bert_classifier import BERTClassifier

# 🧠 Set Streamlit page title
st.title("🧠 BERT Text Classifier")

# -------------------------------
# Load BERT model + tokenizer with caching
# -------------------------------
try:
    bert_model, bert_tokenizer = get_bert_drive(
        "1v5024dYPwsYmoA4UC97_mHt0x3rdaASH",  # BERT zip on Drive
        "1v5024dYPwsYmoA4UC97_mHt0x3rdaASH"
    )


    # ✅ Create classifier instance with loaded model + tokenizer
    classifier = BERTClassifier(bert_model, bert_tokenizer)

except Exception as e:
    logger.error("Error while loading BERT model for text classification", exc_info=True)
    raise PpeVision360Exception(e, sys)


# -------------------------------
# Streamlit input
# -------------------------------
text = st.text_area("Enter your text here...")

# -------------------------------
# Run classification when button clicked
# -------------------------------
if st.button("Classify Text") and text.strip():
    try:
        with st.spinner("Classifying..."):
            label, confidence = classifier.classify(text)

            # ✅ Show results
            st.success(f"Prediction: **{label}**")
            st.write(f"Confidence: {confidence:.2%}")

    except Exception as e:
        logger.error("Error during text classification with BERT", exc_info=True)
        raise PpeVision360Exception(e, sys)
