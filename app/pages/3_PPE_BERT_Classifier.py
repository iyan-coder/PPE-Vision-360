import os
import sys
import streamlit as st
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception

# -------------------------------
# Ensure project root is on path
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from src.PPE_VISION_360.chat_engine.model_loader import get_bert
from src.PPE_VISION_360.chat_engine.bert_classifier import BERTClassifier

# 🧠 Set Streamlit page title
st.title("🧠 BERT Text Classifier")

# -------------------------------
# Load BERT model + tokenizer with caching
# -------------------------------
try:
    model, tokenizer = get_bert(
        r"D:\PPE-Vision-360\models\saved_distillbert", 
        r"D:\PPE-Vision-360\models\saved_distillbert"
    )

    # ✅ Create classifier instance with loaded model + tokenizer
    classifier = BERTClassifier(model, tokenizer)

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
