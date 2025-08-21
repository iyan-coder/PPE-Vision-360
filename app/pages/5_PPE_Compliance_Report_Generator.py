import streamlit as st
import os
import sys
import re

# -----------------------------
# Ensure src path is visible
# -----------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.PPE_VISION_360.logger.logger import logger
logger.info(f"Project root added to sys.path: {project_root}")

from src.PPE_VISION_360.app_engine.compliance_reasoning import generate_compliance_reasoning
from src.PPE_VISION_360.app_engine.bert_classifier import BERTClassifier
from src.PPE_VISION_360.app_engine.ner_tagger import NERTagger
from src.PPE_VISION_360.app_engine.model_loader import get_bert_drive, get_ner_drive
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception

st.title("📝 PPE Compliance Reasoning - Batch Mode (Negation-aware, Color Highlight)")
st.write("Paste multiple PPE reports below, one per line:")

user_input = st.text_area("Enter your text here...", height=200)

if st.button("Generate Reasoning") and user_input.strip():
    try:
        logger.info("Batch reasoning started.")
        
        reports = [line.strip() for line in user_input.split("\n") if line.strip()]
        logger.info(f"Total reports received: {len(reports)}")

        # -----------------------------
        # Load cached models
        # -----------------------------
        bert_model_path = "1v5024dYPwsYmoA4UC97_mHt0x3rdaASH",
        tokenizer_path = "1v5024dYPwsYmoA4UC97_mHt0x3rdaASH",
        ner_model_path = "1OrHQb7f03nvUA7hUO_zulg3BClWP3WVW"

        bert_model, bert_tokenizer = get_bert_drive(bert_model_path, tokenizer_path)
        logger.info("BERT model and tokenizer loaded successfully.")

        ner_model = get_ner_drive(ner_model_path)
        logger.info("NER model loaded successfully.")

        bert_classifier = BERTClassifier(model=bert_model, tokenizer=bert_tokenizer)
        ner_tagger = NERTagger(nlp_model=ner_model)

        # -----------------------------
        # PPE & negation setup
        # -----------------------------
        required_ppe = ["helmet", "gloves", "boots", "vest", "goggles"]
        negation_patterns = [
            r"without (\w+)",
            r"missing (\w+)",
            r"not wearing (\w+)",
        ]

        for i, report in enumerate(reports, start=1):
            logger.info(f"Processing report {i}: {report}")

            # BERT classification
            bert_label_raw = bert_classifier.classify(report)
            bert_label = bert_label_raw[0]
            logger.info(f"BERT classification: {bert_label}")

            # NER extraction
            ner_items_raw = ner_tagger.detect_entities(report)
            ner_items = [
                str(item[0]) if isinstance(item, (tuple, list)) else str(item)
                for item in ner_items_raw
                if str(item[0] if isinstance(item, (tuple, list)) else item).lower() in required_ppe
            ]
            logger.info(f"NER detected PPE items: {ner_items}")

            # Negation handling
            negated_items = set()
            report_lower = report.lower()
            for pattern in negation_patterns:
                matches = re.findall(pattern, report_lower)
                for match in matches:
                    if match in required_ppe:
                        negated_items.add(match)
            logger.info(f"Negated items detected: {negated_items}")

            # Remove negated items from detected PPE (they are considered missing)
            ner_items = [item for item in ner_items if item not in negated_items]

            # Phase 7 reasoning generation
            reasoning_text = generate_compliance_reasoning(bert_label, ner_items)
            logger.info(f"Generated reasoning for report {i}")

            # Highlight missing and detected items
            missing_items = [item for item in required_ppe if item not in ner_items]
            detected_items = [item for item in ner_items if item in required_ppe]

            highlighted_reasoning = reasoning_text
            for item in missing_items:
                highlighted_reasoning = highlighted_reasoning.replace(
                    item, f"<span style='color:red;font-weight:bold'>{item}</span>"
                )
            for item in detected_items:
                highlighted_reasoning = highlighted_reasoning.replace(
                    item, f"<span style='color:green;font-weight:bold'>{item}</span>"
                )

            # Display results
            st.markdown(f"### Report {i}")
            st.write("**Text:**", report)
            st.write("**BERT Classification:**", bert_label)
            st.write("**Detected PPE Items:**", ner_items)
            st.markdown("**Phase 7 Reasoning:**", unsafe_allow_html=True)
            st.markdown(highlighted_reasoning, unsafe_allow_html=True)
            st.markdown("---")
            logger.info(f"Displayed report {i} results.")

        logger.info("Batch reasoning completed successfully.")

    except Exception as e:
        logger.error("Error in PPE Compliance Reasoning batch module.", exc_info=True)
        raise PpeVision360Exception(e, sys)
