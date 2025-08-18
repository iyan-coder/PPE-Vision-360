# chat_engine/compliance_reasoning.py
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception
import sys

def generate_compliance_reasoning(bert_label, ner_items):
    """
    Generate human-readable reasoning for PPE compliance reports based on
    BERT classification and NER-detected PPE items.

    Args:
        bert_label (str): BERT classification label of the report.
        ner_items (list): List of PPE items extracted from the report.

    Returns:
        str: Human-readable compliance reasoning.
    """
    try:
        required_ppe = ["helmet", "gloves", "boots", "vest", "goggles"]

        # Ensure NER items are strings, not tuples/lists
        ner_items = [item[0] if isinstance(item, (tuple, list)) else item for item in ner_items]

        missing_items = [item for item in required_ppe if item not in ner_items]

        # Reasoning logic
        if missing_items:
            raw_text = (
                f"Report is non-compliant. Missing PPE: {', '.join(missing_items)}. "
                "Please add them for full safety compliance."
            )
            logger.info(f"Missing PPE detected: {missing_items}")
        else:
            if bert_label == "PPE_Compliance":
                raw_text = "This report is compliant. All required PPE items are present."
            elif bert_label == "PPE_NonCompliance":
                raw_text = (
                    "Report flagged as non-compliant by BERT, "
                    "but all required PPE items are present. Please check proper usage."
                )
            elif bert_label == "Emergency_Response":
                raw_text = (
                    "This report relates to an emergency response. Ensure all safety protocols are followed, "
                    "emergency exits are accessible, and PPE is worn correctly during response."
                )
            elif bert_label == "Hazard_Reporting":
                raw_text = (
                    "This report is a hazard report. Verify the identified hazards are addressed promptly, "
                    "and relevant PPE is used when handling the hazards."
                )
            elif bert_label == "Safety_Procedure":
                raw_text = (
                    "This report describes a safety procedure. Confirm that the procedure includes all required PPE, "
                    "instructions are clear, and compliance is regularly monitored."
                )
            else:
                raw_text = f"Report classified as {bert_label}. Detected PPE items: {', '.join(ner_items)}."

        logger.info(f"Phase 7 Reasoning Generated | BERT Label: {bert_label} | NER Items: {ner_items} | Reasoning: {raw_text}")
        return raw_text

    except Exception as e:
        logger.error("Error during Phase 7 compliance reasoning", exc_info=True)
        raise PpeVision360Exception(e, sys)
