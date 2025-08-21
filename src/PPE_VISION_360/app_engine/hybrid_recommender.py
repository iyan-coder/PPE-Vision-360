import sys
from src.PPE_VISION_360.app_engine.image_detection import ImageComplianceChecker
from src.PPE_VISION_360.app_engine.bert_classifier import BERTClassifier
from src.PPE_VISION_360.app_engine.ner_tagger import NERTagger
from src.PPE_VISION_360.app_engine.compliance_reasoning import generate_compliance_reasoning
from src.PPE_VISION_360.app_engine.model_loader import get_bert_drive, get_ner_drive
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception


class HybridRecommender:
    def __init__(self, bert_model_path, tokenizer_path, ner_model_path):
        try:
            # Load models
            self.bert_model, self.bert_tokenizer = get_bert_drive(bert_model_path, tokenizer_path)
            self.ner_model = get_ner_drive(ner_model_path)

            # Initialize modules
            self.image_checker = ImageComplianceChecker()
            self.bert_classifier = BERTClassifier(model=self.bert_model, tokenizer=self.bert_tokenizer)
            self.ner_tagger = NERTagger(nlp_model=self.ner_model)

            logger.info("Hybrid Recommender initialized successfully.")
        except Exception as e:
            logger.error("Error initializing Hybrid Recommender", exc_info=True)
            raise PpeVision360Exception(e, sys)

    def recommend(self, image_file, report_text):
        try:
            # ---------------- IMAGE PIPELINE ----------------
            image_result = self.image_checker.check_image(image_file)
            image_status = image_result.get("overall_status", "Unknown")
            detected_from_image = set(image_result.get("detected_items", []))

            # ---------------- TEXT PIPELINE ----------------
            bert_label = self.bert_classifier.classify(report_text)[0]
            ner_items = [ent[0] if isinstance(ent, (tuple, list)) else ent for ent in self.ner_tagger.detect_entities(report_text)]

            # ---------------- COMPLIANCE REASONING ----------------
            reasoning_text = generate_compliance_reasoning(bert_label, ner_items)

            # ---------------- FUSION ----------------
            fusion_recommendation = {
                "image_status": image_status,
                "text_label": bert_label,
                "text_reasoning": reasoning_text,
                "detected_image_items": list(detected_from_image),
                "detected_text_items": ner_items,
            }

            # Rule: If either image or text shows missing PPE → flag as non-compliant
            if "Non-Compliant" in bert_label or image_status != "Fully Compliant":
                fusion_recommendation["final_decision"] = "❌ Non-Compliant"
                fusion_recommendation["final_recommendation"] = (
                    f"Check PPE compliance: {reasoning_text}. "
                    f"Image also flagged: {image_status}"
                )
            else:
                fusion_recommendation["final_decision"] = "✅ Compliant"
                fusion_recommendation["final_recommendation"] = "Worker is fully compliant based on both text and image."

            return fusion_recommendation

        except Exception as e:
            logger.error("Error in hybrid recommendation process", exc_info=True)
            raise PpeVision360Exception(e, sys)
