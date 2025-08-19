import streamlit as st
import sys
from src.PPE_VISION_360.chat_engine.hybrid_recommender import HybridRecommender
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception

# ---------------- UI ----------------
st.title("🔀 Hybrid PPE Recommender (Image + Text)")
st.write("Upload an image and paste a PPE report to get a unified compliance recommendation.")

# Upload image + text
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
report_text = st.text_area("Paste PPE Report")

# Button to trigger the pipeline
if st.button("🚀 Run Hybrid Compliance Check"):
    try:
        logger.info("🚀 Hybrid Compliance Check triggered in PPE Vision 360 module")

        # Validation: user must provide both image + report text
        if not uploaded_file or not report_text.strip():
            st.warning("Please upload an image AND paste a report.")
            logger.warning("⚠️ User did not provide both image and text input.")
        else:
            logger.info("✅ Both image and text report provided by user")

            # Paths to ML models
            bert_path = r"D:\PPE-Vision-360\models\saved_distillbert"
            tokenizer_path = r"D:\PPE-Vision-360\models\saved_distillbert"
            ner_path = r"D:\PPE-Vision-360\models\ppe_ner_model"

            # Load hybrid recommender
            logger.info("📥 Loading Hybrid Recommender with models...")
            recommender = HybridRecommender(bert_path, tokenizer_path, ner_path)

            # Run recommendation
            logger.info("🔎 Running recommendation on uploaded image and report text")
            result = recommender.recommend(uploaded_file, report_text)

            # ---------------- Friendly Results Section ----------------
            st.markdown("## 📊 Hybrid Recommendation Results")
            st.write("**Image Status:**", result['image_status'])
            st.write("**Text Classification:**", result['text_label'])
            st.write("**Detected PPE (Image):**", result['detected_image_items'])
            st.write("**Detected PPE (Text):**", result['detected_text_items'])
            st.write("**Phase 7 Reasoning (Text):**", result['text_reasoning'])

            # Friendly formatted compliance report
            if "✅" in result['final_decision']:
                # If compliant ✅
                compliance_report = f"""
                ✅ PPE Compliance Check: **Compliant**

                All required PPE items are present.  
                Great job! 🎉 You and your team are protected and compliant.  

                👉 Keep following best practices to maintain safety on site.
                """
                st.success(compliance_report)
                logger.info(f"✅ Final Decision: {result['final_decision']} | Recommendation: {result['final_recommendation']}")

            else:
                # If non-compliant ⚠️
                compliance_report = """
                ⚠️ PPE Compliance Check: **Non-Compliant**

                Missing PPE items:
                - 🧤 Gloves
                - 👢 Boots
                - 🦺 Safety Vest
                - 🥽 Goggles

                👉 Recommendation: Please ensure all listed PPE is worn before entering the site.  
                Your safety comes first — compliance protects **you and your team**.
                """
                st.error(compliance_report)
                logger.error(f"❌ Final Decision: {result['final_decision']} | Recommendation: {result['final_recommendation']}")

    except Exception as e:
        # Logs the full stack trace if something breaks
        logger.exception("🔥 Error occurred in Hybrid Recommender Streamlit UI")
        raise PpeVision360Exception(e, sys)

    finally:
        # Always runs, even if error happens
        logger.info("🔚 Hybrid Compliance Check process completed in PPE Vision 360")
