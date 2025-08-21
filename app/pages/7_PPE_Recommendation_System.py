import streamlit as st
import sys
import os

# -----------------------------
# Ensure src path is visible
# -----------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.PPE_VISION_360.logger.logger import logger
logger.info(f"Project root added to sys.path: {project_root}")

from src.PPE_VISION_360.chat_engine.hybrid_recommender import HybridRecommender
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

            # Paths to ML models (Google Drive IDs)
            bert_path = "1v5024dYPwsYmoA4UC97_mHt0x3rdaASH"
            tokenizer_path = "1v5024dYPwsYmoA4UC97_mHt0x3rdaASH"
            ner_path = "1OrHQb7f03nvUA7hUO_zulg3BClWP3WVW"

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

            # ---------------- Dynamic Compliance Logic ----------------
            required_ppe = {"Gloves", "Boots", "Vest", "Goggles"}  # Safety Vest -> Vest
            detected_ppe = set(result['detected_image_items']) | set(result['detected_text_items'])
            missing_ppe = required_ppe - detected_ppe

            # Recommendations per missing item
            recommendations = {
                "Gloves": "Wear safety gloves to protect your hands.",
                "Boots": "Wear safety boots to protect your feet.",
                "Vest": "Wear a reflective vest.",
                "Goggles": "Wear protective goggles."
            }

            if not missing_ppe:
                # Fully compliant
                compliance_report = f"""
✅ PPE Compliance Check: **Compliant** 🎉

All required PPE items are present: {', '.join(detected_ppe)}  

👉 Keep following best practices to maintain safety on site.
"""
                st.success(compliance_report)
                logger.info(f"✅ Final Decision: Compliant | Detected PPE: {', '.join(detected_ppe)}")
            else:
                # Non-compliant, dynamic missing list
                missing_list = ", ".join(missing_ppe)
                detected_list = ", ".join(detected_ppe) if detected_ppe else "None detected"
                missing_recs = [recommendations[item] for item in missing_ppe]
                missing_recs_str = "\n- ".join(missing_recs)  # precompute string to avoid backslash in f-string

                compliance_report = f"""
⚠️ PPE Compliance Check: **Non-Compliant**

Detected PPE: {detected_list}  
Missing PPE: {missing_list}

👉 Recommendations:
- {missing_recs_str}
"""
                st.error(compliance_report)
                logger.warning(f"❌ Final Decision: Non-Compliant | Detected: {detected_list} | Missing: {missing_list}")

    except Exception as e:
        # Logs the full stack trace if something breaks
        logger.exception("🔥 Error occurred in Hybrid Recommender Streamlit UI")
        raise PpeVision360Exception(e, sys)

    finally:
        # Always runs, even if error happens
        logger.info("🔚 Hybrid Compliance Check process completed in PPE Vision 360")
