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

from src.PPE_VISION_360.chat_engine.image_detection import ImageComplianceChecker
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception

# ------------------ Streamlit UI ------------------
st.title("🖼️ Image Compliance Checker")

# Initialize compliance checker
checker = ImageComplianceChecker()

# File uploader widget
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Button to trigger compliance check
    if st.button("🚀 Check Compliance"):
        try:
            with st.spinner("Analyzing Image..."):
                # Run compliance check
                result = checker.check_image(uploaded_file)

                # ✅ If results are returned
                if result:
                    # Loop through each detected item
                    for item, status in result['item_results'].items():
                        color = "#34A853" if "Compliant" in status else "#EA4335"
                        icon = "✅" if "Compliant" in status else "❌"

                        # Show each item compliance status
                        st.markdown(
                            f"<span style='color:{color}; font-weight:bold'>{icon} {item}: {status}</span>",
                            unsafe_allow_html=True
                        )

                    # Show overall compliance status
                    overall_icon = "✅" if result['overall_status']=="Fully Compliant" else "🚨"
                    st.markdown(f"### {overall_icon} **Overall Status:** {result['overall_status']}")

        except Exception as e:
            # Log error with stack trace
            logger.error("Error during image compliance check", exc_info=True)

            # Raise custom exception for consistency
            raise PpeVision360Exception(e, sys)
