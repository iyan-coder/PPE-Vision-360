import sys
import os
import streamlit as st

# -------------------------------
# Ensure project root is on path
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from src.PPE_VISION_360.chat_engine.model_loader import get_faiss_drive
from src.PPE_VISION_360.chat_engine.osha_qa import OSHAQA
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception

# 🚀 Streamlit app title
st.title("💬 OSHA QA Chatbot")

try:
    # -------------------------------
    # Load embedding model, FAISS index, and QA dataset
    # -------------------------------
    embedding_model, qa_data, faiss_index, qa_embeddings = get_faiss_drive(
        "all-MiniLM-L6-v2",
        "1Xmp6O3DzMqcTev53Y_SwurjxfnNUcocJ",  # faiss_index.bin
        "1Lr7vVUF-9BhvQQ7b-TdGMFLRMMuG8UvQ",  # cleaned CSV
        "1NrTbmHBdn5YLywkuNHl5Rjdcp9LGYo_w"   # embeddings.npy
    )

    logger.info("FAISS index and QA dataset loaded successfully.")

    # -------------------------------
    # Initialize QA bot
    # -------------------------------
    qa_bot = OSHAQA(embedding_model, faiss_index, qa_data)

    # -------------------------------
    # Streamlit user input
    # -------------------------------
    query = st.text_input("Type your question here...")

    if st.button("🔍 Search Answer") and query.strip():
        with st.spinner("Finding best match..."):
            try:
                # ✅ Get best matching QA from FAISS search
                matched_question, matched_answer, distance = qa_bot.get_best_match(query)

                # ✅ Display results
                st.success("✅ Found a Match!")
                st.markdown(f"**Question:** {matched_question}")
                st.markdown(f"**Answer:** {matched_answer}")
                st.caption(f"🔎 Distance Score: {distance:.4f}")

                logger.info(f"Query answered successfully | Query: {query} | Distance: {distance:.4f}")

            except Exception as e:
                logger.error("Error while searching FAISS index", exc_info=True)
                raise PpeVision360Exception(e, sys)

except Exception as e:
    logger.error("Error while loading QA system", exc_info=True)
    raise PpeVision360Exception(e, sys)
