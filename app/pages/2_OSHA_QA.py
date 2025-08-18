import sys
import os
import streamlit as st

# -------------------------------
# Ensure project root is on path
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from PPE_VISION_360.chat_engine.model_loader import get_faiss
from PPE_VISION_360.chat_engine.osha_qa import OSHAQA
from PPE_VISION_360.logger.logger import logger
from PPE_VISION_360.exception.exception import PpeVision360Exception

# 🚀 Streamlit app title
st.title("💬 OSHA QA Chatbot")

try:
    # -------------------------------
    # Load embedding model, FAISS index, and QA dataset
    # -------------------------------
    embedding_model, qa_data, faiss_index, _ = get_faiss(
        embedding_model_name="all-MiniLM-L6-v2",      # Embedding model name
        faiss_index_path=r"D:\PPE-Vision-360\datasets\nlp\faiss_index.bin",  # Saved FAISS index path
        data_csv_path=r"D:\PPE-Vision-360\datasets\nlp\osha_qa_cleaned.csv", # QA dataset
        embeddings_path=r"D:\PPE-Vision-360\datasets\nlp\qa_embeddings.npy"  # Precomputed embeddings
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
