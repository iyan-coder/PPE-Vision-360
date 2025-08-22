import os
import sys
import streamlit as st
import pandas as pd
import faiss
import numpy as np

# -----------------------------
# Ensure src path is visible
# -----------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.utils.chat_llm_utils import GroqClient
from src.PPE_VISION_360.app_engine.chat_llm import PPEChatbotEngine
from src.PPE_VISION_360.app_engine.bert_classifier import BERTClassifier
from src.PPE_VISION_360.app_engine.ner_tagger import NERTagger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception
from src.PPE_VISION_360.app_engine.model_loader import get_faiss_drive, get_bert_drive, get_ner_drive

st.title("💬 PPE Assistant Chat (RAG)")

try:
    # -----------------------------
    # Initialize session state
    # -----------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        logger.info("Initialized chat_history in session state.")
    if "pending_message" not in st.session_state:
        st.session_state.pending_message = ""
        logger.info("Initialized pending_message in session state.")

    # -----------------------------
    # Load FAISS + embeddings + CSV
    # -----------------------------
    embedding_model, qa_data, faiss_index, qa_embeddings = get_faiss_drive(
        "all-MiniLM-L6-v2",
        "1Xmp6O3DzMqcTev53Y_SwurjxfnNUcocJ",  # faiss_index.bin
        "1Lr7vVUF-9BhvQQ7b-TdGMFLRMMuG8UvQ",  # cleaned CSV
        "1NrTbmHBdn5YLywkuNHl5Rjdcp9LGYo_w"   # embeddings.npy
    )

    doc_texts = (qa_data['clean_question'] + " " + qa_data['clean_answer']).tolist()
    k = 1  # Top-k retrieval
    logger.info("FAISS index, embeddings, and document texts loaded successfully.")

    # -----------------------------
    # Load BERT + tokenizer
    # -----------------------------
    bert_model, bert_tokenizer = get_bert_drive(
        "1v5024dYPwsYmoA4UC97_mHt0x3rdaASH",
        "1v5024dYPwsYmoA4UC97_mHt0x3rdaASH"
    )
    bert_classifier = BERTClassifier(bert_model, bert_tokenizer)
    logger.info("BERTClassifier loaded successfully.")

    # -----------------------------
    # Load NER model
    # -----------------------------
    ner_model = get_ner_drive("1OrHQb7f03nvUA7hUO_zulg3BClWP3WVW")
    ner_tagger = NERTagger(ner_model)
    logger.info("NERTagger loaded successfully.")

    # -----------------------------
    # Load SentenceTransformer ONCE using Streamlit cache
    # -----------------------------
    from sentence_transformers import SentenceTransformer

    @st.cache_resource
    def load_sentence_transformer():
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("SentenceTransformer model loaded once via cache_resource.")
        return model

    faiss_model = load_sentence_transformer()

    # -----------------------------
    # Initialize Groq + chatbot engine
    # -----------------------------
    groq_client = GroqClient()
    chat_bot = PPEChatbotEngine(
        groq_client=groq_client,
        faiss_index=faiss_index,
        doc_texts=doc_texts,
        k=k
    )
    logger.info("PPEChatbotEngine initialized successfully.")

    # -----------------------------
    # Chat container
    # -----------------------------
    chat_container = st.container()

    # -----------------------------
    # Input form
    # -----------------------------
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Ask PPE-related questions only", 
            height=50, 
            placeholder="Type your message..."
        )
        submit = st.form_submit_button("Send")
        if submit and user_input.strip():
            st.session_state.pending_message = user_input.strip()
            logger.info(f"User submitted message: {user_input.strip()}")

    # -----------------------------
    # Process pending message
    # -----------------------------
    if st.session_state.pending_message:
        msg = st.session_state.pending_message
        st.session_state.chat_history.append({"role": "user", "content": msg})
        logger.info(f"Appended user message to chat_history: {msg}")

        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        irrelevant_keywords = ["weather", "sports", "food", "music", "movie"]

        # 1️⃣ Check for greetings
        if msg.lower() in greetings:
            reply = "Hello! How can I help you with PPE today?"
            logger.info("Detected greeting message.")

        # 2️⃣ Check for clearly irrelevant questions
        elif any(word in msg.lower() for word in irrelevant_keywords):
            reply = "Sorry, I can only answer questions related to PPE."
            logger.info("Detected irrelevant message. Sent polite fallback.")

        # 3️⃣ Otherwise, treat as PPE-related and use FAISS similarity
        else:
            query_embedding = faiss_model.encode([msg])
            D, I = faiss_index.search(query_embedding, k)

            similarity_threshold = 0.3  # Tune this threshold
            if D[0][0] < similarity_threshold:
                # Low similarity → fallback
                reply = "Sorry, I am not sure about that. Can you ask a PPE-specific question?"
                logger.info(f"Low FAISS similarity ({D[0][0]}). Sent fallback message.")
            else:
                # --- BERT classification ---
                bert_label_raw = bert_classifier.classify(msg)
                bert_label = bert_label_raw[0]
                logger.info(f"BERT classified message as: {bert_label}")

                # --- NER extraction ---
                ner_items_raw = ner_tagger.detect_entities(msg)
                required_ppe = ["helmet", "gloves", "boots", "vest"]
                ner_items = [
                    str(item[0]) if isinstance(item, (tuple, list)) else str(item)
                    for item in ner_items_raw
                    if str(item[0] if isinstance(item, (tuple, list)) else item).lower() in required_ppe
                ]
                logger.info(f"NER extracted PPE items: {ner_items}")

                # --- Chatbot engine reasoning ---
                reply = chat_bot.process_message(msg, bert_label=bert_label, ner_items=ner_items)
                logger.info(f"Generated assistant reply: {reply}")

        # Append assistant reply
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.session_state.pending_message = ""
        logger.info("Appended assistant reply to chat_history.")

    # -----------------------------
    # Render chat history
    # -----------------------------
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f"""
                <div style='text-align:right; margin-bottom:8px;'>
                    <div style='display:inline-block; background-color:#0f9d58; color:white; padding:8px 12px; border-radius:20px; max-width:80%;'>
                        {chat['content']}
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='text-align:left; margin-bottom:8px;'>
                    <div style='display:flex; align-items:flex-start; max-width:80%;'>
                        <img src='https://i.imgur.com/6VBx3io.png' width='35' style='margin-right:8px; border-radius:50%;'/>
                        <div style='background-color:#1E1E1E; color:#E0E0E0; padding:8px 12px; border-radius:20px;'>{chat['content']}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

        # Auto-scroll to bottom
        st.markdown("<div id='bottom'></div>", unsafe_allow_html=True)
        st.markdown("<script>var element = document.getElementById('bottom'); element.scrollIntoView();</script>", unsafe_allow_html=True)
        logger.info("Rendered chat history with auto-scroll.")

    # -----------------------------
    # Download chat history
    # -----------------------------
    if st.session_state.chat_history:
        chat_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history])
        st.download_button(
            label="Download Chat History",
            data=chat_text,
            file_name="ppe_chat_history.txt"
        )
        logger.info("Provided download button for chat history.")

except Exception as e:
    logger.error("Error in PPE Assistant Chat module", exc_info=True)
    raise PpeVision360Exception(e, sys)
