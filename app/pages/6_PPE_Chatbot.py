import os
import sys
import streamlit as st
import pandas as pd
from src.PPE_VISION_360.utils.chat_llm_utils import GroqClient
from src.PPE_VISION_360.chat_engine.chat_llm import PPEChatbotEngine
from src.PPE_VISION_360.chat_engine.bert_classifier import BERTClassifier
from src.PPE_VISION_360.chat_engine.ner_tagger import NERTagger
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception
from src.PPE_VISION_360.chat_engine.model_loader import get_faiss, get_bert, get_ner

# -----------------------------
# Ensure src path is visible
# -----------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
logger.info(f"Project root added to sys.path: {project_root}")

st.title("💬 PPE Assistant Chat (RAG)")

try:
    # -----------------------------
    # Session state init
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
    embedding_model_name = "all-MiniLM-L6-v2"
    faiss_index_path = r"D:\PPE-Vision-360\datasets\nlp\faiss_index.bin"
    data_csv_path = r"D:\PPE-Vision-360\datasets\nlp\osha_qa_cleaned.csv"
    embeddings_path = r"D:\PPE-Vision-360\datasets\nlp\qa_embeddings.npy"

    embedding_model, qa_data, faiss_index, qa_embeddings = get_faiss(
        embedding_model_name, faiss_index_path, data_csv_path, embeddings_path
    )
    doc_texts = (qa_data['clean_question'] + " " + qa_data['clean_answer']).tolist()
    k = 1  # Top-k retrieval
    logger.info("FAISS index, embeddings, and document texts loaded successfully.")

    # -----------------------------
    # Load BERT + tokenizer
    # -----------------------------
    bert_model_path = r"D:\PPE-Vision-360\models\saved_distillbert"
    bert_tokenizer_path = r"D:\PPE-Vision-360\models\saved_distillbert"
    bert_model, bert_tokenizer = get_bert(bert_model_path, bert_tokenizer_path)
    bert_classifier = BERTClassifier(bert_model, bert_tokenizer)
    logger.info("BERTClassifier loaded successfully.")

    # -----------------------------
    # Load NER
    # -----------------------------
    ner_model_path = r"D:\PPE-Vision-360\models\ppe_ner_model"
    ner_model = get_ner(ner_model_path)
    ner_tagger = NERTagger(ner_model)
    logger.info("NERTagger loaded successfully.")

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
        user_input = st.text_area("", height=50, placeholder="Type your message...")
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
        ppe_keywords = ["helmet", "gloves", "boots", "vest", "goggles", "ppe", "safety", "protection"]

        if msg.lower() in greetings:
            reply = "Hello! How can I help you with PPE today?"
            logger.info("Detected greeting message.")
        elif not any(word.lower() in msg.lower() for word in ppe_keywords):
            reply = "Sorry, I can only answer questions related to PPE. Please ask about helmets, gloves, boots, vests, or goggles."
            logger.info("Non-PPE message detected. Sent polite fallback response.")
        else:
            # BERT classification
            bert_label_raw = bert_classifier.classify(msg)
            bert_label = bert_label_raw[0]
            logger.info(f"BERT classified message as: {bert_label}")

            # NER extraction
            ner_items_raw = ner_tagger.detect_entities(msg)
            required_ppe = ["helmet", "gloves", "boots", "vest", "goggles"]
            ner_items = [
                str(item[0]) if isinstance(item, (tuple, list)) else str(item)
                for item in ner_items_raw
                if str(item[0] if isinstance(item, (tuple, list)) else item).lower() in required_ppe
            ]
            logger.info(f"NER extracted PPE items: {ner_items}")

            # Chatbot engine (RAG + reasoning)
            reply = chat_bot.process_message(msg, bert_label=bert_label, ner_items=ner_items)
            logger.info(f"Generated assistant reply: {reply}")

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

        # Auto-scroll
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
