import sys
import spacy
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception
import streamlit as st


# ------------------------------------------------------
# ModelLoader class: handles loading all ML models
# (BERT, NER, FAISS, Sentence Embeddings)
# ------------------------------------------------------
class ModelLoader:
    def __init__(self):
        try:
            logger.info("Initializing ModelLoader class...")

            # Placeholders for models / resources
            self.bert_model = None
            self.bert_tokenizer = None
            self.ner_model = None
            self.embedding_model = None
            self.faiss_index = None
            self.qa_data = None
            self.qa_embeddings = None

            logger.info("ModelLoader initialized successfully.")

        except Exception as e:
            logger.error("Error during ModelLoader initialization", exc_info=True)
            raise PpeVision360Exception(e, sys)

    # -------------------------------
    # Load BERT model + tokenizer
    # -------------------------------
    def load_bert(self, model_path, tokenizer_path):
        try:
            logger.info("Loading BERT model and tokenizer...")
            self.bert_model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
            self.bert_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info("BERT model and tokenizer loaded successfully.")
            return self.bert_model, self.bert_tokenizer
        except Exception as e:
            logger.error("Error loading BERT models/tokenizer", exc_info=True)
            raise PpeVision360Exception(e, sys)

    # -------------------------------
    # Load spaCy NER model
    # -------------------------------
    def load_ner(self, model_path):
        try:
            logger.info("Loading NER model...")
            self.ner_model = spacy.load(model_path)
            logger.info("NER model loaded successfully.")
            return self.ner_model
        except Exception as e:
            logger.error("Error loading NER model", exc_info=True)
            raise PpeVision360Exception(e, sys)

    # -------------------------------
    # Load FAISS + embeddings + data
    # -------------------------------
    def load_faiss(self, embedding_model_name, faiss_index_path, data_csv_path, embeddings_path):
        try:
            logger.info("Loading FAISS index and embeddings...")
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.qa_data = pd.read_csv(data_csv_path)
            self.faiss_index = faiss.read_index(faiss_index_path)
            self.qa_embeddings = np.load(embeddings_path)
            logger.info("FAISS index and embeddings loaded successfully.")
            return self.embedding_model, self.qa_data, self.faiss_index, self.qa_embeddings
        except Exception as e:
            logger.error("Error loading FAISS/embeddings", exc_info=True)
            raise PpeVision360Exception(e, sys)


# ------------------------------------------------------
# Cached wrapper functions
# These prevent Streamlit from trying to hash `self`
# ------------------------------------------------------

@st.cache_resource
def get_bert(model_path, tokenizer_path):
    loader = ModelLoader()
    return loader.load_bert(model_path, tokenizer_path)

@st.cache_resource
def get_ner(model_path):
    loader = ModelLoader()
    return loader.load_ner(model_path)

@st.cache_resource
def get_faiss(embedding_model_name, faiss_index_path, data_csv_path, embeddings_path):
    loader = ModelLoader()
    return loader.load_faiss(embedding_model_name, faiss_index_path, data_csv_path, embeddings_path)
