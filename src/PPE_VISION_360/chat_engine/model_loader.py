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
import os
import gdown
import zipfile


# -------------------------------------------------------
# Helper to download files from Google Drive and unzip
# -------------------------------------------------------
def download_from_drive(file_id, out_path, unzip=False):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)  # ensure folder exists
        if not os.path.exists(out_path):
            tmp_path = out_path + ".zip" if unzip else out_path
            logger.info(f"Downloading file {file_id} to {tmp_path}...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", tmp_path, quiet=False)

            if unzip:
                logger.info(f"Unzipping {tmp_path} to {out_path}...")
                with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                    zip_ref.extractall(out_path)
                os.remove(tmp_path)

        else:
            logger.info(f"{out_path} already exists. Skipping download.")
        return out_path

    except Exception as e:
        logger.error(f"Failed to download/unzip {file_id}", exc_info=True)
        raise PpeVision360Exception(e, sys)


# -------------------------------------------------------
# ModelLoaderDrive: loads from Drive
# -------------------------------------------------------
class ModelLoader:
    def __init__(self):
        try:
            logger.info("Initializing ModelLoaderDrive...")
            self.bert_model = None
            self.bert_tokenizer = None
            self.ner_model = None
            self.embedding_model = None
            self.faiss_index = None
            self.qa_data = None
            self.qa_embeddings = None
        except Exception as e:
            logger.error("Error initializing ModelLoaderDrive", exc_info=True)
            raise PpeVision360Exception(e, sys)

    # -------------------------------
    # BERT from Drive
    # -------------------------------
    def load_bert(self, model_file_id, tokenizer_file_id):
        try:
            model_path = download_from_drive(model_file_id, "models/saved_distillbert", unzip=True)
            tokenizer_path = download_from_drive(tokenizer_file_id, "models/saved_distillbert", unzip=True)

            self.bert_model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
            self.bert_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            logger.info("BERT model + tokenizer loaded from Drive.")
            return self.bert_model, self.bert_tokenizer
        except Exception as e:
            logger.error("Failed to load BERT", exc_info=True)
            raise PpeVision360Exception(e, sys)

    # -------------------------------
    # NER from Drive
    # -------------------------------
    def load_ner(self, model_file_id):
        try:
            model_path = download_from_drive(model_file_id, "models/ppe_ner_model", unzip=True)
            self.ner_model = spacy.load(model_path)
            logger.info("NER model loaded from Drive.")
            return self.ner_model
        except Exception as e:
            logger.error("Failed to load NER", exc_info=True)
            raise PpeVision360Exception(e, sys)

    # -------------------------------
    # FAISS + embeddings from Drive
    # -------------------------------
    def load_faiss(self, embedding_model_name, faiss_file_id, data_csv_file_id, embeddings_file_id):
        try:
            faiss_index_path = download_from_drive(faiss_file_id, "datasets/nlp/faiss_index.bin")
            data_csv_path = download_from_drive(data_csv_file_id, "datasets/nlp/osha_qa_cleaned.csv")
            embeddings_path = download_from_drive(embeddings_file_id, "datasets/nlp/qa_embeddings.npy")

            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.faiss_index = faiss.read_index(faiss_index_path)
            self.qa_data = pd.read_csv(data_csv_path)
            self.qa_embeddings = np.load(embeddings_path)

            logger.info("FAISS + embeddings loaded from Drive.")
            return self.embedding_model, self.qa_data, self.faiss_index, self.qa_embeddings
        except Exception as e:
            logger.error("Failed to load FAISS resources", exc_info=True)
            raise PpeVision360Exception(e, sys)


# -------------------------------------------------------
# Streamlit cache wrappers
# -------------------------------------------------------
@st.cache_resource()
def get_bert_drive(model_file_id, tokenizer_file_id):
    loader = ModelLoader()
    return loader.load_bert(model_file_id, tokenizer_file_id)

@st.cache_resource()
def get_ner_drive(model_file_id):
    loader = ModelLoader()
    return loader.load_ner(model_file_id)

@st.cache_resource()
def get_faiss_drive(embedding_model_name, faiss_file_id, data_csv_file_id, embeddings_file_id):
    loader = ModelLoader()
    return loader.load_faiss(embedding_model_name, faiss_file_id, data_csv_file_id, embeddings_file_id)
