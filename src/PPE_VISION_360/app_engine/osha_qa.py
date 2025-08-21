import sys
import numpy as np
import streamlit as st
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception

class OSHAQA:
    def __init__(self, model, faiss_index, qa_data):
        try:
            # Store reference to embedding model (e.g., SentenceTransformer)
            self.model = model

            # Store reference to FAISS index for similarity search
            self.faiss_index = faiss_index

            # Store QA dataset (Pandas DataFrame with clean_question + clean_answer)
            self.qa_data = qa_data

            logger.info("OSHAQA class initialized successfully.")

        except Exception as e:
            # Log and raise exception if initialization fails
            logger.error("Error during OSHAQA initialization", exc_info=True)
            raise PpeVision360Exception(e, sys)

    def get_best_match(self, query):
        """
        Given a query string, find the best matching Q&A from FAISS index
        """
        try:
            logger.info(f"Searching best match for query: {query}")

            # Generate embedding for the input query
            query_embedding = self.model.encode([query])

            # Perform FAISS nearest neighbor search (k=1 means return top match)
            D, I = self.faiss_index.search(
                np.array(query_embedding).astype('float32'), 
                k=1
            )

            # Get the index of the best matched question
            matched_idx = I[0][0]

            # Retrieve the corresponding question from the QA dataset
            matched_question = self.qa_data.iloc[matched_idx]["clean_question"]

            # Retrieve the corresponding answer from the QA dataset
            matched_answer = self.qa_data.iloc[matched_idx]["clean_answer"]

            # Get the similarity/distance score (smaller = more similar)
            distance = D[0][0]

            logger.info("Best match found successfully.")
            return matched_question, matched_answer, distance

        except Exception as e:
            # Log error with traceback and raise custom exception
            logger.error("Error during FAISS search in OSHAQA", exc_info=True)
            raise PpeVision360Exception(e, sys)
