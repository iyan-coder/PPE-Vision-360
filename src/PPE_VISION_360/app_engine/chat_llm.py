import sys
from src.PPE_VISION_360.logger.logger import logger
from src.PPE_VISION_360.exception.exception import PpeVision360Exception
from src.PPE_VISION_360.app_engine.compliance_reasoning import generate_compliance_reasoning
import numpy as np
from sentence_transformers import SentenceTransformer


class PPEChatbotEngine:
    def __init__(self, groq_client, faiss_index=None, doc_texts=None, k=1):
        """
        Initialize the PPE chat assistant with:
        - groq_client: LLM handler (Groq)
        - faiss_index: FAISS index for retrieval (optional)
        - doc_texts: corresponding document texts for retrieval
        - k: top-k retrieved documents
        """
        self.client = groq_client
        self.faiss_index = faiss_index
        self.doc_texts = doc_texts
        self.k = k
        logger.info("PPEChatbotEngine initialized. FAISS retrieval: %s", "enabled" if faiss_index else "disabled")

    def process_message(self, message: str, bert_label=None, ner_items=None) -> str:
        """
        Process the user's input message.
        Steps:
        1. Check greetings
        2. Optionally run Phase 7 reasoning
        3. Optionally retrieve related docs from FAISS
        4. Query LLM with reasoning + retrieved docs
        """
        try:
            greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
            logger.info(f"Received user message: {message}")

            # 1️⃣ Greetings
            if message.lower() in greetings:
                logger.info("User input detected as greeting. Returning predefined response.")
                return "Hello! How can I help you with PPE today?"

            # 2️⃣ Phase 7 reasoning (BERT + NER)
            reasoning_text = ""
            if bert_label and ner_items is not None:
                reasoning_text = generate_compliance_reasoning(bert_label, ner_items)
                logger.info(f"Generated Phase 7 reasoning: {reasoning_text}")

            # 3️⃣ Retrieval from FAISS (if enabled)
            retrieved_texts = ""
            if self.faiss_index and self.doc_texts:
               
                embedder = SentenceTransformer("all-MiniLM-L6-v2")  # your embedding model
                query_vec = embedder.encode([message]).astype(np.float32)
                D, I = self.faiss_index.search(query_vec, self.k)
                retrieved_texts = "\n".join([self.doc_texts[i] for i in I[0]])
                logger.info(f"Top-{self.k} retrieved docs: {retrieved_texts}")

            # 4️⃣ Construct prompt for LLM
            system_prompt = (
                "You are a PPE compliance assistant. Answer naturally and clearly. "
                "Only provide advice for helmet, vest, boots, or gloves. "
                "Include retrieved references if any. Never mention seeing or observing the user."
            )
            user_prompt = message
            if reasoning_text or retrieved_texts:
                user_prompt = f"{reasoning_text}\n\nRetrieved info:\n{retrieved_texts}\n\nUser question:\n{message}"

            # 5️⃣ Query Groq LLM
            reply = self.client.query([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            logger.info(f"LLM response: {reply}")
            return reply

        except Exception as e:
            logger.error("Error during PPEChat message processing", exc_info=True)
            raise PpeVision360Exception(e, sys)
