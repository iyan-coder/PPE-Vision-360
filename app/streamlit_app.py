import streamlit as st
import requests
from PIL import Image
import io
import spacy
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

# ---- PAGE CONFIG ----
st.set_page_config(page_title="PPE Vision 360", layout="centered")

# ---- CUSTOM CSS (Dark Mode) ----
st.markdown("""
    <style>
    .main {
        background-color: #121212;
        color: #E0E0E0;
    }
    .stButton > button {
        background-color: #0f9d58;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stSpinner {
        color: #0f9d58;
    }
    .uploadedFile {
        border: 2px dashed #888;
        padding: 10px;
        background-color: #1E1E1E;
    }
    </style>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
st.sidebar.title("🦺 PPE Vision 360")
st.sidebar.info("AI-driven PPE Compliance Assistant\n\n🔍 Image Detection + 🧑‍💻 Chatbot Support + 🧠 BERT Text Classifier + 🧾 PPE NER Tagger" )

# ---- Load BERT model and tokenizer (once, cached for efficiency) ----
@st.cache_resource
def load_bert_model():
    model = TFAutoModelForSequenceClassification.from_pretrained(r"D:\PPE-Vision-360\models\saved_distillbert")
    tokenizer = AutoTokenizer.from_pretrained(r"D:\PPE-Vision-360\models\saved_distillbert")
    return model, tokenizer

bert_model, bert_tokenizer = load_bert_model()

# ---- MAIN TABS ----
tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image Compliance", "💬 Ask OSHA Bot", "🧠 BERT Classifier", "🧾 PPE NER Tagger"])

# ----------------- TAB 1: IMAGE COMPLIANCE -----------------
with tab1:
    st.title("🖼️ Image Compliance Checker")
    st.markdown("Upload an image to check for **PPE compliance (Helmet, Gloves, Vest, Shoes)**")
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="img_upload")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        if st.button("🚀 Check Compliance", key="btn_image"):
            with st.spinner('🔍 Analyzing Image...'):
                try:
                    files = {'file': (uploaded_file.name, img_bytes, 'image/png')}
                    response = requests.post("http://127.0.0.1:8000/check_compliance", files=files)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Compliance Check Completed!")

                        st.subheader("📊 Item-wise Compliance Result:")
                        for item, status in result['item_results'].items():
                            color = "#34A853" if "Compliant" in status else "#EA4335"
                            icon = "✅" if "Compliant" in status else "❌"
                            st.markdown(f"<span style='color:{color};font-weight:bold;'>{icon} {item}: {status}</span>", unsafe_allow_html=True)

                        overall_icon = "✅" if result['overall_status'] == "Fully Compliant" else "🚨"
                        st.markdown(f"### {overall_icon} **Overall Status:** {result['overall_status']}")
                    else:
                        st.error(f"❌ API Request Failed! Status Code: {response.status_code}")
                        st.text(response.text)
                except Exception as e:
                    st.error("⚠️ Something went wrong while calling the API.")
                    st.text(str(e))

# ----------------- TAB 2: OSHA CHATBOT -----------------
with tab2:
    st.title("💬 OSHA Compliance Chatbot")
    st.markdown("Ask questions related to **PPE Safety & OSHA Guidelines**")
    user_query = st.text_input("Type your question here...", key="qa_input")

    # Load embeddings, FAISS index, CSV data once
    if 'faiss_index' not in st.session_state:
        st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
        st.session_state.qa_data = pd.read_csv(r"D:\PPE-Vision-360\datasets\nlp\osha_qa_cleaned.csv")
        st.session_state.faiss_index = faiss.read_index(r"D:\PPE-Vision-360\datasets\nlp\faiss_index.bin")
        st.session_state.embeddings = np.load(r"D:\PPE-Vision-360\datasets\nlp\qa_embeddings.npy")

    if st.button("🔍 Search Answer", key="btn_qa") and user_query.strip():
        with st.spinner('Retrieving best match...'):
            query_embedding = st.session_state.model.encode([user_query])
            D, I = st.session_state.faiss_index.search(np.array(query_embedding).astype('float32'), k=1)
            matched_idx = I[0][0]
            matched_question = st.session_state.qa_data.iloc[matched_idx]["clean_question"]
            matched_answer = st.session_state.qa_data.iloc[matched_idx]['clean_answer']
            distance = D[0][0]

            st.success("✅ Found a Match!")
            st.markdown(f"**Best Match:** {matched_question}")
            st.markdown(f"**Answer:** {matched_answer}")
            st.caption(f"🔎 Distance Score: {distance:.4f}")

# ----------------- TAB 3: BERT CLASSIFIER -----------------
with tab3:
    st.title("🧠 BERT Text Classifier")
    st.markdown("Enter text to classify using the trained BERT model.")
    user_text_bert = st.text_area("Enter your text here...", key="bert_input")

    if st.button("Classify Text", key="btn_bert") and user_text_bert.strip():
        with st.spinner("Classifying..."):
            inputs = bert_tokenizer(
                user_text_bert,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="tf"
            )
            outputs = bert_model(**inputs)
            logits = outputs.logits
            probs = tf.nn.softmax(logits, axis=-1)
            pred_class_idx = tf.argmax(probs, axis=1).numpy()[0]
            confidence = probs[0, pred_class_idx].numpy()

            class_names = ["Emergency_Response","Hazard_Reporting","PPE_Compliance", "PPE_NonCompliance", "Safety_Procedure"]
            predicted_label = class_names[pred_class_idx]

            st.success(f"Prediction: **{predicted_label}**")
            st.write(f"Confidence: {confidence:.2%}")

# ----------------- TAB 4: PPE NER -----------------
with tab4:
    st.title("🧾 PPE NER Tagger")
    st.markdown("Enter text to detect PPE items using the trained spaCy NER model.")

    @st.cache_resource
    def load_ner_model():
        # Load the trained NER model from local path
        return spacy.load(r"D:\PPE-Vision-360\models\ppe_ner_model")

    nlp_ner = load_ner_model()
    user_text_ner = st.text_area("Enter your text here...", key="ner_input")

    if st.button("🔍 Detect PPE Items", key="btn_ner") and user_text_ner.strip():
        with st.spinner("Running NER..."):
            doc = nlp_ner(user_text_ner)
            if doc.ents:
                st.success("✅ Entities Detected:")
                for ent in doc.ents:
                    st.markdown(f"- **{ent.text}** → {ent.label_}")
            else:
                st.info("No PPE items detected in the text.")

