import streamlit as st
import requests
from PIL import Image
import io
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

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
st.sidebar.info("AI-driven PPE Compliance Assistant\n\n🔍 Image Detection + 🧑‍💻 Chatbot Support")

# ---- MAIN TABS ----
tab1, tab2 = st.tabs(["🖼️ Image Compliance", "💬 Ask OSHA Bot"])

# ---- TAB 1: IMAGE COMPLIANCE ----
with tab1:
    st.title("🖼️ Image Compliance Checker")
    st.markdown("Upload an image to check for **PPE compliance (Helmet, Gloves, Vest, Shoes)**")

    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        if st.button("🚀 Check Compliance"):
            with st.spinner('🔍 Analyzing Image...'):
                try:
                    files = {'file': (uploaded_file.name, img_bytes, 'image/png')}
                    response = requests.post("http://127.0.0.1:8000/check_compliance", files=files)

                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Compliance Check Completed!")

                        st.subheader("📊 Item-wise Compliance Result:")
                        for item, status in result['item_results'].items():
                            if "Compliant" in status:
                                st.markdown(f"<span style='color:#34A853;font-weight:bold;'>✅ {item}: {status}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span style='color:#EA4335;font-weight:bold;'>❌ {item}: {status}</span>", unsafe_allow_html=True)

                        overall_icon = "✅" if result['overall_status'] == "Fully Compliant" else "🚨"
                        st.markdown(f"### {overall_icon} **Overall Status:** {result['overall_status']}")

                    else:
                        st.error(f"❌ API Request Failed! Status Code: {response.status_code}")
                        st.text(response.text)

                except Exception as e:
                    st.error("⚠️ Something went wrong while calling the API.")
                    st.text(str(e))


# ---- TAB 2: ASK OSHA CHATBOT ----
with tab2:
    st.title("💬 OSHA Compliance Chatbot")
    st.markdown("Ask questions related to **PPE Safety & OSHA Guidelines**")

    user_query = st.text_input("Type your question here...")

    if 'faiss_index' not in st.session_state:
        # Load model (for encoding the user query only)
        st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load CSV data
        st.session_state.qa_data = pd.read_csv(r"D:\PPE-Vision-360\datasets\nlp\osha_qa_cleaned.csv")

        # Load precomputed FAISS index
        st.session_state.faiss_index = faiss.read_index(r"D:\PPE-Vision-360\datasets\nlp\faiss_index.bin")

        # Load precomputed embeddings (optional, only if you need them for debugging or analysis)
        st.session_state.embeddings = np.load(r"D:\PPE-Vision-360\datasets\nlp\qa_embeddings.npy")

    if st.button("🔍 Search Answer") and user_query:
        with st.spinner('Retrieving best match...'):
            # Encode query
            query_embedding = st.session_state.model.encode([user_query])

            # Search FAISS
            D, I = st.session_state.faiss_index.search(
                np.array(query_embedding).astype('float32'), k=1
            )

            matched_idx = I[0][0]
            matched_question = st.session_state.qa_data.iloc[matched_idx]["clean_question"]
            matched_answer = st.session_state.qa_data.iloc[matched_idx]['clean_answer']
            distance = D[0][0]

            # Show results
            st.success("✅ Found a Match!")
            st.markdown(f"**Best Match:** {matched_question}")
            st.markdown(f"**Answer:** {matched_answer}")
            st.caption(f"🔎 Distance Score: {distance:.4f}")
