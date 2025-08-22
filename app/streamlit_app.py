import streamlit as st

st.set_page_config(page_title="SafeSight-AI-Assistant(PPE-Vision-360)", layout="wide")

# ----------------- STYLES -----------------
st.markdown("""
    <style>
        .card {
            background-color: #1e1e1e;
            border-radius: 16px;
            padding: 24px;
            margin: 10px;
            text-align: center;
            color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transition: 0.3s;
        }
        .card:hover {
            transform: scale(1.05);
            background-color: #0f9d58;
            cursor: pointer;
        }
        .card h3 {
            font-size: 20px;
            margin-bottom: 12px;
        }
        .card p {
            font-size: 14px;
            color: #ddd;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- HEADER -----------------
st.title("🦺 SafeSight-AI-Assistant(PPE-Vision-360)")
st.write("All-in-One PPE Compliance Platform")

# ----------------- MODULES -----------------
modules = [
    ("🖼️ PPE Image Compliance", "YOLOv8 detection, segmentation, classification", "pages/1_PPE_Image_Compliance.py"),
    ("💬 PPE OSHA QA Chatbot", "Retrieval-based Q&A with FAISS", "pages/2_PPE_OSHA_QA_Chatbot.py"),
    ("🧠 PPE BERT Classifier", "Classify PPE-related text", "pages/3_PPE_BERT_Classifier.py"),
    ("🧾 PPE NER Tagger", "Extract PPE-related entities", "pages/4_PPE_NER_Tagger.py"),
    ("📑 PPE Compliance Report Generator", "Generate structured compliance reports", "pages/5_PPE_Compliance_Report_Generator.py"),
    ("🤖 PPE Chat Assistant", "Conversational multi-turn assistant", "pages/6_PPE_Chat_Assistant.py"),
    ("🎯 PPE Recommendation System", "Suggest improvements/actions for compliance", "pages/7_PPE_Recommendation_System.py")
]

# ----------------- GRID LAYOUT -----------------
cols = st.columns(3)  # 3 cards per row

for i, (title, desc, page) in enumerate(modules):
    with cols[i % 3]:
        if st.button(f"{title}", key=i):
            st.switch_page(page)   # 👈 jumps to the actual module page
        st.markdown(f"""
            <div class="card">
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
        """, unsafe_allow_html=True)
