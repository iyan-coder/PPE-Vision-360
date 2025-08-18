import streamlit as st

st.set_page_config(
    page_title="PPE Vision 360",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ----
st.markdown("""
<style>
.main { background-color: #121212; color: #E0E0E0; }
.stButton > button { 
    background-color: #0f9d58; 
    color: white; 
    border-radius: 10px; 
    padding: 10px 20px; 
    font-size: 16px; 
}
</style>
""", unsafe_allow_html=True)

# ---- Sidebar Navigation ----
st.sidebar.title("🦺 PPE Vision 360")
st.sidebar.info(
    "Welcome to PPE Vision 360!\n\n"
    "Use the pages on the left to navigate:\n"
    "- 1️⃣ 🖼️ Image Compliance\n"
    "- 2️⃣ 💬 OSHA QA Chatbot\n"
    "- 3️⃣ 🧠 BERT Classifier\n"
    "- 4️⃣ 🧾 PPE NER Tagger\n"
    "- 5️⃣ 📝 PPE Compliance Reasoning\n"
    "- 6️⃣ 🤖 PPE Chat Assistant"
)

# ---- Home Page ----
st.title("🦺 PPE Vision 360 Dashboard")
st.markdown(
    """
    This is your AI-driven PPE Compliance Assistant platform. 
    
    **Features:**  
    1. Image Compliance Checker  
    2. OSHA QA Chatbot  
    3. BERT Text Classifier  
    4. PPE NER Tagger  
    5. PPE Compliance Reasoning 
    6. Interactive PPE Chat Assistant  

    Use the sidebar to navigate to the feature you want to use.
    """
)

# ---- Logo ----
st.image(r"D:\PPE-Vision-360\logo\PPE Assistant Logo Design.png", width=250)
