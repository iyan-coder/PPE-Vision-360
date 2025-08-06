import streamlit as st
import requests
from PIL import Image
import io

# Page Config
st.set_page_config(page_title="PPE Vision 360", layout="centered")

# Title & Intro
st.title("🦺 PPE Vision 360 - Compliance Checker")
st.markdown("Upload an image to check for **PPE compliance (Helmet, Gloves, Vest, Shoes)**")

# File Upload
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display Uploaded Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Convert Image to Bytes for API Call
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Compliance Check Button
    if st.button("🚀 Check Compliance"):
        with st.spinner('🔍 Analyzing Image...'):
            try:
                files = {'file': (uploaded_file.name, img_bytes, 'image/png')}
                response = requests.post("http://127.0.0.1:8000/check_compliance", files=files)

                if response.status_code == 200:
                    result = response.json()

                    st.success("✅ Compliance Check Completed!")

                    # Display Results in Table
                    st.subheader("📊 Item-wise Compliance Result:")
                    for item, status in result['item_results'].items():
                        status_icon = "✅" if "Compliant" in status else "❌"
                        st.write(f"{status_icon} {item}: {status}")

                    # Overall Status
                    overall_icon = "✅" if result['overall_status'] == "Fully Compliant" else "🚨"
                    st.markdown(f"### {overall_icon} **Overall Status:** {result['overall_status']}")

                else:
                    st.error(f"❌ API Request Failed! Status Code: {response.status_code}")
                    st.text(response.text)

            except Exception as e:
                st.error("⚠️ Something went wrong while calling the API.")
                st.text(str(e))

# ---- Future Additions Placeholder ----
# st.sidebar.title("🧑‍💻 Chatbot Assistant (Coming Soon)")
# st.sidebar.info("A smart assistant to help interpret compliance issues, suggest fixes, and recommend actions.")
