import streamlit as st
from PIL import Image
from predict import predict_image

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Plant Health AI - Potato Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# ---------------------------------------------------------
# Custom CSS — background, fonts, cards, highlighted name
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Merriweather:wght@700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* App background — soft green gradient to match a plant theme */
.stApp {
    background: linear-gradient(160deg, #eafaf1 0%, #d4f1e0 40%, #eafaf1 100%);
}

/* Title styling */
h1 {
    color: #1b4332;
    font-weight: 700;
    text-align: center;
    padding-bottom: 0.2rem;
}

/* Intro / markdown text block */
.intro-box {
    background-color: #ffffff;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
    font-size: 1.05rem;
    color: #2d3436;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1b4332;
}
section[data-testid="stSidebar"] * {
    color: #eafaf1 !important;
}

/* File uploader box */
.stFileUploader {
    background-color: #ffffff;
    border: 2px dashed #52b788;
    border-radius: 12px;
    padding: 1rem;
}

/* Result card */
.result-card {
    background-color: #ffffff;
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1.5rem;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.10);
}
.result-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1b4332;
}
.result-confidence {
    font-size: 1.05rem;
    color: #40916c;
    margin-top: 0.3rem;
}

/* Footer with highlighted developer name */
.footer {
    text-align: center;
    margin-top: 3rem;
    padding: 1rem;
    font-size: 0.95rem;
    color: #2d3436;
}
.footer .name {
    font-family: 'Merriweather', serif;
    font-weight: 700;
    font-size: 1.15rem;
    background: linear-gradient(90deg, #2d6a4f, #52b788);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🌿 Plant Health AI — Potato Disease Detector")

st.markdown("""
<div class="intro-box">
Welcome to the <b>Plant Disease Classification System</b>.<br>
Upload a potato leaf image below to diagnose potential diseases
(<b>Early Blight</b>, <b>Late Blight</b>, or <b>Healthy</b>) using
<b>MobileNetV2 Deep Transfer Learning</b>.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("🌱 About the System")
st.sidebar.info("""
- **Model**: MobileNetV2 (Transfer Learning)
- **Target Plant**: Potato
- **Accuracy**: ~97% Validation Accuracy
- **Supported Classes**:
  - 🥔 Potato Early Blight
  - 🥔 Potato Late Blight
  - 🟢 Potato Healthy
""")

# ---------------------------------------------------------
# Upload + Prediction
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload a potato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    with st.spinner("🔍 Analyzing leaf patterns..."):
        predicted_class, confidence = predict_image(image)

    display_name = predicted_class.replace("Potato___", "").replace("_", " ").title()
    CONFIDENCE_THRESHOLD = 70.0

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(f"⚠️ **Low Confidence Warning ({confidence:.2f}%)**\n\nThe uploaded image could not be reliably diagnosed. Please make sure to upload a clear photo of a **Potato leaf**.")
    else:
        if "healthy" in predicted_class.lower():
            icon = "✅"
        else:
            icon = "⚠️"

        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">{icon} {display_name}</div>
            <div class="result-confidence">Confidence: {confidence:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("👆 Please upload a potato leaf image to begin diagnosis.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div class="footer">
    Developed by <span class="name">HABIB UR RAHMAN</span>
</div>
""", unsafe_allow_html=True)
