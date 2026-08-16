import streamlit as st
from PIL import Image
from predict import predict_image

st.set_page_config(
    page_title="Plant Health AI - Potato Disease Detector",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Health AI — Potato Disease Detector")
st.markdown("""
Welcome to the **Plant Disease Classification System**. 
Upload a potato leaf image below to diagnose potential diseases (**Early Blight**, **Late Blight**, or **Healthy**) using MobileNetV2 Deep Transfer Learning.
""")

st.sidebar.header("About the System")
st.sidebar.info("""
- **Model**: MobileNetV2 (Transfer Learning)
- **Target Plant**: Potato
- **Accuracy**: ~97% Validation Accuracy
- **Supported Classes**:
  - 🥔 Potato Early Blight
  - 🥔 Potato Late Blight
  - 🟢 Potato Healthy
""")

uploaded_file = st.file_uploader("Upload a potato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
    
    with st.spinner("Analyzing leaf patterns..."):
        predicted_class, confidence = predict_image(image)
        
    st.divider()
    
    display_name = predicted_class.replace("Potato___", "").replace("_", " ").title()
    CONFIDENCE_THRESHOLD = 70.0
    
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(f"⚠️ **Low Confidence Warning ({confidence:.2f}%)**\n\nThe uploaded image could not be reliably diagnosed. Please make sure to upload a clear photo of a **Potato leaf**.")
    elif "healthy" in predicted_class.lower():
        st.success(f"### Diagnosis: **{display_name}** 🟢")
        st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
        st.progress(min(int(confidence), 100))
    else:
        st.error(f"### Diagnosis: **{display_name}** ⚠️")
        st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
        st.progress(min(int(confidence), 100))

st.divider()
st.caption("Built with TensorFlow & Streamlit | Plant Health AI")
