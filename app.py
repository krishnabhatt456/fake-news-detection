import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np
import cv2

st.title("🧠 Fake News & Deepfake Detector")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
    
                    
classifier = load_model()

# TEXT
st.subheader("Fake News Detection")
text = st.text_area("Enter News Text")

def analyze_text(text):
    result = classifier(text)[0]
    score = round(result['score'] * 100, 2)

    if score < 70:
        return f"⚠️ Uncertain result (Confidence: {score}%)"

    if result['label'] == 'LABEL_1':
        return f"🚨 Fake News (Confidence: {score}%)"
    else:
        return f"✅ Real News (Confidence: {score}%)"

# IMAGE
st.subheader("Deepfake Detection")
uploaded_image = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

def analyze_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200)
    if np.mean(edges) > 50:
        return "⚠️ Possibly Manipulated"
    else:
        return "✅ Likely Real"

# BUTTON
if st.button("Analyze"):
    if text:
        st.success(analyze_text(text))
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image)
        st.warning(analyze_image(image))
