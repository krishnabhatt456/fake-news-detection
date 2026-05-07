import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np
import cv2
import re

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Fake News & Deepfake Detector",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI-Based Fake News & Deepfake Detector")
st.markdown(
    "Detect fake news text and suspicious manipulated images using AI."
)

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():

    # Model 1
    classifier1 = pipeline(
        "text-classification",
        model="mrm8488/bert-tiny-finetuned-fake-news-detection"
    )

    # Model 2
    classifier2 = pipeline(
        "text-classification",
        model="hamzab/roberta-fake-news-classification"
    )

    return classifier1, classifier2


classifier1, classifier2 = load_models()

# ================= TEXT CLEANING =================
def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ================= TEXT ANALYSIS =================
def analyze_text(text):

    cleaned_text = clean_text(text)

    # Short text handling
    if len(cleaned_text.split()) < 5:
        return (
            "⚠️ Please enter a longer news article or headline."
        )

    try:

        # MODEL 1
        result1 = classifier1(cleaned_text)[0]

        # MODEL 2
        result2 = classifier2(cleaned_text)[0]

        # Scores
        score1 = round(result1["score"] * 100, 2)
        score2 = round(result2["score"] * 100, 2)

        fake_votes = 0
        real_votes = 0

        # ================= MODEL 1 LOGIC =================
        label1 = result1["label"].lower()

        if (
            "fake" in label1
            or "label_1" in label1
            or "negative" in label1
        ):
            fake_votes += 1
        else:
            real_votes += 1

        # ================= MODEL 2 LOGIC =================
        label2 = result2["label"].lower()

        if (
            "fake" in label2
            or "label_1" in label2
            or "negative" in label2
        ):
            fake_votes += 1
        else:
            real_votes += 1

        # Average confidence
        avg_score = round((score1 + score2) / 2, 2)

        # ================= FINAL DECISION =================
        if avg_score < 70:
            return (
                f"⚠️ Low Confidence Result\n\n"
                f"Confidence: {avg_score}%"
            )

        if fake_votes > real_votes:
            return (
                f"🚨 Potentially Fake News\n\n"
                f"Confidence: {avg_score}%\n\n"
                f"Model Votes: {fake_votes} Fake / "
                f"{real_votes} Real"
            )

        return (
            f"✅ Likely Real News\n\n"
            f"Confidence: {avg_score}%\n\n"
            f"Model Votes: {real_votes} Real / "
            f"{fake_votes} Fake"
        )

    except Exception as e:
        return f"Error analyzing text: {e}"


# ================= IMAGE ANALYSIS =================
def analyze_image(image):

    try:

        img = np.array(image)

        # Convert RGB → BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Resize
        img = cv2.resize(img, (256, 256))

        # Gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge Detection
        edges = cv2.Canny(gray, 100, 200)

        # Edge Score
        edge_mean = np.mean(edges)

        # Blur Score
        blur_score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        # Noise analysis
        noise = np.std(gray)

        # ================= DECISION LOGIC =================
        suspicious_score = 0

        if edge_mean > 45:
            suspicious_score += 1

        if blur_score < 120:
            suspicious_score += 1

        if noise < 35:
            suspicious_score += 1

        # ================= FINAL RESULT =================
        if suspicious_score >= 2:

            return (
                "⚠️ Image may be AI-generated or manipulated\n\n"
                f"Edge Score: {round(edge_mean, 2)}\n"
                f"Blur Score: {round(blur_score, 2)}\n"
                f"Noise Score: {round(noise, 2)}"
            )

        return (
            "✅ Image appears relatively normal\n\n"
            f"Edge Score: {round(edge_mean, 2)}\n"
            f"Blur Score: {round(blur_score, 2)}\n"
            f"Noise Score: {round(noise, 2)}"
        )

    except Exception as e:
        return f"Error analyzing image: {e}"


# ================= SIDEBAR =================
st.sidebar.header("📌 Project Information")

st.sidebar.info(
    """
This AI system analyzes:

• Fake News Text  
• AI-Generated Images  
• Manipulated Images  

Built using:
- Streamlit
- Transformers
- OpenCV
- Python
"""
)

# ================= TEXT SECTION =================
st.subheader("📰 Fake News Detection")

text = st.text_area(
    "Enter News Text",
    height=180,
    placeholder="Paste news article or headline here..."
)

# ================= IMAGE SECTION =================
st.subheader("🖼️ Deepfake Image Detection")

uploaded_image = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ================= ANALYZE BUTTON =================
if st.button("🔍 Analyze Content"):

    # ---------- TEXT ----------
    if text:

        with st.spinner("Analyzing news text..."):

            text_result = analyze_text(text)

            if "Fake" in text_result:
                st.error(text_result)

            elif "Low Confidence" in text_result:
                st.warning(text_result)

            else:
                st.success(text_result)

    # ---------- IMAGE ----------
    if uploaded_image:

        with st.spinner("Analyzing image..."):

            image = Image.open(
                uploaded_image
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            image_result = analyze_image(image)

            if "manipulated" in image_result:
                st.warning(image_result)

            else:
                st.success(image_result)

    # ---------- EMPTY INPUT ----------
    if not text and not uploaded_image:
        st.error(
            "Please enter text or upload an image."
        )

# ================= FOOTER =================
st.markdown("---")

st.caption(
    "Built using Streamlit, Transformers, OpenCV, and Python"
)