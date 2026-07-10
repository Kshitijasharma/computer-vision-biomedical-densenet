import json
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px


# pyrefly: ignore [missing-import]
from PIL import Image

# pyrefly: ignore [missing-import]
import streamlit as st
import tensorflow as tf

# =====================================================
# Page Config
# =====================================================

st.set_page_config(
    page_title="Blood Cell Classification",
    page_icon="🩸",
    layout="wide"
)

# =====================================================
# Load Model
# =====================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("DenseNet121_best.keras")


model = load_model()

# =====================================================
# Load Class Names
# =====================================================

with open("class_names.json") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)

# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.title("🩸 Model Information")

    st.markdown("""
### Architecture
DenseNet121

### Task
Blood Cell Classification

### Number of Classes
8

### Input Size
224 × 224

### Training Strategy
Transfer Learning

Fine-Tuning

### Framework

TensorFlow

Streamlit
""")

# =====================================================
# Image Preprocessing
# =====================================================

def preprocess_image(image):

    image = image.resize(IMG_SIZE)

    image = np.array(image)

    image = image.astype(np.float32)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image


# =====================================================
# Prediction
# =====================================================

def predict(image):

    img = preprocess_image(image)

    pred = model.predict(img, verbose=0)[0]

    idx = np.argmax(pred)

    label = class_names[idx]

    confidence = pred[idx] * 100

    return label, confidence, pred


# =====================================================
# Main UI
# =====================================================

st.title("🩸 Blood Cell Classification System")

st.write(
    "Upload a microscopy image of a blood cell to classify it using a DenseNet121 model trained with transfer learning."
)

uploaded_file = st.file_uploader(
    "Upload Blood Cell Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    prediction, confidence, probabilities = predict(image)

    left, right = st.columns([1.2, 1])

    # -------------------------------------------------

    with left:

        st.subheader("Uploaded Image")

        st.image(image, use_container_width=True)

        width, height = image.size

        st.info(f"""
Filename : **{uploaded_file.name}**

Original Resolution : **{width} × {height}**

Model Input : **224 × 224**
""")

    # -------------------------------------------------

    with right:

        st.subheader("Prediction")

        st.success(f"### {prediction.upper()}")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.success("Prediction Completed")

        st.subheader("Top-3 Predictions")

        top3 = np.argsort(probabilities)[::-1][:3]

        for rank, idx in enumerate(top3, start=1):

            st.write(
                f"**{rank}. {class_names[idx].title()}** — {probabilities[idx]*100:.2f}%"
            )

        st.subheader("Prediction Probabilities")

        prob_df = pd.DataFrame({
            "Cell Type": class_names,
            "Probability": probabilities
        })

        prob_df = prob_df.sort_values(
            by="Probability",
            ascending=True
        )

        fig = px.bar(
            prob_df,
            x="Probability",
            y="Cell Type",
            orientation="h",
            text="Probability"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        fig.update_layout(
            height=450,
            xaxis_title="Probability",
            yaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

st.markdown("---")

st.warning(
    "This application is intended for research and educational purposes only and should not be used for clinical diagnosis."
)

st.caption(
    "Developed by **Kshitija Sharma** | DenseNet121 | TensorFlow | Streamlit"
)