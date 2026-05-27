"""
AI-Based Mental Health Sentiment Monitoring System
Streamlit Web Application
"""

import streamlit as st
import numpy as np
import pickle
import re
import string
import os
import nltk

# Download NLTK data on first run (needed on Streamlit Cloud)
try:
    from nltk.corpus import stopwords
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Sentiment Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_LEN = 100

EMOTION_META = {
    "Normal": {
        "emoji": "😊",
        "color": "#2ecc71",
        "bg": "#eafaf1",
        "message": "You seem to be doing well. Keep maintaining healthy habits and positive connections.",
        "tips": ["Stay active and exercise regularly", "Maintain social connections", "Practice gratitude daily"],
    },
    "Depression": {
        "emoji": "😔",
        "color": "#3498db",
        "bg": "#ebf5fb",
        "message": "Signs of depression detected. Please consider speaking with a mental health professional.",
        "tips": ["Reach out to a trusted friend or family member", "Contact a counselor or therapist", "Call a helpline: 988 (Suicide & Crisis Lifeline)"],
    },
    "Suicidal": {
        "emoji": "🆘",
        "color": "#e74c3c",
        "bg": "#fdedec",
        "message": "⚠️ Crisis indicators detected. Please seek immediate help.",
        "tips": ["Call 988 (Suicide & Crisis Lifeline) immediately", "Go to your nearest emergency room", "Text HOME to 741741 (Crisis Text Line)"],
    },
    "Anxiety": {
        "emoji": "😰",
        "color": "#f39c12",
        "bg": "#fef9e7",
        "message": "Anxiety patterns detected. Breathing exercises and professional support can help.",
        "tips": ["Try deep breathing: inhale 4s, hold 4s, exhale 4s", "Limit caffeine and screen time", "Consider speaking with a therapist"],
    },
    "Bipolar": {
        "emoji": "🔄",
        "color": "#9b59b6",
        "bg": "#f5eef8",
        "message": "Mood fluctuation patterns detected. A psychiatrist can provide proper evaluation.",
        "tips": ["Track your mood daily", "Maintain a consistent sleep schedule", "Consult a psychiatrist for evaluation"],
    },
    "Stress": {
        "emoji": "😤",
        "color": "#e67e22",
        "bg": "#fdf2e9",
        "message": "High stress levels detected. Taking breaks and self-care can make a big difference.",
        "tips": ["Take short breaks every hour", "Practice mindfulness or meditation", "Prioritize tasks and set boundaries"],
    },
    "Personality disorder": {
        "emoji": "🌀",
        "color": "#1abc9c",
        "bg": "#e8f8f5",
        "message": "Complex emotional patterns detected. Long-term therapy can be very effective.",
        "tips": ["Dialectical Behavior Therapy (DBT) is highly effective", "Build a consistent daily routine", "Work with a mental health specialist"],
    },
}

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load model, tokenizer, and label encoder from saved_models/."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model("saved_models/rnn_mental_health_model.h5")

        with open("saved_models/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

        with open("saved_models/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)

        return model, tokenizer, le, None
    except Exception as e:
        return None, None, None, str(e)


# ── Text preprocessing ────────────────────────────────────────────────────────
try:
    from nltk.corpus import stopwords as _sw
    _STOP_WORDS = set(_sw.words("english"))
except Exception:
    _STOP_WORDS = set()


def preprocess_text(text: str) -> str:
    """Mirror the preprocessing used during training."""
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in _STOP_WORDS]
    return " ".join(tokens)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(text: str, model, tokenizer, le):
    """Return (label, confidence_pct, all_probs_dict)."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = le.classes_[idx]
    confidence = float(probs[idx]) * 100
    all_probs = {cls: float(p) * 100 for cls, p in zip(le.classes_, probs)}
    return label, confidence, all_probs


# ── UI helpers ────────────────────────────────────────────────────────────────
def render_result_card(label: str, confidence: float, all_probs: dict):
    meta = EMOTION_META.get(label, EMOTION_META["Normal"])

    st.markdown(
        f"""
        <div style="background:{meta['bg']};border-left:6px solid {meta['color']};
                    border-radius:10px;padding:20px 24px;margin-bottom:16px;">
            <h2 style="color:{meta['color']};margin:0 0 4px 0;">
                {meta['emoji']} {label}
            </h2>
            <p style="font-size:1.1rem;color:#555;margin:0;">
                Confidence: <strong>{confidence:.1f}%</strong>
            </p>
            <p style="color:#333;margin-top:10px;">{meta['message']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Recommendations
    with st.expander("💡 Recommendations", expanded=(label in ["Suicidal", "Depression"])):
        for tip in meta["tips"]:
            st.markdown(f"- {tip}")

    # Confidence bars for all classes
    st.markdown("#### Confidence Scores")
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_probs:
        m = EMOTION_META.get(cls, {})
        color = m.get("color", "#888")
        st.markdown(
            f"""
            <div style="margin-bottom:6px;">
                <span style="display:inline-block;width:180px;font-size:0.9rem;">{m.get('emoji','')}&nbsp;{cls}</span>
                <div style="display:inline-block;background:#eee;border-radius:4px;width:55%;height:18px;vertical-align:middle;">
                    <div style="background:{color};width:{prob:.1f}%;height:100%;border-radius:4px;"></div>
                </div>
                <span style="margin-left:8px;font-size:0.85rem;color:#555;">{prob:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/brain.png", width=80)
        st.title("🧠 Mental Health\nSentiment Monitor")
        st.markdown("---")
        st.markdown(
            """
            **About this app**  
            An AI-powered tool that analyzes text messages and identifies emotional sentiment patterns to assist mental health monitoring.

            **Sentiment Classes**
            """
        )
        for cls, meta in EMOTION_META.items():
            st.markdown(f"{meta['emoji']} {cls}")

        st.markdown("---")
        st.markdown(
            """
            **⚠️ Disclaimer**  
            This tool is for educational purposes only and is **not** a substitute for professional medical advice.  
            If you are in crisis, please call **988** immediately.
            """
        )

    # Main content
    st.title("🧠 AI-Based Mental Health Sentiment Monitoring System")
    st.markdown(
        "Analyze text messages to identify emotional sentiment patterns and assist with early mental health intervention."
    )

    # Load model
    model, tokenizer, le, error = load_artifacts()

    if error:
        st.error(
            f"⚠️ Could not load model: {error}\n\n"
            "Please run the training notebook first and place the saved artifacts in `saved_models/`."
        )
        st.info(
            "**Steps to get started:**\n"
            "1. Open `notebooks/RNN_Mental_Health_Sentiment.ipynb` in Google Colab\n"
            "2. Upload `Combined Data.csv` to Colab\n"
            "3. Run all cells\n"
            "4. Download `saved_models.zip` and extract into this project folder"
        )
        return

    st.success("✅ Model loaded successfully!")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Analysis", "📋 Batch Analysis", "ℹ️ About the Model"])

    # ── Tab 1: Single Analysis ────────────────────────────────────────────────
    with tab1:
        st.subheader("Analyze a Single Message")

        # Quick examples
        st.markdown("**Try an example:**")
        examples = {
            "😔 Depression": "I feel so hopeless and empty, nothing matters anymore",
            "😰 Anxiety": "I can't stop worrying about everything, my heart is racing",
            "🆘 Suicidal": "Sometimes I think everyone would be better off without me",
            "😊 Normal": "I had a great day today and feeling really positive about life",
            "😤 Stress": "I am completely overwhelmed with deadlines and pressure from all sides",
        }
        cols = st.columns(len(examples))
        selected_example = None
        for col, (name, text) in zip(cols, examples.items()):
            if col.button(name, use_container_width=True):
                selected_example = text

        user_input = st.text_area(
            "Enter your message here:",
            value=selected_example or "",
            height=120,
            placeholder="Type or paste a message to analyze...",
        )

        if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
            if not user_input.strip():
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    label, confidence, all_probs = predict(user_input, model, tokenizer, le)
                render_result_card(label, confidence, all_probs)

    # ── Tab 2: Batch Analysis ─────────────────────────────────────────────────
    with tab2:
        st.subheader("Analyze Multiple Messages")
        st.markdown("Enter one message per line:")

        batch_input = st.text_area(
            "Messages (one per line):",
            height=200,
            placeholder="I feel anxious today\nHaving a great week\nCan't sleep, mind racing...",
        )

        if st.button("🔍 Analyze All", type="primary", use_container_width=True):
            lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
            if not lines:
                st.warning("Please enter at least one message.")
            else:
                results = []
                progress = st.progress(0)
                for i, line in enumerate(lines):
                    label, confidence, _ = predict(line, model, tokenizer, le)
                    meta = EMOTION_META.get(label, {})
                    results.append({
                        "Message": line[:80] + ("..." if len(line) > 80 else ""),
                        "Sentiment": f"{meta.get('emoji','')} {label}",
                        "Confidence": f"{confidence:.1f}%",
                    })
                    progress.progress((i + 1) / len(lines))

                import pandas as pd
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)

                # Summary chart
                import collections
                label_counts = collections.Counter(
                    r["Sentiment"] for r in results
                )
                st.markdown("#### Sentiment Distribution")
                chart_data = pd.DataFrame(
                    {"Count": list(label_counts.values())},
                    index=list(label_counts.keys()),
                )
                st.bar_chart(chart_data)

    # ── Tab 3: About ──────────────────────────────────────────────────────────
    with tab3:
        st.subheader("About the Model")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                **Architecture: Simple RNN**
                - Embedding Layer (vocab: 20,000, dim: 64)
                - SimpleRNN Layer (128 units, tanh)
                - Dropout (0.4)
                - Dense Hidden Layer (64 units, ReLU)
                - Dropout (0.3)
                - Output Layer (7 classes, Softmax)

                **Training Config**
                - Optimizer: Adam (lr=0.001)
                - Loss: Categorical Cross-Entropy
                - Batch Size: 64
                - Max Epochs: 20 (with early stopping)
                """
            )

        with col2:
            st.markdown(
                """
                **Dataset**
                - Source: Kaggle — Sentiment Analysis for Mental Health
                - Total samples: ~53,000
                - Classes: 7 sentiment categories
                - Split: 70% train / 15% val / 15% test

                **Preprocessing Pipeline**
                1. Lowercase conversion
                2. Digit & punctuation removal
                3. Stopword removal (NLTK)
                4. Tokenization & word indexing
                5. Sequence padding (max_len=100)
                """
            )

        st.markdown(
            """
            ---
            **How RNN Works for Sentiment Analysis**

            The RNN processes text sequentially, one word at a time.  
            At each step `t`, it updates its hidden state:

            ```
            h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
            ```

            The final hidden state `h_T` encodes the full sentence context  
            and is passed to the Dense layer for sentiment classification.
            """
        )


if __name__ == "__main__":
    main()
