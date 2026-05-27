# 🧠 AI-Based Mental Health Sentiment Monitoring System

An AI-powered web application that analyzes user text messages and identifies emotional sentiment patterns using a **Simple RNN** model built with TensorFlow/Keras.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 📌 Problem Statement

A mental wellness startup wants to monitor emotional well-being, identify negative sentiment trends, and assist counselors with early intervention by analyzing user text messages.

---

## 🎯 Features

- **7-class sentiment classification**: Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality Disorder
- **Confidence scores** for every prediction
- **Emotional status feedback** with actionable recommendations
- **Batch analysis** — analyze multiple messages at once
- **Crisis alerts** for high-risk sentiments (Suicidal, Depression)

---

## 🗂️ Project Structure

```
RNN_SentimentAnalysis/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── packages.txt                    # System packages for Streamlit Cloud
├── setup_nltk.py                   # NLTK data downloader
├── .streamlit/
│   └── config.toml                 # Streamlit theme config
├── notebooks/
│   └── RNN_Mental_Health_Sentiment.ipynb   # Full training notebook (Colab)
├── saved_models/                   # Generated after training
│   ├── rnn_mental_health_model.h5
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   └── config.pkl
└── Combined Data.csv               # Dataset (from Kaggle)
```

---

## 🚀 Quick Start

### Step 1 — Train the Model (Google Colab)

1. Open `notebooks/RNN_Mental_Health_Sentiment.ipynb` in [Google Colab](https://colab.research.google.com)
2. Upload `Combined Data.csv` to the Colab session
3. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
4. Run all cells
5. Download `saved_models.zip` at the end
6. Extract the zip and place the `saved_models/` folder in this project root

### Step 2 — Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python setup_nltk.py

# Launch the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repository to GitHub (including `saved_models/` folder)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set main file to `app.py`
4. Click **Deploy**

> **Note:** The `saved_models/` folder contains large binary files. If they exceed GitHub's 100 MB limit, use [Git LFS](https://git-lfs.github.com/) or host the models on Google Drive and load them at runtime.

---

## 🏗️ Model Architecture

```
Input (padded sequence, max_len=100)
    │
    ▼
Embedding Layer     (vocab=20000, dim=64)
    │
    ▼
SimpleRNN Layer     (128 units, tanh activation)
    │
    ▼
Dropout             (0.4)
    │
    ▼
Dense Hidden Layer  (64 units, ReLU)
    │
    ▼
Dropout             (0.3)
    │
    ▼
Dense Output Layer  (7 units, Softmax)
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) |
| Total Samples | ~53,000 |
| Classes | 7 |
| Split | 70% train / 15% val / 15% test |

**Class Distribution:**

| Class | Count |
|-------|-------|
| Normal | 16,351 |
| Depression | 15,404 |
| Suicidal | 10,653 |
| Anxiety | 3,888 |
| Bipolar | 2,877 |
| Stress | 2,669 |
| Personality Disorder | 1,201 |

---

## 🔬 Tasks Covered

| Task | Description |
|------|-------------|
| Task 1 | Dataset Understanding — class distribution, text length analysis |
| Task 2 | Text Preprocessing — lowercase, punctuation removal, stopwords, tokenization |
| Task 3 | Sequence Preparation — Tokenizer, word indexing, padding |
| Task 4 | Simple RNN Architecture — Embedding → RNN → Dense |
| Task 5 | Model Training — GPU, early stopping, LR scheduling |
| Task 6 | Model Evaluation — Accuracy, Precision, Recall, F1, Confusion Matrix |
| Task 7 | Sequence Understanding — hidden state concept explained |
| Task 8 | Real-Time Prediction — custom sentence testing |
| Task 9 | Save Model — `.h5`, tokenizer, label encoder exported |

---

## ⚠️ Disclaimer

This application is for **educational purposes only** and is **not** a substitute for professional medical advice, diagnosis, or treatment. If you or someone you know is in crisis, please contact:

- **988 Suicide & Crisis Lifeline**: Call or text **988**
- **Crisis Text Line**: Text HOME to **741741**
- **Emergency Services**: **911**

---

## 🛠️ Tech Stack

- **Model**: TensorFlow / Keras (Simple RNN)
- **Frontend**: Streamlit
- **NLP**: NLTK, Keras Tokenizer
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
