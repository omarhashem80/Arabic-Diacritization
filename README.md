# 🔤 Arabic Text Diacritization (BiLSTM)

## 📌 Overview

Arabic diacritics are short vowels that affect both pronunciation and meaning, yet they are usually omitted from text. This creates ambiguity and hurts NLP systems such as **Text-to-Speech** and **Machine Translation**.

This project implements a **character-level Arabic diacritization system** using a **Bidirectional LSTM (BiLSTM)**. Given undiacritized Arabic text, the model restores the correct diacritics using contextual information from surrounding characters.

Evaluation is performed using **Diacritic Error Rate (DER)** and results are benchmarked through a **Kaggle competition**.

---

## ✨ Key Points

* Character-level sequence labeling
* Context-aware BiLSTM architecture
* Supports compound diacritics
* End-to-end diacritization pipeline
* Evaluated with DER on Kaggle

---

## 🧠 Task Definition

**Input:**

```
ذهب الطالب الى الجامعة
```

**Output:**

```
ذَهَبَ الطَّالِبُ إِلَى الجَامِعَةِ
```

Each character is classified into one of **16 diacritic labels**, including vowels, tanween, sukun, shadda combinations, no-diacritic, and padding.

---

## 🗂 Dataset

| Split      | Description                      |
| ---------- | -------------------------------- |
| Train      | Fully diacritized sentences      |
| Validation | Fully diacritized sentences      |
| Test       | Undiacritized sentences (Kaggle) |

One sentence per line.

---

## 🔧 Pipeline

```
Text Cleaning
   → Character Tokenization
   → Trainable Embeddings
   → BiLSTM Encoder
   → Diacritic Classification
   → Diacritized Text
```

Each character is predicted using both left and right context.

---

## 🧹 Preprocessing

* Remove non-Arabic characters
* Normalize text and spacing
* Strip diacritics from inputs
* Fixed max sentence length (600 chars)
* Padding with masking

---

## 🏗 Model

**BiLSTM with Trainable Embeddings**

* Embedding size: 300
* BiLSTM layers: 5
* Hidden size: 256 per direction
* Dropout between layers
* Fully connected output layer

---

## 📊 Accuracy & Metrics

### BiLSTM (Trainable Embeddings)

**Training**

* Loss: **0.01227**
* Accuracy: **98.86%** (Last char: 98.223%)
* DER: **1.14%** (Last char: 1.777%)
* F1: **0.964**

**Validation**

* Accuracy: **98.29%** (Last char: 96.553%)
* DER: **1.71%** (Last char: 3.447%)
* F1: **0.934**

> DER is the primary Kaggle ranking metric.

---

## 🏆 Kaggle Results

### Final Ranking

<p align="center">
  <img src="images/kaggle_rank_1.png" width="45%" />
  <img src="images/kaggle_rank_2.png" width="45%" />
</p>

---

## 🔮 Future Work

* Transformer-based models (AraBERT, ByT5)
* CRF decoding layer
* Larger datasets
* Faster inference
* TTS integration
