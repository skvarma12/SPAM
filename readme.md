# Project 3: Spam Filter for Quora Questions

This project builds a deep learning model using pre-trained GloVe word embeddings to identify whether a Quora question is spam.

---

## 📁 Files

| File                     | Description                                       |
|--------------------------|---------------------------------------------------|
| `spam.py`                | Main Python script for training and saving model |
| `train.csv`              | Dataset with `question_text` and `target` label  |
| `glove.6B.100d.txt`      | Pre-trained GloVe word vectors (100d)            |
| `quora_spam_model.keras` | Saved trained model                              |
| `model_config.json`      | Model architecture in JSON format                |

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install tensorflow pandas scikit-learn
