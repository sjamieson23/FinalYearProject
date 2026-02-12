"""
Ensemble selection with a Logistic Regression meta-classifier:
combine BertBodyAndSubj and BertBody.

Instead of picking the prediction with higher confidence when the two
models disagree, we train a Logistic Regression classifier on simple
meta-features derived from both models' outputs and use that to produce
the final prediction.

Run from project root: python -m EnsembleAgent.SelectingLr
Or from EnsembleAgent: python SelectingLr.py (uses paths relative to project root via __file__).
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import BertForSequenceClassification, BertTokenizer

# Paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_CSV = os.path.join(PROJECT_ROOT, "EnsembleAgent", "Data", "all_data_test.csv")
BERT_BODY_AND_SUBJ_DIR = os.path.join(PROJECT_ROOT, "EnsembleAgent", "SavedModels", "BertBodyAndSubj", "model")
BERT_BODY_MODEL_DIR = os.path.join(PROJECT_ROOT, "EnsembleAgent", "SavedModels", "BertBody", "model")
BERT_BODY_TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "EnsembleAgent", "SavedModels", "BertBody", "tokenizer")

MAX_LENGTH = 512
BATCH_SIZE = 32
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Apple Silicon GPU
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def _prepare_text_body_and_subj(df: pd.DataFrame):
    subj = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    return (subj.str.strip() + " [SEP] " + body.str.strip()).tolist()


def _prepare_text_body_only(df: pd.DataFrame):
    return df["body"].fillna("").astype(str).str.strip().tolist()


def _get_predictions_and_probs(
    model, tokenizer, texts, device, batch_size=BATCH_SIZE
):
    """Return (preds, confidences) where preds are 0/1 and confidences are max probability per sample."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            all_logits.append(out.logits.cpu())
    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1).numpy()
    confidences = probs.max(dim=1).values.numpy()
    return preds, confidences


def main():
    print("Loading test data...")
    test_df = pd.read_csv(TEST_CSV)
    test_df["label"] = test_df["label"].astype(int)
    labels = test_df["label"].values

    print("Loading BertBodyAndSubj (subject + body)...")
    tokenizer_bs = BertTokenizer.from_pretrained(BERT_BODY_AND_SUBJ_DIR)
    model_bs = BertForSequenceClassification.from_pretrained(BERT_BODY_AND_SUBJ_DIR, num_labels=2)
    model_bs.to(DEVICE)

    print("Loading BertBody (body only)...")
    tokenizer_b = BertTokenizer.from_pretrained(BERT_BODY_TOKENIZER_DIR)
    model_b = BertForSequenceClassification.from_pretrained(BERT_BODY_MODEL_DIR, num_labels=2)
    model_b.to(DEVICE)

    texts_bs = _prepare_text_body_and_subj(test_df)
    texts_b = _prepare_text_body_only(test_df)

    print("Running BertBodyAndSubj...")
    preds_bs, conf_bs = _get_predictions_and_probs(model_bs, tokenizer_bs, texts_bs, DEVICE)
    print("Running BertBody...")
    preds_b, conf_b = _get_predictions_and_probs(model_b, tokenizer_b, texts_b, DEVICE)

    # Build meta-features for Logistic Regression:
    # [pred_bs, conf_bs, pred_b, conf_b]
    print("Training Logistic Regression meta-classifier...")
    X_meta = np.column_stack([preds_bs, conf_bs, preds_b, conf_b])
    y_meta = labels

    lr_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr_clf.fit(X_meta, y_meta)

    preds_ensemble = lr_clf.predict(X_meta)

    f1_bs = f1_score(labels, preds_bs, average="binary", zero_division=0)
    f1_b = f1_score(labels, preds_b, average="binary", zero_division=0)
    f1_ensemble = f1_score(labels, preds_ensemble, average="binary", zero_division=0)

    print("\n--- F1 comparison ---")
    print(f"BertBodyAndSubj (alone):        F1 = {f1_bs:.4f}")
    print(f"BertBody (alone):               F1 = {f1_b:.4f}")
    print(f"LR ensemble (meta-classifier):  F1 = {f1_ensemble:.4f}")
    print("\nEnsemble vs BertBodyAndSubj:", f1_ensemble - f1_bs, "(positive = ensemble better)")
    print("Ensemble vs BertBody:     ", f1_ensemble - f1_b, "(positive = ensemble better)")


if __name__ == "__main__":
    main()

