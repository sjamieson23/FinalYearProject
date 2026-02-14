"""
Ensemble selection: combine BertBodyAndSubj and BertBody.
For each email, take the 'preferred' classification (same as both, or higher-confidence when they disagree).
Compare F1 of the two single models vs the ensemble.

Run from project root: python -m EnsembleAgent.Selecting
Or from EnsembleAgent: python Selecting.py (uses paths relative to project root via __file__).
"""

import os

import joblib
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.metrics import f1_score
from transformers import BertForSequenceClassification, BertTokenizer

# Paths relative to project root
TEST_CSV = os.path.join(
    "Data", "all_data_test.csv"
)
BERT_BODY_AND_SUBJ_DIR = os.path.join(
    "SavedModels", "BertBodyAndSubj", "model"
)
BERT_BODY_MODEL_DIR = os.path.join(
    "SavedModels", "BertBody", "model"
)
BERT_BODY_TOKENIZER_DIR = os.path.join(
    "SavedModels", "BertBody", "tokenizer"
)
BERT_SUBJ_MODEL_DIR = os.path.join(
    "SavedModels", "BertSubj", "model"
)
BERT_SUBJ_TOKENIZER_DIR = os.path.join(
    "SavedModels", "BertSubj", "tokenizer"
)

RESULTS_TFIDF_MLP = os.path.join(
    "SavedModels", "TFIDFBodyAndSubjectMLP"
)
RESULTS_TFIDF_LR = os.path.join(
    "SavedModels", "TFIDFBodyAndSubjectLR"
)
RESULTS_TFIDF_RF = os.path.join(
    "SavedModels", "TFIDFBodyAndSubjectRF"
)
RESULTS_TFIDF_NB = os.path.join(
    "SavedModels", "TFIDFBodyAndSubjectNB"
)
RESULTS_TFIDF_DT = os.path.join(
    "SavedModels", "TFIDFBodyAndSubjectDT"
)
RESULTS_W2V_MLP = os.path.join(
    "SavedModels", "Word2VecMLP"
)
RESULTS_W2V_RF = os.path.join(
    "SavedModels", "Word2VecRF"
)

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


def _combine_text_simple(df: pd.DataFrame) -> np.ndarray:
    subj = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    return (subj.str.strip() + body.str.strip()).values


def _combine_text_with_sep(df: pd.DataFrame) -> np.ndarray:
    subj = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    return (subj.str.strip() + " [SEP] " + body.str.strip()).values


def _tokenize_text(text: str):
    return str(text).lower().split()


def _document_vector(w2v_model: Word2Vec, doc: str) -> np.ndarray:
    words = _tokenize_text(doc)
    word_vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
    if not word_vectors:
        return np.zeros(w2v_model.vector_size, dtype=np.float32)
    return np.mean(word_vectors, axis=0)


def main():
    print("Loading test data...")
    test_df = pd.read_csv(TEST_CSV)
    test_df["label"] = test_df["label"].astype(int)
    labels = test_df["label"].values
    # ------------------------------------------------------------------
    # Load BERT models
    # ------------------------------------------------------------------
    print("Loading BertBodyAndSubj (subject + body)...")
    tokenizer_bs = BertTokenizer.from_pretrained(BERT_BODY_AND_SUBJ_DIR)
    model_bs = BertForSequenceClassification.from_pretrained(
        BERT_BODY_AND_SUBJ_DIR, num_labels=2
    )
    model_bs.to(DEVICE)

    print("Loading BertBody (body only)...")
    tokenizer_b = BertTokenizer.from_pretrained(BERT_BODY_TOKENIZER_DIR)
    model_b = BertForSequenceClassification.from_pretrained(
        BERT_BODY_MODEL_DIR, num_labels=2
    )
    model_b.to(DEVICE)

    print("Loading BertSubj (subject only)...")
    tokenizer_subj = BertTokenizer.from_pretrained(BERT_SUBJ_MODEL_DIR)
    model_subj = BertForSequenceClassification.from_pretrained(
        BERT_SUBJ_MODEL_DIR, num_labels=2
    )
    model_subj.to(DEVICE)

    texts_bs = _prepare_text_body_and_subj(test_df)
    texts_b = _prepare_text_body_only(test_df)
    texts_subj = test_df["subject"].fillna("").astype(str).str.strip().tolist()

    print("Running BertBodyAndSubj...")
    preds_bs, conf_bs = _get_predictions_and_probs(
        model_bs, tokenizer_bs, texts_bs, DEVICE
    )
    print("Running BertBody...")
    preds_b, conf_b = _get_predictions_and_probs(
        model_b, tokenizer_b, texts_b, DEVICE
    )
    print("Running BertSubj...")
    preds_subj, conf_subj = _get_predictions_and_probs(
        model_subj, tokenizer_subj, texts_subj, DEVICE
    )

    # ------------------------------------------------------------------
    # Load TF-IDF-based models
    # ------------------------------------------------------------------
    print("Loading TFIDFBodyAndSubjectMLP...")
    tfidf_mlp_model = joblib.load(
        os.path.join(RESULTS_TFIDF_MLP, "tfidf_mlp_model.joblib")
    )
    tfidf_mlp_vectorizer = joblib.load(
        os.path.join(RESULTS_TFIDF_MLP, "tfidf_vectorizer.joblib")
    )

    print("Loading TFIDFBodyAndSubjectLR...")
    tfidf_lr_model = joblib.load(
        os.path.join(RESULTS_TFIDF_LR, "tfidf_lr_model.joblib")
    )
    tfidf_lr_vectorizer = joblib.load(
        os.path.join(RESULTS_TFIDF_LR, "tfidf_vectorizer.joblib")
    )

    print("Loading TFIDFBodyAndSubjectRF...")
    tfidf_rf_model = joblib.load(
        os.path.join(RESULTS_TFIDF_RF, "tfidf_rf_model.joblib")
    )
    tfidf_rf_vectorizer = joblib.load(
        os.path.join(RESULTS_TFIDF_RF, "tfidf_vectorizer.joblib")
    )

    print("Loading TFIDFBodyAndSubjectNB...")
    tfidf_nb_model = joblib.load(
        os.path.join(RESULTS_TFIDF_NB, "tfidf_nb_model.joblib")
    )
    tfidf_nb_vectorizer = joblib.load(
        os.path.join(RESULTS_TFIDF_NB, "tfidf_vectorizer.joblib")
    )

    print("Loading TFIDFBodyAndSubjectDT...")
    tfidf_dt_model = joblib.load(
        os.path.join(RESULTS_TFIDF_DT, "tfidf_dt_model.joblib")
    )
    tfidf_dt_vectorizer = joblib.load(
        os.path.join(RESULTS_TFIDF_DT, "tfidf_vectorizer.joblib")
    )

    # Texts for TF-IDF models
    texts_tfidf_simple = _combine_text_simple(test_df)
    texts_tfidf_sep = _combine_text_with_sep(test_df)

    print("Running TFIDFBodyAndSubjectMLP...")
    X_tfidf_mlp = tfidf_mlp_vectorizer.transform(texts_tfidf_simple)
    probs_tfidf_mlp = tfidf_mlp_model.predict_proba(X_tfidf_mlp)
    preds_tfidf_mlp = probs_tfidf_mlp.argmax(axis=1)
    conf_tfidf_mlp = probs_tfidf_mlp.max(axis=1)

    print("Running TFIDFBodyAndSubjectLR...")
    X_tfidf_lr = tfidf_lr_vectorizer.transform(texts_tfidf_simple)
    probs_tfidf_lr = tfidf_lr_model.predict_proba(X_tfidf_lr)
    preds_tfidf_lr = probs_tfidf_lr.argmax(axis=1)
    conf_tfidf_lr = probs_tfidf_lr.max(axis=1)

    print("Running TFIDFBodyAndSubjectRF...")
    X_tfidf_rf = tfidf_rf_vectorizer.transform(texts_tfidf_sep)
    probs_tfidf_rf = tfidf_rf_model.predict_proba(X_tfidf_rf)
    preds_tfidf_rf = probs_tfidf_rf.argmax(axis=1)
    conf_tfidf_rf = probs_tfidf_rf.max(axis=1)

    print("Running TFIDFBodyAndSubjectNB...")
    X_tfidf_nb = tfidf_nb_vectorizer.transform(texts_tfidf_sep)
    probs_tfidf_nb = tfidf_nb_model.predict_proba(X_tfidf_nb)
    preds_tfidf_nb = probs_tfidf_nb.argmax(axis=1)
    conf_tfidf_nb = probs_tfidf_nb.max(axis=1)

    print("Running TFIDFBodyAndSubjectDT...")
    X_tfidf_dt = tfidf_dt_vectorizer.transform(texts_tfidf_sep)
    probs_tfidf_dt = tfidf_dt_model.predict_proba(X_tfidf_dt)
    preds_tfidf_dt = probs_tfidf_dt.argmax(axis=1)
    conf_tfidf_dt = probs_tfidf_dt.max(axis=1)

    # ------------------------------------------------------------------
    # Load Word2Vec-based models
    # ------------------------------------------------------------------
    print("Loading Word2VecBodyAndSubjectMLP...")
    w2v_mlp_model = joblib.load(
        os.path.join(RESULTS_W2V_MLP, "word2vec_mlp_model.joblib")
    )
    w2v_mlp_w2v = Word2Vec.load(
        os.path.join(RESULTS_W2V_MLP, "word2vec_model.model")
    )

    print("Loading Word2VecBodyAndSubjectRF...")
    w2v_rf_model = joblib.load(
        os.path.join(RESULTS_W2V_RF, "word2vec_rf_model.joblib")
    )
    w2v_rf_w2v = Word2Vec.load(
        os.path.join(RESULTS_W2V_RF, "word2vec_model.model")
    )

    # Texts for Word2Vec models
    texts_w2v_simple = _combine_text_simple(test_df)
    texts_w2v_sep = _combine_text_with_sep(test_df)

    print("Running Word2VecBodyAndSubjectMLP...")
    X_w2v_mlp = np.vstack(
        [_document_vector(w2v_mlp_w2v, t) for t in texts_w2v_simple]
    )
    probs_w2v_mlp = w2v_mlp_model.predict_proba(X_w2v_mlp)
    preds_w2v_mlp = probs_w2v_mlp.argmax(axis=1)
    conf_w2v_mlp = probs_w2v_mlp.max(axis=1)

    print("Running Word2VecBodyAndSubjectRF...")
    X_w2v_rf = np.vstack(
        [_document_vector(w2v_rf_w2v, t) for t in texts_w2v_sep]
    )
    probs_w2v_rf = w2v_rf_model.predict_proba(X_w2v_rf)
    preds_w2v_rf = probs_w2v_rf.argmax(axis=1)
    conf_w2v_rf = probs_w2v_rf.max(axis=1)

    # ------------------------------------------------------------------
    # Single-model F1 scores
    # ------------------------------------------------------------------
    f1_bs = f1_score(labels, preds_bs, average="binary", zero_division=0)
    f1_b = f1_score(labels, preds_b, average="binary", zero_division=0)
    f1_w2v_mlp = f1_score(
        labels, preds_w2v_mlp, average="binary", zero_division=0
    )
    f1_tfidf_mlp = f1_score(
        labels, preds_tfidf_mlp, average="binary", zero_division=0
    )
    f1_tfidf_lr = f1_score(
        labels, preds_tfidf_lr, average="binary", zero_division=0
    )
    f1_w2v_rf = f1_score(
        labels, preds_w2v_rf, average="binary", zero_division=0
    )
    f1_tfidf_rf = f1_score(
        labels, preds_tfidf_rf, average="binary", zero_division=0
    )
    f1_tfidf_nb = f1_score(
        labels, preds_tfidf_nb, average="binary", zero_division=0
    )
    f1_subj = f1_score(
        labels, preds_subj, average="binary", zero_division=0
    )
    f1_tfidf_dt = f1_score(
        labels, preds_tfidf_dt, average="binary", zero_division=0
    )

    print("\n--- Individual model F1 scores ---")
    print(f"1. BertBodyAndSubj:              F1 = {f1_bs:.4f}")
    print(f"2. BertBody:                     F1 = {f1_b:.4f}")
    print(f"3. Word2VecBodyAndSubjectMLP:    F1 = {f1_w2v_mlp:.4f}")
    print(f"4. TFIDFBodyAndSubjectMLP:       F1 = {f1_tfidf_mlp:.4f}")
    print(f"5. TFIDFBodyAndSubjectLR:        F1 = {f1_tfidf_lr:.4f}")
    print(f"6. Word2VecBodyAndSubjectRF:     F1 = {f1_w2v_rf:.4f}")
    print(f"7. TFIDFBodyAndSubjectRF:        F1 = {f1_tfidf_rf:.4f}")
    print(f"8. TFIDFBodyAndSubjectNB:        F1 = {f1_tfidf_nb:.4f}")
    print(f"9. BertSubj:                     F1 = {f1_subj:.4f}")
    print(f"10. TFIDFBodyAndSubjectDT:       F1 = {f1_tfidf_dt:.4f}")

    # ------------------------------------------------------------------
    # Confidence-based ensemble, adding models one by one
    # ------------------------------------------------------------------
    model_names = [
        "BertBodyAndSubj",
        "BertBody",
        "Word2VecBodyAndSubjectMLP",
        "TFIDFBodyAndSubjectMLP",
        "TFIDFBodyAndSubjectLR",
        "Word2VecBodyAndSubjectRF",
        "TFIDFBodyAndSubjectRF",
        "TFIDFBodyAndSubjectNB",
        "BertSubj",
        "TFIDFBodyAndSubjectDT",
    ]
    model_preds = [
        preds_bs,
        preds_b,
        preds_w2v_mlp,
        preds_tfidf_mlp,
        preds_tfidf_lr,
        preds_w2v_rf,
        preds_tfidf_rf,
        preds_tfidf_nb,
        preds_subj,
        preds_tfidf_dt,
    ]
    model_conf = [
        conf_bs,
        conf_b,
        conf_w2v_mlp,
        conf_tfidf_mlp,
        conf_tfidf_lr,
        conf_w2v_rf,
        conf_tfidf_rf,
        conf_tfidf_nb,
        conf_subj,
        conf_tfidf_dt,
    ]

    print("\n--- Incremental confidence-based ensemble F1 ---")
    prev_f1 = None
    for k in range(1, len(model_names) + 1):
        # For each sample, pick the prediction from the model (among first k)
        # with the highest confidence (generalising the previous 2-model rule).
        conf_stack = np.vstack(model_conf[:k])  # shape: (k, n_samples)
        best_idx = conf_stack.argmax(axis=0)  # which model "wins" per sample

        # Build ensemble predictions from the winning model per sample
        preds_stack = np.vstack(model_preds[:k])  # shape: (k, n_samples)
        ensemble_preds = preds_stack[best_idx, np.arange(preds_stack.shape[1])]

        f1_k = f1_score(labels, ensemble_preds, average="binary", zero_division=0)
        if prev_f1 is None:
            print(
                f"Using 1 model ({model_names[0]}): F1 = {f1_k:.4f} (baseline)"
            )
        else:
            improvement = f1_k - prev_f1
            print(
                f"After adding {model_names[k-1]} (#{k}): F1 = {f1_k:.4f} "
                f"(ΔF1 vs previous = {improvement:+.4f})"
            )
        prev_f1 = f1_k


if __name__ == "__main__":
    main()
