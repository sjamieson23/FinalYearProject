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

import joblib
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import BertForSequenceClassification, BertTokenizer

# Paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_CSV = os.path.join(
    PROJECT_ROOT, "EnsembleAgent", "Data", "all_data_test.csv"
)
BERT_BODY_AND_SUBJ_DIR = os.path.join(
    PROJECT_ROOT, "EnsembleAgent", "SavedModels", "BertBodyAndSubj", "model"
)
BERT_BODY_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "EnsembleAgent", "SavedModels", "BertBody", "model"
)
BERT_BODY_TOKENIZER_DIR = os.path.join(
    PROJECT_ROOT, "EnsembleAgent", "SavedModels", "BertBody", "tokenizer"
)
BERT_SUBJ_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "EnsembleAgent", "SavedModels", "BertSubj", "model"
)
BERT_SUBJ_TOKENIZER_DIR = os.path.join(
    PROJECT_ROOT, "EnsembleAgent", "SavedModels", "BertSubj", "tokenizer"
)

# Classical model result directories (trained in SingleAgent)
SINGLE_AGENT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "SingleAgent", "Results")

RESULTS_TFIDF_MLP = os.path.join(
    SINGLE_AGENT_RESULTS_DIR, "TFIDFBodyAndSubjectMLP"
)
RESULTS_TFIDF_LR = os.path.join(
    SINGLE_AGENT_RESULTS_DIR, "TFIDFBodyAndSubjectLR"
)
RESULTS_TFIDF_RF = os.path.join(
    SINGLE_AGENT_RESULTS_DIR, "TFIDFBodyAndSubjectRF"
)
RESULTS_TFIDF_NB = os.path.join(
    SINGLE_AGENT_RESULTS_DIR, "TFIDFBodyAndSubjectNB"
)
RESULTS_TFIDF_DT = os.path.join(
    SINGLE_AGENT_RESULTS_DIR, "TFIDFBodyAndSubjectDT"
)
RESULTS_W2V_MLP = os.path.join(
    SINGLE_AGENT_RESULTS_DIR, "Word2VecBodyAndSubjectMLP"
)
RESULTS_W2V_RF = os.path.join(
    SINGLE_AGENT_RESULTS_DIR, "Word2VecBodyAndSubjectRF"
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
    tokenizer_subj = BertTokenizer.from_pretrained(BERT_SUBJ_TOKENIZER_DIR)
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
    # Build meta-features for Logistic Regression over all models
    # Order must match the requested sequence:
    # BertBodyAndSubj, BertBody, Word2VecMLP, TFIDF MLP, TFIDF LR,
    # Word2VecRF, TFIDF RF, TFIDF NB, BertSubj, TFIDF DT
    # For each model we use [pred, confidence] as features.
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
    preds_list = [
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
    conf_list = [
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

    # Also report individual F1 scores for reference
    f1_individual = []
    for name, preds in zip(model_names, preds_list):
        f1 = f1_score(labels, preds, average="binary", zero_division=0)
        f1_individual.append((name, f1))

    print("\n--- Individual model F1 scores ---")
    for i, (name, f1) in enumerate(f1_individual, start=1):
        print(f"{i}. {name:30s} F1 = {f1:.4f}")

    print("\n--- Incremental LR meta-ensemble F1 ---")
    prev_f1 = None
    for k in range(1, len(model_names) + 1):
        # Stack features for first k models: [pred_1, conf_1, ..., pred_k, conf_k]
        features_k = [preds_list[i] for i in range(k)] + [
            conf_list[i] for i in range(k)
        ]
        X_meta_k = np.column_stack(features_k)

        lr_clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", n_jobs=-1
        )
        lr_clf.fit(X_meta_k, labels)

        preds_ens_k = lr_clf.predict(X_meta_k)
        f1_k = f1_score(labels, preds_ens_k, average="binary", zero_division=0)

        if prev_f1 is None:
            print(
                f"Using 1 model ({model_names[0]}): F1 = {f1_k:.4f} (baseline meta-ensemble)"
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

