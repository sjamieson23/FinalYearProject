#I want this file to look at the top 5 models, and compare coverage
#Look at uniquely correct predictions
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

    print("Loading TFIDF models...")
    tfidf_mlp_model = joblib.load(
        os.path.join(RESULTS_TFIDF_MLP, "tfidf_mlp_model.joblib")
    )
    tfidf_lr_model = joblib.load(
        os.path.join(RESULTS_TFIDF_LR, "tfidf_lr_model.joblib")
    )
    tfidf_rf_model = joblib.load(
        os.path.join(RESULTS_TFIDF_RF, "tfidf_rf_model.joblib")
    )
    tfidf_nb_model = joblib.load(
        os.path.join(RESULTS_TFIDF_NB, "tfidf_nb_model.joblib")
    )
    tfidf_dt_model = joblib.load(
        os.path.join(RESULTS_TFIDF_DT, "tfidf_dt_model.joblib")
    )
    # All TF-IDF vectorizers are identical (same params, same training data)
    # So we only need to load and use one
    tfidf_vectorizer = joblib.load(
        os.path.join(RESULTS_TFIDF_MLP, "tfidf_vectorizer.joblib")
    )

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

    # ------------------------------------------------------------------
    # Helper functions for text preparation and prediction
    # ------------------------------------------------------------------
    def combine_text_tfidf(row):
        subj = str(row["subject"]) if not pd.isna(row["subject"]) else ""
        body = str(row["body"]) if not pd.isna(row["body"]) else ""
        return subj.strip() + body.strip()

    def combine_text_w2v(row):
        subj = str(row["subject"]) if not pd.isna(row["subject"]) else ""
        body = str(row["body"]) if not pd.isna(row["body"]) else ""
        return subj.strip() + " [SEP] " + body.strip()

    def tokenize_text(text):
        return str(text).lower().split()

    def document_vector(w2v_model, doc):
        words = tokenize_text(doc)
        word_vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
        if not word_vectors:
            return np.zeros(w2v_model.vector_size)
        return np.mean(word_vectors, axis=0)

    def bert_predict_labels(texts, tokenizer, model, device, batch_size=BATCH_SIZE):
        model.eval()
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                encodings = tokenizer(
                    batch_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LENGTH,
                )
                encodings = {k: v.to(device) for k, v in encodings.items()}
                outputs = model(**encodings)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.append(preds)
        return np.concatenate(all_preds, axis=0)

    # ------------------------------------------------------------------
    # Compute predictions for all models on the test set
    # ------------------------------------------------------------------
    y_true = labels

    # Text for TF-IDF models (subject + body, no separator)
    test_df["text_tfidf"] = test_df.apply(combine_text_tfidf, axis=1)

    # All TF-IDF models use the same vectorizer, so transform once
    X_tfidf = tfidf_vectorizer.transform(test_df["text_tfidf"])

    pred_tfidf_mlp = tfidf_mlp_model.predict(X_tfidf)
    pred_tfidf_lr = tfidf_lr_model.predict(X_tfidf)
    pred_tfidf_rf = tfidf_rf_model.predict(X_tfidf)
    pred_tfidf_nb = tfidf_nb_model.predict(X_tfidf)
    pred_tfidf_dt = tfidf_dt_model.predict(X_tfidf)

    # Text for Word2Vec models (subject + [SEP] + body)
    test_df["text_w2v"] = test_df.apply(combine_text_w2v, axis=1)
    X_w2v_mlp = np.vstack(
        [document_vector(w2v_mlp_w2v, doc) for doc in test_df["text_w2v"]]
    )
    X_w2v_rf = np.vstack(
        [document_vector(w2v_rf_w2v, doc) for doc in test_df["text_w2v"]]
    )

    pred_w2v_mlp = w2v_mlp_model.predict(X_w2v_mlp)
    pred_w2v_rf = w2v_rf_model.predict(X_w2v_rf)

    # Text for BERT models
    subj_series = test_df["subject"].fillna("").astype(str)
    body_series = test_df["body"].fillna("").astype(str)

    texts_bs = (subj_series.str.strip() + " [SEP] " + body_series.str.strip()).tolist()
    texts_body = body_series.str.strip().tolist()
    texts_subj = subj_series.str.strip().tolist()

    pred_bert_bs = bert_predict_labels(texts_bs, tokenizer_bs, model_bs, DEVICE)
    pred_bert_body = bert_predict_labels(texts_body, tokenizer_b, model_b, DEVICE)
    pred_bert_subj = bert_predict_labels(texts_subj, tokenizer_subj, model_subj, DEVICE)

    # ------------------------------------------------------------------
    # Count uniquely correct predictions per model
    # ------------------------------------------------------------------
    model_preds = {
        "TFIDFBodyAndSubjectMLP": pred_tfidf_mlp,
        "TFIDFBodyAndSubjectLR": pred_tfidf_lr,
        "TFIDFBodyAndSubjectRF": pred_tfidf_rf,
        "TFIDFBodyAndSubjectNB": pred_tfidf_nb,
        "TFIDFBodyAndSubjectDT": pred_tfidf_dt,
        "Word2VecBodyAndSubjectMLP": pred_w2v_mlp,
        "Word2VecBodyAndSubjectRF": pred_w2v_rf,
        "BertBodyAndSubj": pred_bert_bs,
        "BertBody": pred_bert_body,
        "BertSubj": pred_bert_subj,
    }

    unique_counts = {name: 0 for name in model_preds.keys()}

    for i in range(len(y_true)):
        correct_models = [
            name
            for name, preds in model_preds.items()
            if preds[i] == y_true[i]
        ]
        if len(correct_models) == 1:
            unique_counts[correct_models[0]] += 1

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("Uniquely correct predictions per model:")
    for name, count in unique_counts.items():
        print(f"{name}: {count}")

if __name__ == "__main__":
    main()