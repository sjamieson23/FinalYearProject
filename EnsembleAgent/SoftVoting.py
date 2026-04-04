"""
Discover models under SavedModels/ and FinetunedOutputs/, compute soft-voting probabilities
on ensemble_training_data.csv, and greedily build an ensemble:

1. Start with the single model with highest training F1.
2. Repeatedly add the remaining model that yields the largest training F1 when averaged
   (soft vote) with the current set.
3. Stop when the best available relative F1 gain is below 1% (vs the current ensemble F1).
4. Evaluate the final soft-voting ensemble on ensemble_testing_data.csv, print metrics, and
   save a joblib artifact for reuse (see load_soft_voting_ensemble / predict_soft_voting_proba).

Run from project root: python -m EnsembleAgent.SoftVoting
"""

from __future__ import annotations

import gc
import glob
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import BertForSequenceClassification, BertTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(SCRIPT_DIR, "Data", "ensemble_training_data.csv")
TEST_CSV = os.path.join(SCRIPT_DIR, "Data", "ensemble_testing_data.csv")
ENSEMBLE_SAVE_DIR = os.path.join(SCRIPT_DIR, "SavedSoftVotingEnsemble")
ENSEMBLE_JOBLIB_PATH = os.path.join(ENSEMBLE_SAVE_DIR, "soft_voting_ensemble.joblib")
ENSEMBLE_TEST_METRICS_PATH = os.path.join(ENSEMBLE_SAVE_DIR, "ensemble_test_metrics.txt")
MODEL_ROOTS = [
    os.path.join(SCRIPT_DIR, "SavedModels"),
    os.path.join(SCRIPT_DIR, "FinetunedOutputs"),
]

MAX_LENGTH = 512
BATCH_SIZE = 32
MIN_RELATIVE_F1_GAIN = 0.01

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def _torch_release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _bert_kind(folder_name: str):
    if "BertBodyAndSubj" in folder_name:
        return "bert_bs"
    if "BertBody" in folder_name:
        return "bert_body"
    if "BertSubj" in folder_name:
        return "bert_subj"
    return None


def _has_bert_weights(model_dir: str) -> bool:
    return os.path.isfile(os.path.join(model_dir, "model.safetensors")) or os.path.isfile(
        os.path.join(model_dir, "pytorch_model.bin")
    )


def discover_model_specs():
    """Return list of dicts: display_name, kind, path, and paths for joblib models."""
    specs = []
    for root in MODEL_ROOTS:
        if not os.path.isdir(root):
            continue
        root_tag = os.path.basename(root)
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if not os.path.isdir(path):
                continue
            display_name = f"{root_tag}/{name}"
            model_dir = os.path.join(path, "model")
            cfg = os.path.join(model_dir, "config.json")

            bkind = _bert_kind(name)
            if bkind and os.path.isfile(cfg) and _has_bert_weights(model_dir):
                specs.append({"display_name": display_name, "kind": bkind, "path": path})
                continue

            tfidf_models = glob.glob(os.path.join(path, "tfidf_*_model.joblib"))
            vec_path = os.path.join(path, "tfidf_vectorizer.joblib")
            if len(tfidf_models) == 1 and os.path.isfile(vec_path):
                specs.append(
                    {
                        "display_name": display_name,
                        "kind": "tfidf",
                        "path": path,
                        "clf_path": tfidf_models[0],
                        "vectorizer_path": vec_path,
                    }
                )
                continue

            w2v_clfs = sorted(glob.glob(os.path.join(path, "word2vec_*_model.joblib")))
            w2v_path = os.path.join(path, "word2vec_model.model")
            if w2v_clfs and os.path.isfile(w2v_path):
                specs.append(
                    {
                        "display_name": display_name,
                        "kind": "w2v",
                        "path": path,
                        "clf_path": w2v_clfs[0],
                        "w2v_path": w2v_path,
                    }
                )
    return specs


def combine_text_tfidf(row):
    subj = str(row["subject"]) if not pd.isna(row["subject"]) else ""
    body = str(row["body"]) if not pd.isna(row["body"]) else ""
    return subj.strip() + body.strip()


def combine_text_tfidf_sep(row):
    subj = str(row["subject"]) if not pd.isna(row["subject"]) else ""
    body = str(row["body"]) if not pd.isna(row["body"]) else ""
    return subj.strip() + " [SEP] " + body.strip()


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


def _tfidf_use_sep(folder_name: str) -> bool:
    """Match Selecting.py: RF / NB / DT bundles used subject+body with [SEP]."""
    u = folder_name.upper()
    return "SUBJECTRF" in u or "SUBJECTNB" in u or "SUBJECTDT" in u


def _normalize_proba(raw: np.ndarray) -> np.ndarray:
    p = np.asarray(raw, dtype=np.float64)
    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    if p.shape[1] != 2:
        raise ValueError(f"Expected binary proba with 2 columns, got shape {p.shape}")
    s = p.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return p / s


def bert_predict_proba(texts, tokenizer, model, device, batch_size=BATCH_SIZE) -> np.ndarray:
    model.eval()
    model.to(device)
    chunks = []
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
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            chunks.append(probs)
    return np.vstack(chunks)


def predict_proba_for_spec(
    spec: dict[str, Any],
    data_df: pd.DataFrame,
    subj_series: pd.Series,
    body_series: pd.Series,
    device,
) -> np.ndarray:
    kind = spec["kind"]
    path = spec["path"]
    name = os.path.basename(path)

    if kind == "bert_bs":
        d = os.path.join(path, "model")
        tokenizer = BertTokenizer.from_pretrained(d)
        model = BertForSequenceClassification.from_pretrained(d, num_labels=2)
        texts = (subj_series.str.strip() + " [SEP] " + body_series.str.strip()).tolist()
        proba = bert_predict_proba(texts, tokenizer, model, device)
        del model, tokenizer
        _torch_release()
        return _normalize_proba(proba)

    if kind == "bert_body":
        tok_dir = os.path.join(path, "tokenizer")
        mod_dir = os.path.join(path, "model")
        tokenizer = BertTokenizer.from_pretrained(tok_dir)
        model = BertForSequenceClassification.from_pretrained(mod_dir, num_labels=2)
        texts = body_series.str.strip().tolist()
        proba = bert_predict_proba(texts, tokenizer, model, device)
        del model, tokenizer
        _torch_release()
        return _normalize_proba(proba)

    if kind == "bert_subj":
        tok_dir = os.path.join(path, "tokenizer")
        mod_dir = os.path.join(path, "model")
        tokenizer = BertTokenizer.from_pretrained(tok_dir)
        model = BertForSequenceClassification.from_pretrained(mod_dir, num_labels=2)
        texts = subj_series.str.strip().tolist()
        proba = bert_predict_proba(texts, tokenizer, model, device)
        del model, tokenizer
        _torch_release()
        return _normalize_proba(proba)

    if kind == "tfidf":
        clf = joblib.load(spec["clf_path"])
        vectorizer = joblib.load(spec["vectorizer_path"])
        col = "text_tfidf_sep" if _tfidf_use_sep(name) else "text_tfidf"
        X = vectorizer.transform(data_df[col])
        proba = clf.predict_proba(X)
        return _normalize_proba(proba)

    if kind == "w2v":
        clf = joblib.load(spec["clf_path"])
        w2v = Word2Vec.load(spec["w2v_path"])
        X = np.vstack([document_vector(w2v, doc) for doc in data_df["text_w2v"]])
        proba = clf.predict_proba(X)
        return _normalize_proba(proba)

    raise ValueError(f"Unknown spec kind: {kind}")


def soft_vote_proba(proba_list: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(proba_list, axis=0), axis=0)


def soft_vote_f1(proba_list: list[np.ndarray], y_true: np.ndarray) -> float:
    stacked = soft_vote_proba(proba_list)
    preds = stacked.argmax(axis=1)
    return float(f1_score(y_true, preds, average="binary", zero_division=0))


def relative_gain(f_old: float, f_new: float) -> float:
    if f_old <= 1e-12:
        return f_new - f_old
    return (f_new - f_old) / f_old


def _prepare_email_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    out = df.copy()
    out["text_tfidf"] = out.apply(combine_text_tfidf, axis=1)
    out["text_tfidf_sep"] = out.apply(combine_text_tfidf_sep, axis=1)
    out["text_w2v"] = out.apply(combine_text_w2v, axis=1)
    subj = out["subject"].fillna("").astype(str)
    body = out["body"].fillna("").astype(str)
    return out, subj, body


def _spec_to_abspath(spec: dict[str, Any]) -> dict[str, Any]:
    s = dict(spec)
    s["path"] = os.path.abspath(spec["path"])
    if "clf_path" in spec:
        s["clf_path"] = os.path.abspath(spec["clf_path"])
    if "vectorizer_path" in spec:
        s["vectorizer_path"] = os.path.abspath(spec["vectorizer_path"])
    if "w2v_path" in spec:
        s["w2v_path"] = os.path.abspath(spec["w2v_path"])
    return s


def classification_metrics_dict(y_true: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    preds = np.asarray(preds).astype(int)
    tp = fp = tn = fn = 0
    for i in range(len(y_true)):
        t, p = int(y_true[i]), int(preds[i])
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 0:
            tn += 1
        else:
            fn += 1
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, preds, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, preds, average="binary", zero_division=0)),
    }


def load_soft_voting_ensemble(path: str = ENSEMBLE_JOBLIB_PATH) -> dict[str, Any]:
    """Load artifact saved by main() (ordered member specs + metadata)."""
    return joblib.load(path)


def predict_soft_voting_proba(
    artifact: dict[str, Any],
    df: pd.DataFrame,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Run saved soft-voting ensemble on a DataFrame with subject, body columns.
    Returns (n_samples, 2) normalized class probabilities (mean across members).
    """
    if device is None:
        device = DEVICE
    data_df, subj_series, body_series = _prepare_email_df(df)
    probs = []
    for spec in artifact["specs_ordered"]:
        p = predict_proba_for_spec(spec, data_df, subj_series, body_series, device)
        probs.append(p)
    return soft_vote_proba(probs)


def predict_soft_voting_labels(
    artifact: dict[str, Any],
    df: pd.DataFrame,
    device: torch.device | None = None,
) -> np.ndarray:
    return predict_soft_voting_proba(artifact, df, device=device).argmax(axis=1)


def main():
    specs = discover_model_specs()
    if not specs:
        print("No models found under SavedModels/ or FinetunedOutputs/.")
        return

    print(f"Discovered {len(specs)} model bundle(s).")
    print(f"Loading training data: {TRAIN_CSV}")
    train_df = pd.read_csv(TRAIN_CSV)
    train_df = train_df.dropna(subset=["label"])
    train_df["label"] = train_df["label"].astype(int)
    y = train_df["label"].values
    train_df, subj_series, body_series = _prepare_email_df(train_df)

    name_to_proba: dict[str, np.ndarray] = {}
    single_f1: dict[str, float] = {}

    for spec in specs:
        name = spec["display_name"]
        print(f"\nPredicting proba: {name} ...")
        try:
            proba = predict_proba_for_spec(spec, train_df, subj_series, body_series, DEVICE)
        except Exception as ex:
            print(f"  Skipped ({ex})")
            continue
        if proba.shape[0] != len(y):
            print(f"  Skipped (length mismatch {proba.shape[0]} vs {len(y)})")
            continue
        name_to_proba[name] = proba
        preds = proba.argmax(axis=1)
        single_f1[name] = float(f1_score(y, preds, average="binary", zero_division=0))

    if not name_to_proba:
        print("No models produced valid probabilities.")
        return

    print("\n--- Per-model F1 (training, argmax of own proba) ---")
    for n in sorted(single_f1.keys(), key=lambda k: (-single_f1[k], k)):
        print(f"  {n}: {single_f1[n]:.4f}")

    best_seed = max(single_f1.keys(), key=lambda k: (single_f1[k], -len(k), k))
    selected = [best_seed]
    remaining = [n for n in name_to_proba if n != best_seed]
    current_f1 = soft_vote_f1([name_to_proba[best_seed]], y)

    print("\n--- Greedy soft voting (mean of class probabilities) ---")
    print(
        f"Step 1: seed {best_seed} — training F1 = {current_f1:.4f} "
        f"(best single-model F1 = {single_f1[best_seed]:.4f})"
    )

    while remaining:
        best_name = None
        best_f = -1.0
        for cand in remaining:
            trial_f1 = soft_vote_f1([name_to_proba[m] for m in selected + [cand]], y)
            if trial_f1 > best_f or (
                abs(trial_f1 - best_f) < 1e-12 and (best_name is None or cand < best_name)
            ):
                best_f = trial_f1
                best_name = cand

        assert best_name is not None
        gain = relative_gain(current_f1, best_f)
        if best_f <= current_f1:
            print(
                f"\nNo candidate improves F1 (best candidate {best_name} "
                f"would give F1 = {best_f:.4f}). Stopping."
            )
            break
        if gain < MIN_RELATIVE_F1_GAIN:
            print(
                f"\nBest add-on {best_name} would yield F1 = {best_f:.4f} "
                f"(relative gain {gain * 100:.2f}% < {MIN_RELATIVE_F1_GAIN * 100:.0f}%). Stopping."
            )
            break

        selected.append(best_name)
        remaining.remove(best_name)
        print(
            f"Step {len(selected)}: add {best_name} — training F1 = {best_f:.4f} "
            f"(relative gain vs previous ensemble: {gain * 100:.2f}%)"
        )
        current_f1 = best_f

    print("\n--- Final ensemble ---")
    print("Models (order = greedy order):", selected)
    print(f"Training F1 (soft vote): {current_f1:.4f}")

    spec_by_display = {s["display_name"]: s for s in specs}

    test_metrics: dict[str, float] | None = None
    if os.path.isfile(TEST_CSV):
        print(f"\nLoading test data: {TEST_CSV}")
        test_df_raw = pd.read_csv(TEST_CSV)
        test_df_raw = test_df_raw.dropna(subset=["label"])
        test_df_raw["label"] = test_df_raw["label"].astype(int)
        y_test = test_df_raw["label"].values
        test_df, subj_t, body_t = _prepare_email_df(test_df_raw)

        print("\n--- Test set: member probabilities (reload per model) ---")
        test_probas: list[np.ndarray] = []
        for name in selected:
            spec = spec_by_display[name]
            print(f"  {name} ...")
            proba = predict_proba_for_spec(spec, test_df, subj_t, body_t, DEVICE)
            if proba.shape[0] != len(y_test):
                raise RuntimeError(
                    f"Test proba length {proba.shape[0]} != labels {len(y_test)} for {name}"
                )
            test_probas.append(proba)

        ens_proba = soft_vote_proba(test_probas)
        test_preds = ens_proba.argmax(axis=1)
        test_metrics = classification_metrics_dict(y_test, test_preds)

        print("\n--- Test metrics (soft-voting ensemble) ---")
        print(
            f"  accuracy:  {test_metrics['accuracy']:.4f}\n"
            f"  precision: {test_metrics['precision']:.4f}\n"
            f"  recall:    {test_metrics['recall']:.4f}\n"
            f"  f1:        {test_metrics['f1']:.4f}\n"
            f"  tp={int(test_metrics['tp'])}, fp={int(test_metrics['fp'])}, "
            f"tn={int(test_metrics['tn'])}, fn={int(test_metrics['fn'])}"
        )
    else:
        print(f"\nTest file not found ({TEST_CSV}); skipping test evaluation.")

    os.makedirs(ENSEMBLE_SAVE_DIR, exist_ok=True)
    artifact: dict[str, Any] = {
        "version": 1,
        "selected_display_names": list(selected),
        "specs_ordered": [_spec_to_abspath(spec_by_display[n]) for n in selected],
        "training_f1_soft_vote": float(current_f1),
        "min_relative_f1_gain": MIN_RELATIVE_F1_GAIN,
        "test_csv": os.path.abspath(TEST_CSV) if os.path.isfile(TEST_CSV) else None,
        "test_metrics": test_metrics,
    }
    joblib.dump(artifact, ENSEMBLE_JOBLIB_PATH)
    print(f"\nSaved ensemble artifact: {ENSEMBLE_JOBLIB_PATH}")

    if test_metrics is not None:
        metrics_lines = [
            "Soft voting ensemble — test set evaluation",
            f"Test CSV: {TEST_CSV}",
            f"Models (order): {selected}",
            "",
            f"accuracy:  {test_metrics['accuracy']:.6f}",
            f"precision: {test_metrics['precision']:.6f}",
            f"recall:    {test_metrics['recall']:.6f}",
            f"f1:        {test_metrics['f1']:.6f}",
            f"tp={int(test_metrics['tp'])}, fp={int(test_metrics['fp'])}, "
            f"tn={int(test_metrics['tn'])}, fn={int(test_metrics['fn'])}",
            "",
            f"Training F1 (soft vote, selection set): {current_f1:.6f}",
            f"Artifact: {ENSEMBLE_JOBLIB_PATH}",
        ]
        with open(ENSEMBLE_TEST_METRICS_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(metrics_lines) + "\n")
        print(f"Wrote metrics file: {ENSEMBLE_TEST_METRICS_PATH}")


if __name__ == "__main__":
    main()
