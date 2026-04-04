# Compare all models under FinetunedOutputs/ and SavedModels/ on the test set;
# report metrics, uniquely correct counts, an UpSet plot of correct-prediction overlaps,
# and a pairwise prediction-agreement heatmap.
import gc
import glob
import os
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from transformers import BertForSequenceClassification, BertTokenizer

try:
    from upsetplot import UpSet, from_contents
except ImportError as e:
    UpSet = None
    from_contents = None
    _UPSETPLOT_IMPORT_ERROR = e
else:
    _UPSETPLOT_IMPORT_ERROR = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CSV = os.path.join(SCRIPT_DIR, "Data", "ensemble_testing_data.csv")
MODEL_ROOTS = [
    os.path.join(SCRIPT_DIR, "TopModels"),
    #os.path.join(SCRIPT_DIR, "FinetunedOutputs"),
    #os.path.join(SCRIPT_DIR, "SavedModels"),
]
UPSET_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "model_comparison_upset.png")
PAIRWISE_AGREEMENT_PNG = os.path.join(SCRIPT_DIR, "model_pairwise_agreement.png")
PAIRWISE_AGREEMENT_CSV = os.path.join(SCRIPT_DIR, "model_pairwise_agreement.csv")

MAX_LENGTH = 512
BATCH_SIZE = 32
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
    """Return list of dicts: display_name, kind, path (model bundle root)."""
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
    model.to(device)
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encodings = {k: v.to(device) for k, v in encodings.items()}
            outputs = model(**encodings)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
    return np.concatenate(all_preds, axis=0)


def predict_for_spec(spec, test_df, subj_series, body_series, device):
    kind = spec["kind"]
    path = spec["path"]

    if kind == "bert_bs":
        d = os.path.join(path, "model")
        tokenizer = BertTokenizer.from_pretrained(d)
        model = BertForSequenceClassification.from_pretrained(d, num_labels=2)
        texts = (subj_series.str.strip() + " [SEP] " + body_series.str.strip()).tolist()
        preds = bert_predict_labels(texts, tokenizer, model, device)
        del model, tokenizer
        _torch_release()
        return preds

    if kind == "bert_body":
        tok_dir = os.path.join(path, "tokenizer")
        mod_dir = os.path.join(path, "model")
        tokenizer = BertTokenizer.from_pretrained(tok_dir)
        model = BertForSequenceClassification.from_pretrained(mod_dir, num_labels=2)
        texts = body_series.str.strip().tolist()
        preds = bert_predict_labels(texts, tokenizer, model, device)
        del model, tokenizer
        _torch_release()
        return preds

    if kind == "bert_subj":
        tok_dir = os.path.join(path, "tokenizer")
        mod_dir = os.path.join(path, "model")
        tokenizer = BertTokenizer.from_pretrained(tok_dir)
        model = BertForSequenceClassification.from_pretrained(mod_dir, num_labels=2)
        texts = subj_series.str.strip().tolist()
        preds = bert_predict_labels(texts, tokenizer, model, device)
        del model, tokenizer
        _torch_release()
        return preds

    if kind == "tfidf":
        clf = joblib.load(spec["clf_path"])
        vectorizer = joblib.load(spec["vectorizer_path"])
        X = vectorizer.transform(test_df["text_tfidf"])
        return clf.predict(X)

    if kind == "w2v":
        clf = joblib.load(spec["clf_path"])
        w2v = Word2Vec.load(spec["w2v_path"])
        X = np.vstack([document_vector(w2v, doc) for doc in test_df["text_w2v"]])
        return clf.predict(X)

    raise ValueError(f"Unknown spec kind: {kind}")


def compute_metrics(y_true, preds):
    tp = fp = tn = fn = 0
    for i in range(len(y_true)):
        pred = int(preds[i])
        true = int(y_true[i])
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 0:
            tn += 1
        elif pred == 0 and true == 1:
            fn += 1
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_upset_plot(model_preds, y_true, out_path):
    if UpSet is None or from_contents is None:
        raise ImportError(
            "Install upsetplot to generate the UpSet figure: pip install upsetplot"
        ) from _UPSETPLOT_IMPORT_ERROR

    n = len(y_true)
    indices = np.arange(n, dtype=np.int64)
    y = np.asarray(y_true).astype(int)
    contents = {}
    for name, preds in model_preds.items():
        p = np.asarray(preds).astype(int)
        mask = p == y
        contents[name] = indices[mask]

    upset_data = from_contents(contents)
    n_models = len(model_preds)
    fig_w = max(12.0, 0.55 * n_models + 6)
    fig_h = max(7.0, 0.35 * n_models + 5)
    fig = plt.figure(figsize=(fig_w, fig_h))
    UpSet(upset_data, subset_size="count", show_counts=True).plot(fig=fig)
    fig.suptitle("Test-set intersections: models that predicted correctly (per sample)", y=1.02)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def pairwise_prediction_agreement_matrix(model_preds: dict) -> tuple[np.ndarray, list]:
    """Fraction of test rows where two models assign the same predicted label."""
    names = list(model_preds.keys())
    if not names:
        return np.zeros((0, 0)), names
    P = np.vstack([np.asarray(model_preds[n]).astype(int) for n in names])
    agreement = (P[:, np.newaxis, :] == P[np.newaxis, :, :]).mean(axis=2, dtype=np.float64)
    return agreement, names


def save_pairwise_agreement_heatmap(model_preds: dict, out_png: str, out_csv: Optional[str]):
    agreement, names = pairwise_prediction_agreement_matrix(model_preds)
    m = len(names)
    if m == 0:
        return

    df = pd.DataFrame(agreement, index=names, columns=names)
    if out_csv:
        df.to_csv(out_csv)

    fig_w = max(9.0, 0.42 * m + 5)
    fig_h = max(8.0, 0.42 * m + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(agreement, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(m))
    tick_fs = max(5, min(9, 130 // max(m, 1)))
    ax.set_xticklabels(names, rotation=90, ha="right", fontsize=tick_fs)
    ax.set_yticklabels(names, fontsize=tick_fs)
    ax.set_xlabel("Model")
    ax.set_ylabel("Model")
    ax.set_title("Pairwise agreement: fraction of test samples with identical predicted label")

    if m <= 18:
        for i in range(m):
            for j in range(m):
                ax.text(
                    j,
                    i,
                    f"{agreement[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if agreement[i, j] < 0.5 else "black",
                    fontsize=max(4, tick_fs - 1),
                )

    ax.set_xticks(np.arange(m) - 0.5, minor=True)
    ax.set_yticks(np.arange(m) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.25, alpha=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Agreement")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    specs = discover_model_specs()
    if not specs:
        print("No models found under FinetunedOutputs/ or SavedModels/.")
        return

    print(f"Discovered {len(specs)} model bundle(s):")
    for s in specs:
        print(f"  - {s['display_name']} ({s['kind']})")

    print("\nLoading test data...")
    test_df = pd.read_csv(TEST_CSV)
    print("Length of test data: ", len(test_df))
    test_df["label"] = test_df["label"].astype(int)
    y_true = test_df["label"].values

    test_df["text_tfidf"] = test_df.apply(combine_text_tfidf, axis=1)
    test_df["text_w2v"] = test_df.apply(combine_text_w2v, axis=1)
    subj_series = test_df["subject"].fillna("").astype(str)
    body_series = test_df["body"].fillna("").astype(str)

    model_preds = {}
    for spec in specs:
        name = spec["display_name"]
        print(f"\nRunning {name}...")
        preds = predict_for_spec(spec, test_df, subj_series, body_series, DEVICE)
        model_preds[name] = preds

    model_metrics = {
        name: compute_metrics(y_true, preds) for name, preds in model_preds.items()
    }

    unique_counts = {name: 0 for name in model_preds.keys()}
    for i in range(len(y_true)):
        correct_models = [
            name for name, preds in model_preds.items() if int(preds[i]) == int(y_true[i])
        ]
        if len(correct_models) == 1:
            unique_counts[correct_models[0]] += 1

    print("\nPer-model metrics:")
    for name, metrics in model_metrics.items():
        print(
            f"{name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"tp={metrics['tp']}, fp={metrics['fp']}, tn={metrics['tn']}, fn={metrics['fn']}"
        )

    print("\nUniquely correct predictions per model:")
    for name, count in unique_counts.items():
        print(f"{name}: {count}")

    save_pairwise_agreement_heatmap(model_preds, PAIRWISE_AGREEMENT_PNG, PAIRWISE_AGREEMENT_CSV)
    print(f"\nPairwise agreement heatmap: {PAIRWISE_AGREEMENT_PNG}")
    print(f"Pairwise agreement matrix (CSV): {PAIRWISE_AGREEMENT_CSV}")

    try:
        save_upset_plot(model_preds, y_true, UPSET_OUTPUT_PATH)
        print(f"\nUpSet plot saved to: {UPSET_OUTPUT_PATH}")
    except ImportError as err:
        print(f"\nSkipping UpSet plot: {err}")


if __name__ == "__main__":
    main()
