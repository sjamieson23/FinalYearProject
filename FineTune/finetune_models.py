from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from gensim.models import Word2Vec
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

# -----------------------------------------------------------------------------
# Project imports (expects cwd or sys.path to include project root)
# -----------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from SingleAgent.common import compute_metrics  # noqa: E402


# =============================================================================
# Configuration — edit this block
# =============================================================================

@dataclass
class FinetuneConfig:
    """All user-tunable settings in one place."""

    # Directories (absolute or relative to project root)
    saved_models_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "FineTune" / "SavedModelsFINAL"
    )
    training_csv: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "FineTune" / "Data" / "finetune_training_data.csv"
    )
    testing_csv: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "FineTune" / "Data" / "finetune_testing_data.csv"
    )
    output_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "FineTune" / "FinetunedOutputs"
    )

    # Restrict which subfolders of saved_models_dir to run (None = every model)
    only_models: Optional[List[str]] = field(default_factory=lambda: ["BertBody", "BertBodyAndSubj", "BertSubj"])
    skip_models: List[str] = field(default_factory=lambda: [])

    # BERT / Trainer (defaults aligned with SingleAgent/bert_body_and_subj.py)
    bert_epochs: int = 2
    bert_learning_rate: float = 2e-5
    bert_train_batch_size: int = 10
    bert_eval_batch_size: int = 12
    bert_max_length: int = 512
    bert_weight_decay: float = 0.01
    bert_fp16: bool = True
    bert_seed: int = 1
    bert_logging_steps: int = 1000

    # Sklearn heads on fixed TF-IDF (same vocabulary as the saved vectorizer;
    # initial TF-IDF fit is in SingleAgent/tf_idf_*.py — we do not refit the vectorizer here.)
    mlp_extra_max_iter: int = 200

    # Word2Vec: initial training is on the large corpus in SingleAgent/word2vec_*.py.
    # Default is classifier-only fine-tune (frozen embeddings, no vocab growth) so a small
    # finetune CSV cannot distort the skip-gram model or add spurious rare words.
    w2v_finetune_embeddings: bool = False
    # If True: extra passes only on words already in the saved vocabulary (no new tokens)
    # unless w2v_allow_vocab_update is True.
    w2v_finetune_epochs: int = 1
    w2v_allow_vocab_update: bool = False
    # If vocab update is enabled, use a high min_count (initial training used min_count=5).
    w2v_new_word_min_count: int = 5
    # Cap learning rate for optional embedding updates (gentle fine-tune on small data).
    w2v_finetune_start_alpha: float = 0.005
    w2v_finetune_end_alpha: float = 0.0001

    # Misc
    random_state: int = 42


CONFIG = FinetuneConfig()


# =============================================================================
# Data loading & text builders (match SingleAgent: tf_idf_*.py, word2vec_*.py, bert_*.py)
# =============================================================================

REQUIRED_COLUMNS = ("subject", "body", "label")


def load_labeled_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = df.copy()
    df["label"] = df["label"].astype(int)
    return df


def text_tfidf(row: pd.Series) -> str:
    subj = str(row["subject"]) if not pd.isna(row["subject"]) else ""
    body = str(row["body"]) if not pd.isna(row["body"]) else ""
    return subj.strip() + body.strip()


def text_w2v(row: pd.Series) -> str:
    subj = str(row["subject"]) if not pd.isna(row["subject"]) else ""
    body = str(row["body"]) if not pd.isna(row["body"]) else ""
    return subj.strip() + " [SEP] " + body.strip()


def text_bert_body(row: pd.Series) -> str:
    body = str(row["body"]) if not pd.isna(row["body"]) else ""
    return body.strip()


def text_bert_subj(row: pd.Series) -> str:
    subj = str(row["subject"]) if not pd.isna(row["subject"]) else ""
    return subj.strip()


def text_bert_body_subj(row: pd.Series) -> str:
    subj = str(row["subject"]) if not pd.isna(row["subject"]) else ""
    body = str(row["body"]) if not pd.isna(row["body"]) else ""
    return subj.strip() + " [SEP] " + body.strip()


BERT_TEXT_MODE: dict[str, Callable[[pd.Series], str]] = {
    "BertBody": text_bert_body,
    "BertSubj": text_bert_subj,
    "BertBodyAndSubj": text_bert_body_subj,
}


# =============================================================================
# Model discovery
# =============================================================================

def list_model_dirs(saved_root: Path) -> List[Path]:
    if not saved_root.is_dir():
        raise FileNotFoundError(f"Saved models directory not found: {saved_root}")
    dirs = sorted(p for p in saved_root.iterdir() if p.is_dir() and not p.name.startswith("."))
    return dirs


def classify_saved_model(model_dir: Path) -> str:
    """Return 'bert', 'tfidf', 'word2vec', or 'unknown'."""
    if (model_dir / "model" / "config.json").is_file():
        cfg = json.loads((model_dir / "model" / "config.json").read_text())
        if cfg.get("model_type") == "bert":
            return "bert"
    tfidf_models = list(model_dir.glob("tfidf_*_model.joblib"))
    if tfidf_models and (model_dir / "tfidf_vectorizer.joblib").is_file():
        return "tfidf"
    w2v_models = list(model_dir.glob("word2vec_*_model.joblib"))
    if w2v_models and (model_dir / "word2vec_model.model").is_file():
        return "word2vec"
    return "unknown"


def should_run_model(name: str, cfg: FinetuneConfig) -> bool:
    if cfg.only_models is not None and name not in cfg.only_models:
        return False
    if name in cfg.skip_models:
        return False
    return True


# =============================================================================
# Metrics helpers
# =============================================================================

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }


def write_metrics_txt(path: Path, sections: Sequence[Tuple[str, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for title, metrics in sections:
        lines.append(f"=== {title} ===")
        for k, v in metrics.items():
            lines.append(f"{k}: {v:.4f}")
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


# =============================================================================
# BERT fine-tuning
# =============================================================================

def bert_paths(model_dir: Path) -> Tuple[Path, Path]:
    model_path = model_dir / "model"
    tok_path = model_dir / "tokenizer"
    if not tok_path.is_dir():
        tok_path = model_path
    return model_path, tok_path


def prepare_bert_dataset(
    df: pd.DataFrame,
    text_fn: Callable[[pd.Series], str],
    tokenizer: BertTokenizer,
    max_length: int,
) -> Dataset:
    work = df.copy()
    work["text"] = work.apply(text_fn, axis=1)
    ds = Dataset.from_pandas(work[["text", "label"]], preserve_index=False)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    ds = ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds


def finetune_bert(
    name: str,
    model_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: FinetuneConfig,
    out_dir: Path,
) -> None:
    text_fn = BERT_TEXT_MODE.get(name)
    if text_fn is None:
        raise ValueError(f"No BERT text mode registered for folder name: {name!r}")

    model_path, tokenizer_path = bert_paths(model_dir)
    tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))
    model = BertForSequenceClassification.from_pretrained(str(model_path), num_labels=2)

    train_ds = prepare_bert_dataset(train_df, text_fn, tokenizer, cfg.bert_max_length)
    # Same file for train and eval (per project requirement)
    eval_ds = prepare_bert_dataset(train_df, text_fn, tokenizer, cfg.bert_max_length)
    test_ds = prepare_bert_dataset(test_df, text_fn, tokenizer, cfg.bert_max_length)

    run_dir = out_dir / name / "trainer_output"
    run_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=cfg.bert_learning_rate,
        per_device_train_batch_size=cfg.bert_train_batch_size,
        per_device_eval_batch_size=cfg.bert_eval_batch_size,
        num_train_epochs=cfg.bert_epochs,
        weight_decay=cfg.bert_weight_decay,
        logging_steps=cfg.bert_logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=False,
        seed=cfg.bert_seed,
        fp16=cfg.bert_fp16 and torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # --- "Validation" metrics: same training CSV, after training (honest in-distribution check)
    val_metrics = trainer.evaluate(eval_ds)
    val_scalar = {k.replace("eval_", ""): float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))}

    # --- Final test: one forward pass, no weight updates
    pred = trainer.predict(test_ds)
    test_metrics = compute_metrics((pred.predictions, pred.label_ids))

    write_metrics_txt(
        out_dir / name / "MetricsAndValues.txt",
        [
            ("train_csv_eval_after_finetune", val_scalar),
            ("held_out_test_no_feedback", test_metrics),
        ],
    )

    save_model = out_dir / name / "model"
    save_tok = out_dir / name / "tokenizer"
    save_model.mkdir(parents=True, exist_ok=True)
    save_tok.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_model))
    tokenizer.save_pretrained(str(save_tok))


# =============================================================================
# TF-IDF + sklearn classifier
# =============================================================================

def finetune_tfidf_sklearn(
    name: str,
    model_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: FinetuneConfig,
    out_dir: Path,
) -> None:
    vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")
    model_files = list(model_dir.glob("tfidf_*_model.joblib"))
    if len(model_files) != 1:
        raise ValueError(f"Expected exactly one tfidf_*_model.joblib in {model_dir}, got {model_files}")
    estimator = joblib.load(model_files[0])

    train_texts = train_df.apply(text_tfidf, axis=1)
    test_texts = test_df.apply(text_tfidf, axis=1)
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    X_train = vectorizer.transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Continue training in the *fixed* TF-IDF feature space (same vocabulary as the saved model)
    updated = _continue_sklearn_estimator(estimator, X_train, y_train, cfg)

    train_pred = updated.predict(X_train)
    test_pred = updated.predict(X_test)

    train_metrics = classification_metrics(y_train, train_pred)
    test_metrics = classification_metrics(y_test, test_pred)

    out_sub = out_dir / name
    out_sub.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out_sub / "tfidf_vectorizer.joblib")
    joblib.dump(updated, model_files[0].name)

    write_metrics_txt(
        out_sub / "MetricsAndValues.txt",
        [
            ("train_csv_after_finetune", train_metrics),
            ("held_out_test_no_feedback", test_metrics),
        ],
    )


def _continue_sklearn_estimator(estimator, X_train, y_train, cfg: FinetuneConfig):
    est = estimator

    if isinstance(est, MLPClassifier):
        est.set_params(warm_start=True, max_iter=est.max_iter + cfg.mlp_extra_max_iter)
        est.fit(X_train, y_train)
        return est

    if isinstance(est, MultinomialNB):
        # single pass; counts are replaced on the finetune batch
        est.partial_fit(X_train, y_train, classes=np.array([0, 1]))
        return est

    if isinstance(est, BernoulliNB):
        est.partial_fit(X_train, y_train, classes=np.array([0, 1]))
        return est

    # LogisticRegression: warm_start keeps previous coefficients as the optimisation start.
    # Tree models: plain refit in the same TF-IDF / embedding space (no incremental API).
    if type(est).__name__ == "LogisticRegression":
        est.set_params(warm_start=True)

    est.fit(X_train, y_train)
    return est


# =============================================================================
# Word2Vec + sklearn classifier
# =============================================================================

def tokenize_w2v_corpus(text: str) -> List[str]:
    return str(text).lower().split()


def document_vector(w2v: Word2Vec, doc: str) -> np.ndarray:
    """Same rule as SingleAgent/word2vec_mlp.py ``document_vector`` (mean pooling)."""
    words = tokenize_w2v_corpus(doc)
    vectors = [w2v.wv[w] for w in words if w in w2v.wv]
    if not vectors:
        return np.zeros(w2v.vector_size, dtype=np.float64)
    return np.mean(vectors, axis=0)


def _maybe_finetune_word2vec_embeddings(
    w2v: Word2Vec, sentences: List[List[str]], cfg: FinetuneConfig
) -> None:
    if not cfg.w2v_finetune_embeddings:
        return
    if cfg.w2v_allow_vocab_update:
        w2v.build_vocab(sentences, update=True, min_count=cfg.w2v_new_word_min_count)
    w2v.train(
        sentences,
        total_examples=len(sentences),
        epochs=cfg.w2v_finetune_epochs,
        start_alpha=cfg.w2v_finetune_start_alpha,
        end_alpha=cfg.w2v_finetune_end_alpha,
    )


def finetune_word2vec_sklearn(
    name: str,
    model_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: FinetuneConfig,
    out_dir: Path,
) -> None:
    w2v = Word2Vec.load(str(model_dir / "word2vec_model.model"))
    model_files = list(model_dir.glob("word2vec_*_model.joblib"))
    if len(model_files) != 1:
        raise ValueError(f"Expected one word2vec_*_model.joblib in {model_dir}, got {model_files}")
    estimator = joblib.load(model_files[0])

    train_docs = train_df.apply(text_w2v, axis=1).tolist()
    test_docs = test_df.apply(text_w2v, axis=1).tolist()
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    sentences = [tokenize_w2v_corpus(t) for t in train_docs]
    _maybe_finetune_word2vec_embeddings(w2v, sentences, cfg)

    X_train = np.vstack([document_vector(w2v, d) for d in train_docs])
    X_test = np.vstack([document_vector(w2v, d) for d in test_docs])

    updated = _continue_sklearn_estimator(estimator, X_train, y_train, cfg)

    train_pred = updated.predict(X_train)
    test_pred = updated.predict(X_test)
    train_metrics = classification_metrics(y_train, train_pred)
    test_metrics = classification_metrics(y_test, test_pred)

    out_sub = out_dir / name
    out_sub.mkdir(parents=True, exist_ok=True)
    w2v.save(str(out_sub / "word2vec_model.model"))
    joblib.dump(updated, model_files[0].name)

    write_metrics_txt(
        out_sub / "MetricsAndValues.txt",
        [
            ("train_csv_after_finetune", train_metrics),
            ("held_out_test_no_feedback", test_metrics),
        ],
    )


# =============================================================================
# Main
# =============================================================================

def run_all(cfg: FinetuneConfig = CONFIG) -> None:
    train_df = load_labeled_csv(cfg.training_csv)
    test_df = load_labeled_csv(cfg.testing_csv)

    print(f"Training rows: {len(train_df)}  ({cfg.training_csv})")
    print(f"Test rows:     {len(test_df)}  ({cfg.testing_csv})")
    print(f"Scanning:      {cfg.saved_models_dir}\n")

    for model_dir in list_model_dirs(cfg.saved_models_dir):
        name = model_dir.name
        if not should_run_model(name, cfg):
            print(f"[skip] {name}")
            continue

        kind = classify_saved_model(model_dir)
        print(f"[run] {name}  ({kind})")

        if kind == "unknown":
            print(f"  ! Unrecognised layout, skipping: {model_dir}")
            continue

        try:
            if kind == "bert":
                finetune_bert(name, model_dir, train_df, test_df, cfg, cfg.output_dir)
            elif kind == "tfidf":
                finetune_tfidf_sklearn(name, model_dir, train_df, test_df, cfg, cfg.output_dir)
            elif kind == "word2vec":
                finetune_word2vec_sklearn(name, model_dir, train_df, test_df, cfg, cfg.output_dir)
        except Exception as e:
            print(f"  ! Failed: {e}")
            raise

        print(f"  -> wrote outputs under {cfg.output_dir / name}\n")


if __name__ == "__main__":
    run_all(CONFIG)
