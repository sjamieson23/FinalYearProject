"""
Stacked ensemble: base models produce class probabilities; a logistic regression
meta-learner is fit on P(class=1) from each base model.

Default base models (folder names), resolved under TopModels/ then SavedModels/:
  BertBody_2_Epochs, BertSubj_3_Epochs, BertBodyAndSubj_3_Epochs,
  Word2VecBodyAndSubjectMLP, TFIDFBodyAndSubjectDT, TFIDFBodyAndSubjectMLP

Train on Data/ensemble_training_data.csv, evaluate on Data/ensemble_testing_data.csv,
print metrics, save joblib for reuse (load_stacking_ensemble / predict_stacked_proba).

Run from project root: python -m EnsembleAgent.Stacking
"""

from __future__ import annotations

import glob
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from SoftVoting import (
    DEVICE,
    _bert_kind,
    _has_bert_weights,
    _prepare_email_df,
    _spec_to_abspath,
    classification_metrics_dict,
    predict_proba_for_spec,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(SCRIPT_DIR, "Data", "ensemble_training_data.csv")
TEST_CSV = os.path.join(SCRIPT_DIR, "Data", "ensemble_testing_data.csv")
TOP_MODELS = os.path.join(SCRIPT_DIR, "TopModels")
SAVED_MODELS = os.path.join(SCRIPT_DIR, "SavedModels")
STACK_SAVE_DIR = os.path.join(SCRIPT_DIR, "SavedStackingEnsemble")
STACK_JOBLIB_PATH = os.path.join(STACK_SAVE_DIR, "stacking_ensemble.joblib")
STACK_METRICS_PATH = os.path.join(STACK_SAVE_DIR, "stacking_test_metrics.txt")

# Column order in meta feature matrix matches this list.
BASE_MODEL_FOLDERS = [
    "BertBody_2_Epochs",
    "BertSubj_3_Epochs",
    "BertBodyAndSubj_3_Epochs",
    "Word2VecBodyAndSubjectMLP",
    "TFIDFBodyAndSubjectDT",
    "TFIDFBodyAndSubjectMLP",
]


def resolve_bundle_dir(folder_name: str) -> str:
    for root in (TOP_MODELS, SAVED_MODELS):
        p = os.path.join(root, folder_name)
        if os.path.isdir(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        f"Model folder {folder_name!r} not found under {TOP_MODELS} or {SAVED_MODELS}"
    )


def build_base_spec(bundle_path: str) -> dict[str, Any]:
    """Spec dict compatible with SoftVoting.predict_proba_for_spec."""
    path = os.path.abspath(bundle_path)
    name = os.path.basename(path)
    model_dir = os.path.join(path, "model")
    cfg = os.path.join(model_dir, "config.json")

    bkind = _bert_kind(name)
    if bkind and os.path.isfile(cfg) and _has_bert_weights(model_dir):
        return {"kind": bkind, "path": path}

    tfidf_models = glob.glob(os.path.join(path, "tfidf_*_model.joblib"))
    vec_path = os.path.join(path, "tfidf_vectorizer.joblib")
    if len(tfidf_models) == 1 and os.path.isfile(vec_path):
        return {
            "kind": "tfidf",
            "path": path,
            "clf_path": tfidf_models[0],
            "vectorizer_path": vec_path,
        }

    w2v_clfs = sorted(glob.glob(os.path.join(path, "word2vec_*_model.joblib")))
    w2v_path = os.path.join(path, "word2vec_model.model")
    if w2v_clfs and os.path.isfile(w2v_path):
        return {
            "kind": "w2v",
            "path": path,
            "clf_path": w2v_clfs[0],
            "w2v_path": w2v_path,
        }

    raise ValueError(f"Unrecognized model bundle layout: {path}")


def collect_base_probas(
    specs: list[dict[str, Any]],
    data_df: pd.DataFrame,
    subj: pd.Series,
    body: pd.Series,
    device: torch.device,
    label: str,
    verbose: bool = True,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for i, spec in enumerate(specs):
        if verbose:
            print(f"  [{i + 1}/{len(specs)}] {label} — {spec.get('path', '')} ...")
        proba = predict_proba_for_spec(spec, data_df, subj, body, device)
        out.append(proba)
    return out


def meta_features_from_probas(probas: list[np.ndarray]) -> np.ndarray:
    return np.column_stack([p[:, 1] for p in probas])


def load_stacking_ensemble(path: str = STACK_JOBLIB_PATH) -> dict[str, Any]:
    return joblib.load(path)


def predict_stacked_proba(
    artifact: dict[str, Any],
    df: pd.DataFrame,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = DEVICE
    data_df, subj, body = _prepare_email_df(df)
    probas = collect_base_probas(
        artifact["specs_ordered"],
        data_df,
        subj,
        body,
        device,
        "predict",
        verbose=False,
    )
    X_meta = meta_features_from_probas(probas)
    return artifact["meta_learner"].predict_proba(X_meta)


def predict_stacked_labels(
    artifact: dict[str, Any],
    df: pd.DataFrame,
    device: torch.device | None = None,
) -> np.ndarray:
    return predict_stacked_proba(artifact, df, device=device).argmax(axis=1)


def main():
    specs: list[dict[str, Any]] = []
    resolved_paths: list[str] = []
    for name in BASE_MODEL_FOLDERS:
        bundle = resolve_bundle_dir(name)
        resolved_paths.append(bundle)
        specs.append(build_base_spec(bundle))
        print(f"Resolved {name} -> {bundle}")

    print(f"\nLoading training data: {TRAIN_CSV}")
    train_df = pd.read_csv(TRAIN_CSV)
    train_df = train_df.dropna(subset=["label"])
    train_df["label"] = train_df["label"].astype(int)
    y_train = train_df["label"].values
    train_df, subj_tr, body_tr = _prepare_email_df(train_df)

    print("\nBase model predict_proba (train)...")
    train_probas = collect_base_probas(specs, train_df, subj_tr, body_tr, DEVICE, "train")
    X_meta_train = meta_features_from_probas(train_probas)
    if X_meta_train.shape[0] != len(y_train):
        raise RuntimeError("Train meta feature row count mismatch.")

    meta = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        random_state=42,
    )
    meta.fit(X_meta_train, y_train)
    train_preds = meta.predict(X_meta_train)
    train_metrics = classification_metrics_dict(y_train, train_preds)

    print("\n--- Training metrics (stacked, same data as meta fit) ---")
    print(
        f"  accuracy:  {train_metrics['accuracy']:.4f}\n"
        f"  precision: {train_metrics['precision']:.4f}\n"
        f"  recall:    {train_metrics['recall']:.4f}\n"
        f"  f1:        {train_metrics['f1']:.4f}"
    )

    test_metrics: dict[str, float] | None = None
    if os.path.isfile(TEST_CSV):
        print(f"\nLoading test data: {TEST_CSV}")
        test_df = pd.read_csv(TEST_CSV)
        test_df = test_df.dropna(subset=["label"])
        test_df["label"] = test_df["label"].astype(int)
        y_test = test_df["label"].values
        test_df, subj_te, body_te = _prepare_email_df(test_df)

        print("\nBase model predict_proba (test)...")
        test_probas = collect_base_probas(specs, test_df, subj_te, body_te, DEVICE, "test")
        X_meta_test = meta_features_from_probas(test_probas)
        test_preds = meta.predict(X_meta_test)
        test_metrics = classification_metrics_dict(y_test, test_preds)

        print("\n--- Test metrics (stacked ensemble) ---")
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

    os.makedirs(STACK_SAVE_DIR, exist_ok=True)
    artifact: dict[str, Any] = {
        "version": 1,
        "meta_learner": meta,
        "base_model_folders": list(BASE_MODEL_FOLDERS),
        "base_bundle_paths": [os.path.abspath(p) for p in resolved_paths],
        "specs_ordered": [_spec_to_abspath(s) for s in specs],
        "meta_features": "P(class=1) per base model; columns in BASE_MODEL_FOLDERS order",
        "train_csv": os.path.abspath(TRAIN_CSV),
        "train_metrics": train_metrics,
        "test_csv": os.path.abspath(TEST_CSV) if os.path.isfile(TEST_CSV) else None,
        "test_metrics": test_metrics,
    }
    joblib.dump(artifact, STACK_JOBLIB_PATH)
    print(f"\nSaved stacking artifact: {STACK_JOBLIB_PATH}")

    if test_metrics is not None:
        lines = [
            "Stacking ensemble (logistic regression meta-learner) — test evaluation",
            f"Test CSV: {TEST_CSV}",
            f"Base models (feature order): {BASE_MODEL_FOLDERS}",
            "",
            f"accuracy:  {test_metrics['accuracy']:.6f}",
            f"precision: {test_metrics['precision']:.6f}",
            f"recall:    {test_metrics['recall']:.6f}",
            f"f1:        {test_metrics['f1']:.6f}",
            f"tp={int(test_metrics['tp'])}, fp={int(test_metrics['fp'])}, "
            f"tn={int(test_metrics['tn'])}, fn={int(test_metrics['fn'])}",
            "",
            f"Train CSV: {TRAIN_CSV}",
            f"Train F1 (in-sample meta): {train_metrics['f1']:.6f}",
            f"Artifact: {STACK_JOBLIB_PATH}",
        ]
        with open(STACK_METRICS_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote metrics file: {STACK_METRICS_PATH}")


if __name__ == "__main__":
    main()
