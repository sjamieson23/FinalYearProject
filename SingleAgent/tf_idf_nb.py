import os

import pandas as pd
from dateutil.utils import today
from google.cloud import storage
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid

from SingleAgent.common import uploadDataToBucket, terminateVM, compute_metrics

train_df = pd.read_csv("Data/all_data_train.csv")
val_df = pd.read_csv("Data/all_data_val.csv")
test_df = pd.read_csv("Data/all_data_test.csv")

train_df["label"] = train_df["label"].astype(int)
val_df["label"] = val_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)

def combine_text(row):
    subj = str(row["subject"]) if not pd.isna(row["subject"]) else ""
    body = str(row["body"]) if not pd.isna(row["body"]) else ""
    return subj.strip() + " [SEP] " + body.strip()


train_df["text"] = train_df.apply(combine_text, axis=1)
val_df["text"] = val_df.apply(combine_text, axis=1)
test_df["text"] = test_df.apply(combine_text, axis=1)

# Features and labels
X_train = train_df["text"]
y_train = train_df["label"]

X_val   = val_df["text"]
y_val   = val_df["label"]

X_test  = test_df["text"]
y_test  = test_df["label"]

def main():
    # Create the TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),  # Use unigrams and bigrams; adds some phrase-level info
        min_df=2,  # Ignore terms that appear in only 1 document (very rare)
        max_df=0.9,  # Ignore extremely common terms (appear in >90% of docs)
        max_features=50000  # Optional cap on vocabulary size for efficiency
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    # Use the fitted TF-IDF to transform validation and test text
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    nb_models = {
        "MultinomialNB": MultinomialNB(),
        "ComplementNB": ComplementNB()
    }

    best_f1 = -1.0
    best_name = None
    best_model = None

    for name, model in nb_models.items():
        model.fit(X_train_tfidf, y_train)  # train only on training set
        y_val_pred = model.predict(X_val_tfidf)  # evaluate on validation set

        f1 = f1_score(y_val, y_val_pred)
        print(f"{name}: val F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    print(f"\nBest NB model: {best_name}, val F1={best_f1:.4f}")
    # Get class probabilities on the test set
    probs = best_model.predict_proba(X_test_tfidf)  # shape: (n_samples, n_classes)

    print("\nTest metrics:")
    probs = best_model.predict_proba(X_test_tfidf)
    metrics = compute_metrics((probs, y_test))

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    results_file = os.path.join("Results/TFIDFBodyAndSubjectNB", "MetricsAndValues.txt")

    # Write metrics and best C to the file
    with open(results_file, "w") as f:
        f.write(f"Best NB type based on validation F1: {best_name}, F1={best_f1:.4f}\n")
        f.write("Test metrics:\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"\nMetrics written to: {results_file}")

if __name__ == "__main__":
    try:
        main()
        uploadDataToBucket("Results/TFIDFBodyAndSubjectNB/MetricsAndValues.txt")
    except Exception as e:
        print(e)
        uploadDataToBucket("Results/TFIDFBodyAndSubjectNB/MetricsAndValues.txt")

    client = storage.Client()
    bucket = client.get_bucket("model-storage-data")
    folder_name = today().strftime("%Y-%m-%d") + "_test1"
    blob = bucket.blob(folder_name)
    blob.upload_from_filename("tf_idf_nb.py")
    terminateVM()