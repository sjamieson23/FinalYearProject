import os

import joblib
import numpy as np
import pandas as pd
from dateutil.utils import today
from google.cloud import storage
from gensim.models import Word2Vec
from sklearn.neural_network import MLPClassifier
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
    return subj.strip() + body.strip()


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

def tokenize_text(text):
    """Tokenize text into words (simple whitespace tokenization)"""
    return str(text).lower().split()

def document_vector(model, doc):
    """Create document vector by averaging word vectors"""
    doc_words = tokenize_text(doc)
    word_vectors = [model.wv[word] for word in doc_words if word in model.wv]
    if len(word_vectors) == 0:
        # Return zero vector if no words found
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def main():
    # Tokenize all texts for Word2Vec training
    print("Tokenizing texts for Word2Vec training...")
    train_sentences = [tokenize_text(text) for text in X_train]
    
    # Train Word2Vec model on training data (optimized for large datasets)
    print("Training Word2Vec model...")
    word2vec_model = Word2Vec(
        sentences=train_sentences,
        vector_size=300,  # Dimensionality of word embeddings
        window=10,  # Larger context window for better semantic capture
        min_count=5,  # Higher threshold for large datasets - filters rare/noisy words
        workers=-1,  # Use all available CPU cores for parallelization
        sg=1,  # Skip-gram (1) often performs better than CBOW (0) on large datasets
        epochs=20,  # More epochs for better convergence on large datasets
        negative=5,  # Negative sampling for efficiency (default is 5)
        ns_exponent=0.75,  # Negative sampling exponent (standard value)
        sample=1e-4,  # Downsample frequent words (helps with large vocabularies)
        alpha=0.025,  # Initial learning rate
        min_alpha=0.0001  # Minimum learning rate
    )
    
    # Create document embeddings by averaging word vectors
    print("Creating document embeddings...")
    X_train_embeddings = np.array([document_vector(word2vec_model, doc) for doc in X_train])
    X_val_embeddings = np.array([document_vector(word2vec_model, doc) for doc in X_val])
    X_test_embeddings = np.array([document_vector(word2vec_model, doc) for doc in X_test])

    param_grid = {
        "hidden_layer_sizes": [(200,), (300,), (200, 100), (300, 150), (500, 250)],
        "alpha": [0.00001, 0.0001, 0.001],  # Lower regularization for large datasets
        "learning_rate_init": [0.0001, 0.001, 0.01],
        "batch_size": ['auto', 200, 500]  # Batch size for efficient training on large data
    }

    best_f1 = -1.0
    best_params = None
    best_model = None
    best_word2vec = None

    for params in ParameterGrid(param_grid):
        hidden_layer_sizes = params["hidden_layer_sizes"]
        alpha = params["alpha"]
        learning_rate_init = params["learning_rate_init"]
        batch_size = params["batch_size"]
        # Define MLP classifier
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=1000,  # Increased for large datasets - MLPs need more iterations
            random_state=42,
            early_stopping=True,  # Stop early if validation score doesn't improve
            validation_fraction=0.1,  # Use 10% of training data for validation
            n_iter_no_change=20,  # More patience for large datasets
            tol=1e-4  # Tolerance for optimization
        )
        # train only on training set
        model.fit(X_train_embeddings, y_train)

        # evaluate on validation set
        y_val_pred = model.predict(X_val_embeddings)
        f1 = f1_score(y_val, y_val_pred)

        print(f"hidden_layer_sizes={hidden_layer_sizes}, alpha={alpha}, learning_rate_init={learning_rate_init}, batch_size={batch_size}: val F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_model = model
            best_word2vec = word2vec_model

    print(f"\nBest params based on validation F1: {best_params}, F1={best_f1:.4f}")
    # Get class probabilities on the test set
    probs = best_model.predict_proba(X_test_embeddings)  # shape: (n_samples, n_classes)

    # Use compute_metrics with "logits" = probs and labels = y_test
    metrics = compute_metrics((probs, y_test))

    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    results_dir = "Results/Word2VecBodyAndSubjectMLP"
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, "MetricsAndValues.txt")

    # Write metrics and best params to the file
    with open(results_file, "w") as f:
        f.write(f"Best params based on validation F1: {best_params}, F1={best_f1:.4f}\n")
        f.write("Test metrics:\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"\nMetrics written to: {results_file}")
    
    # Save the trained model and Word2Vec model
    model_file = os.path.join(results_dir, "word2vec_mlp_model.joblib")
    word2vec_file = os.path.join(results_dir, "word2vec_model.model")
    
    joblib.dump(best_model, model_file)
    best_word2vec.save(word2vec_file)
    print(f"\nModel saved to: {model_file}")
    print(f"Word2Vec model saved to: {word2vec_file}")

if __name__ == "__main__":
    try:
        main()
        # Upload metrics, model, and Word2Vec model to bucket
        uploadDataToBucket("Results/Word2VecBodyAndSubjectMLP")
    except Exception as e:
        print(e)
        # Try to upload even if there was an error
        uploadDataToBucket("Results/Word2VecBodyAndSubjectMLP")

    client = storage.Client()
    bucket = client.get_bucket("model-storage-data")
    folder_name = today().strftime("%Y-%m-%d") + "_test1"
    blob = bucket.blob(folder_name)
    blob.upload_from_filename("word2vec_mlp.py")
    terminateVM()

