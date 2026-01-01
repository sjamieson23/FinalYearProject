import os
import time
from datetime import datetime

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
    script_start_time = time.time()
    
    # Tokenize all texts for Word2Vec training
    print(f"[{datetime.now()}] Tokenizing texts for Word2Vec training...")
    tokenize_start = time.time()
    train_sentences = [tokenize_text(text) for text in X_train]
    print(f"[{datetime.now()}] Tokenization completed in {(time.time() - tokenize_start)/60:.1f} minutes")
    
    # Train Word2Vec model on training data (optimized for large datasets and speed)
    print(f"[{datetime.now()}] Training Word2Vec model...")
    w2v_start = time.time()
    word2vec_model = Word2Vec(
        sentences=train_sentences,
        vector_size=300,  # Dimensionality of word embeddings
        window=10,  # Larger context window for better semantic capture
        min_count=5,  # Higher threshold for large datasets - filters rare/noisy words
        workers=8,  # Use 8 workers (matching vCPU count) for parallelization
        sg=1,  # Skip-gram (1) often performs better than CBOW (0) on large datasets
        epochs=15,  # Reduced from 20 to 15 to save time while maintaining quality
        negative=5,  # Negative sampling for efficiency (default is 5)
        ns_exponent=0.75,  # Negative sampling exponent (standard value)
        sample=1e-4,  # Downsample frequent words (helps with large vocabularies)
        alpha=0.025,  # Initial learning rate
        min_alpha=0.0001  # Minimum learning rate
    )
    print(f"[{datetime.now()}] Word2Vec training completed in {(time.time() - w2v_start)/60:.1f} minutes")
    
    # Create document embeddings by averaging word vectors
    print(f"[{datetime.now()}] Creating document embeddings...")
    embed_start = time.time()
    X_train_embeddings = np.array([document_vector(word2vec_model, doc) for doc in X_train])
    X_val_embeddings = np.array([document_vector(word2vec_model, doc) for doc in X_val])
    X_test_embeddings = np.array([document_vector(word2vec_model, doc) for doc in X_test])
    print(f"[{datetime.now()}] Document embeddings created in {(time.time() - embed_start)/60:.1f} minutes")
    print(f"[{datetime.now()}] Embedding shape: {X_train_embeddings.shape}")

    # Optimized parameter grid: Reduced from 135 to 12 combinations for ~5 hour runtime
    # Target: ~1h Word2Vec + embeddings, ~4h MLP grid search (12 combos × ~20 min each)
    param_grid = {
        "hidden_layer_sizes": [(200,), (300,), (200, 100)],  # Reduced from 5 to 3 (removed largest: (300,150), (500,250))
        "alpha": [0.0001, 0.001],  # Reduced from 3 to 2 (removed 0.00001)
        "learning_rate_init": [0.001, 0.01],  # Reduced from 3 to 2 (removed 0.0001)
        "batch_size": ['auto']  # Reduced from 3 to 1 (fixed to 'auto' for consistency)
    }
    # Total: 3 × 2 × 2 × 1 = 12 combinations
    
    param_list = list(ParameterGrid(param_grid))
    total_combinations = len(param_list)
    mlp_start_time = time.time()
    
    print(f"\n[{datetime.now()}] Starting Word2Vec MLP grid search with {total_combinations} combinations")
    print(f"[{datetime.now()}] Target runtime: ~5 hours maximum (including Word2Vec training)")

    best_f1 = -1.0
    best_params = None
    best_model = None
    best_word2vec = None

    for idx, params in enumerate(param_list, 1):
        combo_start_time = time.time()
        hidden_layer_sizes = params["hidden_layer_sizes"]
        alpha = params["alpha"]
        learning_rate_init = params["learning_rate_init"]
        batch_size = params["batch_size"]
        
        print(f"\n[{datetime.now()}] [{idx}/{total_combinations}] Training: hidden_layer_sizes={hidden_layer_sizes}, alpha={alpha}, learning_rate_init={learning_rate_init}, batch_size={batch_size}")
        
        # Define MLP classifier with optimized settings
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=300,  # Reduced from 1000 to 300 - early stopping should catch convergence
            random_state=42,
            early_stopping=True,  # Stop early if validation score doesn't improve
            validation_fraction=0.1,  # Use 10% of training data for validation
            n_iter_no_change=10,  # Reduced from 20 to 10 for faster convergence detection
            tol=1e-4,  # Tolerance for optimization
            verbose=False
        )
        # train only on training set
        model.fit(X_train_embeddings, y_train)

        # evaluate on validation set
        y_val_pred = model.predict(X_val_embeddings)
        f1 = f1_score(y_val, y_val_pred)
        
        combo_elapsed = time.time() - combo_start_time
        mlp_elapsed = time.time() - mlp_start_time
        avg_time_per_combo = mlp_elapsed / idx
        estimated_remaining = avg_time_per_combo * (total_combinations - idx)
        total_elapsed = time.time() - script_start_time
        
        print(f"[{datetime.now()}] Completed in {combo_elapsed:.1f}s ({combo_elapsed/60:.1f}min) - val F1={f1:.4f}")
        print(f"[{datetime.now()}] Progress: {idx}/{total_combinations} | MLP elapsed: {mlp_elapsed/60:.1f}min | Est. remaining: {estimated_remaining/60:.1f}min | Total elapsed: {total_elapsed/60:.1f}min")

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_model = model
            best_word2vec = word2vec_model
            print(f"[{datetime.now()}] *** New best F1: {best_f1:.4f} ***")

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

