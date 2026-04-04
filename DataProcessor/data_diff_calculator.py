import pandas as pd

# Initial on initial test dataset
initial_metrics_initial_test = {
    "Model": [
        "BERT Body & Subject",
        "BERT Body",
        "Word2Vec MLP",
        "TF-IDF MLP",
        "TF-IDF Logistic Regression",
        "Word2Vec Random Forest",
        "TF-IDF Random Forest",
        "TF-IDF Naïve Bayes",
        "BERT Subject",
        "TF-IDF Decision Tree"
    ],
    "Accuracy": [99.37, 99.24, 98.87, 98.60, 98.38, 98.00, 97.89, 96.45, 96.02, 94.35],
    "F1-Score": [99.34, 99.20, 98.80, 98.52, 98.30, 97.88, 97.77, 96.19, 95.82, 94.10],
    "Precision": [99.37, 99.17, 98.99, 98.59, 98.10, 98.13, 98.03, 97.94, 95.38, 93.18],
    "Recall": [99.31, 99.23, 98.61, 98.45, 98.49, 97.63, 97.51, 94.49, 96.26, 95.04]
}


initial_metrics_initial_test_df = pd.DataFrame(initial_metrics_initial_test).set_index("Model")

initial_metrics_spearbot_test = {
    "Model": [
        "BERT Body & Subject",
        "BERT Body",
        "Word2Vec MLP",
        "TF-IDF MLP",
        "BERT Subject"
    ],
    "Accuracy": [50.50, 46.00, 58.00, 51.50, 55.00],
    "F1-Score": [26.67, 6.90, 55.79, 42.60, 34.78],
    "Precision": [51.43, 25.00, 58.89, 52.17, 63.16],
    "Recall": [18.00, 4.00, 53.00, 36.00, 24.00]
}

initial_metrics_spearbot_test_df = pd.DataFrame(initial_metrics_spearbot_test).set_index("Model")

# Fine tuned on spear-bot dataset
fine_tuned_spearbot_metrics = {
    "Model": [
        "BERT body and subject (2 Epochs)",
        "BERT Body (2 Epochs)",
        "BERT Subject (2 Epochs)",
        "BERT body and subject (3 Epochs)",
        "BERT Body (3 Epochs)",
        "BERT Subject (3 Epochs)",
        "Word2Vec MLP",
        "TF-IDF MLP"
    ],
    "Accuracy": [100, 99.50, 97.00, 100, 99.50, 97.50, 76.50, 74.00],
    "F1-Score": [100, 99.50, 97.03, 100, 99.50, 97.51, 77.29, 75.00],
    "Precision": [100, 100, 96.08, 100, 100, 97.03, 74.77, 72.22],
    "Recall": [100, 99.00, 98.00, 100, 99.00, 98.00, 80.00, 78.00]
}
fine_tuned_spearbot_metrics_df = pd.DataFrame(fine_tuned_spearbot_metrics).set_index("Model")

# Fine-tuned on initial test dataset
fine_tuned_initial_test_metrics = {
    "Model": [
        "BERT body and subject (2 Epochs)",
        "BERT Body (2 Epochs)",
        "BERT Subject (2 Epochs)",
        "BERT body and subject (3 Epochs)",
        "BERT Body (3 Epochs)",
        "BERT Subject (3 Epochs)",
        "Word2Vec MLP",
        "TF-IDF MLP"
    ],
    "Accuracy": [98.74, 98.57, 95.17, 98.77, 98.77, 95.24, 97.68, 98.44],
    "F1-Score": [98.68, 98.50, 95.00, 98.71, 98.40, 95.06, 97.60, 98.37],
    "Precision": [97.85, 97.99, 93.31, 97.88, 97.75, 93.57, 95.76, 97.81],
    "Recall": [99.53, 99.01, 96.75, 99.56, 99.05, 96.60, 99.51, 98.93]
}

fine_tuned_initial_test_metrics_df = pd.DataFrame(fine_tuned_initial_test_metrics).set_index("Model")

diff_df = initial_metrics_initial_test_df.copy()

for col in initial_metrics_initial_test_df.columns:
    diff_df[col] = initial_metrics_initial_test_df[col].combine(initial_metrics_spearbot_test_df[col],
       lambda x, y: f"({x:.4f} * 31398 + {y:.4f}) / (31398+200) = {((x*31398 + y*200)/ (31398+200)):.4f}"
    )

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)  # important for your equation strings
pd.set_option('display.width', None)

print(diff_df)
