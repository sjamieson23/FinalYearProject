#!/bin/bash
# Training Runner Script
# This script runs all training models assuming environment is already set up
# Run this from: /home/$USER/FinalYearProject/SingleAgent

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Training Runner Started: $(date)"
echo "Running from directory: $(pwd)"
echo "=========================================="

# Set PYTHONPATH
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"

# Track training status
TRAINING_ERRORS=0
TRAINING_LOG="/home/$USER/training_status_${TRAINING_RUN_ID:-manual}.log"

# Function to run a training script
run_training() {
    local script_name=$1
    local script_path=$2
    local description=$3
    
    echo "[$(date)] ========================================"
    echo "[$(date)] Starting: $description"
    echo "[$(date)] Script: $script_path"
    echo "[$(date)] Working directory: $(pwd)"
    echo "[$(date)] ========================================"
    
    # Run training with simple logging
    if python3 "$script_path" >> "$TRAINING_LOG" 2>&1; then
        echo "[$(date)] ✅ SUCCESS: $description"
        return 0
    else
        echo "[$(date)] ❌ FAILED: $description (check $TRAINING_LOG for details)"
        TRAINING_ERRORS=$((TRAINING_ERRORS + 1))
        return 1
    fi
}

# GPU-INTENSIVE MODELS (Run sequentially)
echo "[$(date)] === Training GPU Models (Sequential) ==="

run_training "bert_body" "bert_body.py" "BERT Body Model"
run_training "bert_subj" "bert_subj.py" "BERT Subject Model"
run_training "bert_body_and_subj" "bert_body_and_subj.py" "BERT Body and Subject Model"

# CPU-BASED MODELS (Run sequentially)
echo "[$(date)] === Training CPU Models (Sequential) ==="

run_training "tf_idf_lr" "tf_idf_lr.py" "TF-IDF Logistic Regression"
run_training "tf_idf_nb" "tf_idf_nb.py" "TF-IDF Naive Bayes"
run_training "tf_idf_dt" "tf_idf_dt.py" "TF-IDF Decision Tree"
run_training "tf_idf_rf" "tf_idf_rf.py" "TF-IDF Random Forest"
run_training "tf_idf_mlp" "tf_idf_mlp.py" "TF-IDF Multi-Layer Perceptron"
run_training "word2vec_mlp" "word2vec_mlp.py" "Word2Vec Multi-Layer Perceptron"
run_training "word2vec_rf" "word2vec_rf.py" "Word2Vec Random Forest"

echo "[$(date)] === Training Pipeline Complete ==="
echo "[$(date)] Total models trained: 10"
echo "[$(date)] Successful: $((10 - TRAINING_ERRORS))"
echo "[$(date)] Failed: $TRAINING_ERRORS"

# ============================================================================
# VM Shutdown
# ============================================================================
echo "[$(date)] Shutting down VM..."
echo "[$(date)] All training complete. Models uploaded to GCS."

# Terminate VM using GCP API
python3 << EOF
from googleapiclient import discovery
import sys

try:
    service = discovery.build('compute', 'v1')
    project = 'final-year-project-477110'
    zone = 'us-west1-a'
    instance_name = 'single-agent-training'
    
    print("Terminating VM...")
    request = service.instances().stop(project=project, zone=zone, instance=instance_name)
    response = request.execute()
    print("Stop request submitted successfully")
    print("Response:", response)
except Exception as e:
    print(f"Error terminating VM: {e}")
    print("VM may need to be stopped manually")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo "[$(date)] ✅ Shutdown command sent. VM will stop shortly."
    echo "[$(date)] Cost optimization: VM will be stopped to prevent idle charges"
else
    echo "[$(date)] ⚠️  Warning: VM termination may have failed. Please check manually."
fi

echo "=========================================="
echo "Training Complete: $(date)"
echo "=========================================="

exit $TRAINING_ERRORS

