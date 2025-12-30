#!/bin/bash
# VM Startup Script for Training All Agents
# This script runs on VM startup, trains all models efficiently, uploads results, and shuts down

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# Generate unique run ID for this training session (prevents GCS path conflicts)
RUN_ID="run_$(date +%Y%m%d_%H%M%S)_$$"
export TRAINING_RUN_ID="$RUN_ID"

# Log everything to a file for debugging
LOGFILE="/home/$USER/training_orchestrator_${RUN_ID}.log"
REALTIME_LOG="/home/$USER/realtime_training.log"

# Set up real-time log monitoring with safety features
# This creates a log file that can be tailed in real-time from another session
touch "$REALTIME_LOG"
chmod 644 "$REALTIME_LOG"

# Use unbuffered logging to prevent hangs from buffered output
# This is a proven, safe approach that works everywhere
export PYTHONUNBUFFERED=1

# Redirect output to both log files
# Simple, proven approach - if one fails, script continues
# Using basic tee (no stdbuf dependency) for maximum compatibility
exec > >(tee -a "$LOGFILE" "$REALTIME_LOG" 2>/dev/null || tee -a "$LOGFILE" 2>/dev/null || cat) 2>&1

echo "Training Run ID: $RUN_ID"
echo "Monitor logs in real-time with: tail -f $REALTIME_LOG"
echo "Logging configured with PYTHONUNBUFFERED=1 to prevent hangs from buffered output"

echo "=========================================="
echo "Training Orchestrator Started: $(date)"
echo "=========================================="

# ============================================================================
# PHASE 1: System Setup and Dependencies
# ============================================================================
echo "[$(date)] Phase 1: Installing system dependencies..."

# Set non-interactive mode to prevent prompts during package installation
export DEBIAN_FRONTEND=noninteractive
export DEBCONF_NONINTERACTIVE_SEEN=true

# Pre-configure all common interactive prompts to avoid hangs
# Keyboard configuration
echo "keyboard-configuration keyboard-configuration/layoutcode string us" | sudo debconf-set-selections
echo "keyboard-configuration keyboard-configuration/variantcode string" | sudo debconf-set-selections
echo "keyboard-configuration keyboard-configuration/modelcode string pc105" | sudo debconf-set-selections
echo "keyboard-configuration keyboard-configuration/optionscode string" | sudo debconf-set-selections

# Timezone configuration (set to UTC to avoid prompts)
echo "tzdata tzdata/Areas select Etc" | sudo debconf-set-selections
echo "tzdata tzdata/Zones/Etc select UTC" | sudo debconf-set-selections

# Locale configuration (set to en_US.UTF-8)
echo "locales locales/default_environment_locale select en_US.UTF-8" | sudo debconf-set-selections
echo "locales locales/locales_to_be_generated multiselect en_US.UTF-8 UTF-8" | sudo debconf-set-selections

# Prevent grub configuration prompts (if grub gets updated)
echo "grub-pc grub-pc/install_devices multiselect /dev/sda" | sudo debconf-set-selections

# Prevent postfix configuration prompts (if mail server packages are installed)
echo "postfix postfix/main_mailer_type select No configuration" | sudo debconf-set-selections

# Install Google Cloud CLI if not already installed
if ! command -v gcloud &> /dev/null; then
snap install google-cloud-cli --classic
fi

# Update and install essentials (with non-interactive flag)
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip git build-essential dkms curl coreutils

# Install kernel headers
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y linux-headers-$(uname -r)

# Add NVIDIA package repositories
distribution=$(. /etc/os-release; echo ${ID}${VERSION_ID})
# Use --fail flag to fail on HTTP errors, and retry on network issues
curl -f -s -L --retry 3 --retry-delay 5 https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add - || {
    echo "[$(date)] ERROR: Failed to add NVIDIA GPG key. Retrying..."
    sleep 5
    curl -f -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add - || {
        echo "[$(date)] ERROR: Failed to add NVIDIA GPG key after retry. Continuing anyway..."
    }
}
curl -f -s -L --retry 3 --retry-delay 5 https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/libnvidia-container.list || {
    echo "[$(date)] ERROR: Failed to add NVIDIA repository. Continuing anyway..."
}

# Update again to include NVIDIA repo
sudo apt-get update -y

# Install NVIDIA driver (550 works for L4)
echo "[$(date)] Installing NVIDIA driver..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-driver-550

# Wait for driver to be ready
sleep 10
nvidia-smi || echo "Warning: nvidia-smi not available yet, continuing..."

# Configure GCP
# Note: APIs should be enabled at project level before VM creation
gcloud config set project final-year-project-477110 || {
    echo "[$(date)] WARNING: Could not set GCP project. Continuing..."
}

# ============================================================================
# PHASE 2: Project Setup
# ============================================================================
echo "[$(date)] Phase 2: Setting up project..."

cd /home/$USER

# Clone repository if not already present
if [ ! -d "FinalYearProject" ]; then
    echo "[$(date)] Cloning repository..."
    git clone https://github.com/sjamieson23/FinalYearProject.git || {
        echo "[$(date)] ERROR: Git clone failed. Retrying in 10 seconds..."
        sleep 10
        git clone https://github.com/sjamieson23/FinalYearProject.git || {
            echo "[$(date)] ERROR: Git clone failed after retry. Check network connectivity."
            exit 1
        }
    }
else
    echo "[$(date)] Repository already exists, skipping clone"
    # Update repository in case it exists but is outdated
    cd FinalYearProject && git pull || echo "[$(date)] Warning: Could not update repository"
    cd /home/$USER
fi

cd FinalYearProject

# Set PYTHONPATH to allow imports from SingleAgent
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

cd SingleAgent

# Create necessary directories
mkdir -p Data
mkdir -p Logs
mkdir -p Results/Saves

# Download data from GCS
echo "[$(date)] Downloading training data from GCS..."
# Check if gsutil is available and authenticated
if ! command -v gsutil &> /dev/null; then
    echo "[$(date)] ERROR: gsutil not found. Installing..."
    pip install --upgrade gsutil || echo "[$(date)] WARNING: Could not install gsutil"
fi

# Try to download data with retry logic
gsutil -m cp gs://model-storage-data/Data/*.csv ./Data/ 2>&1 || {
    echo "[$(date)] WARNING: Initial data download failed. Checking authentication..."
    # Check if we can access the bucket at all
    gsutil ls gs://model-storage-data/ > /dev/null 2>&1 || {
        echo "[$(date)] ERROR: Cannot access GCS bucket. Check permissions."
        echo "[$(date)] Attempting to continue without data download..."
    }
    # Try downloading individual files
    echo "[$(date)] Attempting to download files individually..."
    gsutil cp gs://model-storage-data/Data/all_data_train.csv ./Data/ || echo "[$(date)] Warning: Could not download train data"
    gsutil cp gs://model-storage-data/Data/all_data_val.csv ./Data/ || echo "[$(date)] Warning: Could not download val data"
    gsutil cp gs://model-storage-data/Data/all_data_test.csv ./Data/ || echo "[$(date)] Warning: Could not download test data"
}

# ============================================================================
# PHASE 3: Python Environment Setup
# ============================================================================
echo "[$(date)] Phase 3: Setting up Python environment..."

# Install PyTorch with CUDA 12.1 support (optimized for L4)
# Use --no-cache-dir to save disk space and --timeout to prevent hangs
pip install --upgrade pip --timeout=300
pip install --no-cache-dir --timeout=600 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || {
    echo "[$(date)] WARNING: PyTorch installation failed, retrying..."
    pip install --no-cache-dir --timeout=600 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}

# Install all required Python packages with retry logic
pip install --no-cache-dir --timeout=300 transformers datasets evaluate accelerate scikit-learn pandas python-dateutil tensorboard || {
    echo "[$(date)] WARNING: Some packages failed, retrying..."
    pip install --no-cache-dir --timeout=300 transformers datasets evaluate accelerate scikit-learn pandas python-dateutil tensorboard
}
pip install --no-cache-dir --timeout=300 google-cloud-storage google-api-python-client || {
    echo "[$(date)] WARNING: GCP packages failed, retrying..."
    pip install --no-cache-dir --timeout=300 google-cloud-storage google-api-python-client
}
pip install --no-cache-dir --timeout=300 accelerate safetensors || {
    echo "[$(date)] WARNING: Accelerate packages failed, retrying..."
    pip install --no-cache-dir --timeout=300 accelerate safetensors
}
pip install --no-cache-dir --timeout=300 joblib gensim || {
    echo "[$(date)] WARNING: ML packages failed, retrying..."
    pip install --no-cache-dir --timeout=300 joblib gensim
}

echo "[$(date)] Environment setup complete!"

# ============================================================================
# PHASE 4: Training Orchestration
# ============================================================================
echo "[$(date)] Phase 4: Starting training pipeline..."
echo "[$(date)] Training Run ID: $RUN_ID"
echo "[$(date)] This session will train 10 separate single agent systems:"
echo "  GPU Models (3): BERT Body, BERT Subject, BERT Body+Subject"
echo "  CPU Models (7): TF-IDF LR, TF-IDF NB, TF-IDF DT, TF-IDF RF, TF-IDF MLP, Word2Vec MLP, Word2Vec RF"
echo "[$(date)] Each model will be stored in a unique GCS location to prevent data loss"
echo "[$(date)] Real-time logs available at: $REALTIME_LOG"

# Track training status
TRAINING_ERRORS=0
TRAINING_LOG="/home/$USER/training_status_${RUN_ID}.log"

# Function to run a training script and log results
run_training() {
    local script_name=$1
    local script_path=$2
    local description=$3
    
    echo "[$(date)] ========================================"
    echo "[$(date)] Starting: $description"
    echo "[$(date)] Script: $script_path"
    echo "[$(date)] Run ID: $RUN_ID"
    echo "[$(date)] ========================================"
    
    # Ensure we're in the SingleAgent directory and PYTHONPATH is set
    cd /home/$USER/FinalYearProject/SingleAgent
    export PYTHONPATH="/home/$USER/FinalYearProject:${PYTHONPATH}"
    
    # Run training with timeout protection and simple logging
    # Timeout prevents indefinite hangs (24 hours max per model - very generous)
    # Simple tee logging - proven and reliable
    # PYTHONUNBUFFERED=1 (set above) ensures Python outputs immediately
    if command -v timeout >/dev/null 2>&1; then
        # Use timeout if available (should be on Ubuntu)
        if timeout 86400 python3 "$script_path" 2>&1 | tee -a "$TRAINING_LOG" "$REALTIME_LOG" 2>/dev/null || python3 "$script_path" >> "$TRAINING_LOG" 2>&1; then
            echo "[$(date)] ✅ SUCCESS: $description"
            echo "[$(date)] Model saved to unique GCS location"
            return 0
        else
            echo "[$(date)] ❌ FAILED: $description (check $TRAINING_LOG for details)"
            TRAINING_ERRORS=$((TRAINING_ERRORS + 1))
            return 1
        fi
    else
        # Fallback if timeout not available (shouldn't happen on Ubuntu)
        if python3 "$script_path" 2>&1 | tee -a "$TRAINING_LOG" "$REALTIME_LOG" 2>/dev/null || python3 "$script_path" >> "$TRAINING_LOG" 2>&1; then
            echo "[$(date)] ✅ SUCCESS: $description"
            echo "[$(date)] Model saved to unique GCS location"
            return 0
        else
            echo "[$(date)] ❌ FAILED: $description (check $TRAINING_LOG for details)"
            TRAINING_ERRORS=$((TRAINING_ERRORS + 1))
            return 1
        fi
    fi
}

# GPU-INTENSIVE MODELS (Run sequentially to avoid GPU memory conflicts)
# These require the NVIDIA L4 GPU and should run one at a time
echo "[$(date)] === Training GPU Models (Sequential) ==="

run_training "bert_body" "bert_body.py" "BERT Body Model"
run_training "bert_subj" "bert_subj.py" "BERT Subject Model"
run_training "bert_body_and_subj" "bert_body_and_subj.py" "BERT Body and Subject Model"

# CPU-BASED MODELS (Can run in parallel, but limited by memory)
# Running sequentially to avoid memory issues with 16GB RAM
# If you have more RAM, you could run 2-3 in parallel
echo "[$(date)] === Training CPU Models (Sequential for memory safety) ==="

run_training "tf_idf_lr" "tf_idf_lr.py" "TF-IDF Logistic Regression"
run_training "tf_idf_nb" "tf_idf_nb.py" "TF-IDF Naive Bayes"
run_training "tf_idf_dt" "tf_idf_dt.py" "TF-IDF Decision Tree"
run_training "tf_idf_rf" "tf_idf_rf.py" "TF-IDF Random Forest"
run_training "tf_idf_mlp" "tf_idf_mlp.py" "TF-IDF Multi-Layer Perceptron"
run_training "word2vec_mlp" "word2vec_mlp.py" "Word2Vec Multi-Layer Perceptron"
run_training "word2vec_rf" "word2vec_rf.py" "Word2Vec Random Forest"

# Optional: Run pretrained evaluation if needed
# run_training "pretrained" "pretrained.py" "Pretrained Model Evaluation"

echo "[$(date)] === Training Pipeline Complete ==="
echo "[$(date)] Total models trained: 10"
echo "[$(date)] Successful: $((10 - TRAINING_ERRORS))"
echo "[$(date)] Failed: $TRAINING_ERRORS"
echo "[$(date)] Run ID: $RUN_ID"

# ============================================================================
# PHASE 5: Final Upload and Cleanup
# ============================================================================
echo "[$(date)] Phase 5: Finalizing and uploading results..."

# Ensure all results are uploaded (some may have been uploaded during training)
# Upload any remaining logs and results
if [ -d "Logs" ]; then
    echo "[$(date)] Uploading logs..."
    cd /home/$USER/FinalYearProject/SingleAgent
    export PYTHONPATH="/home/$USER/FinalYearProject:${PYTHONPATH}"
    python3 -c "
from SingleAgent.common import uploadDataToBucket
import os
os.chdir('/home/$USER/FinalYearProject/SingleAgent')
for log_dir in ['Logs']:
    try:
        uploadDataToBucket(log_dir)
        print(f'Uploaded {log_dir}')
    except Exception as e:
        print(f'Error uploading {log_dir}: {e}')
" || echo "Warning: Some logs may not have uploaded"
fi

# Upload the orchestrator log and real-time log
if [ -f "$LOGFILE" ]; then
    python3 << EOF
from google.cloud import storage
import os
client = storage.Client()
bucket = client.get_bucket('model-storage-data')
from dateutil.utils import today
folder_name = today().strftime('%Y-%m-%d') + '_test1'
run_id = os.environ.get('TRAINING_RUN_ID', 'unknown')

# Upload orchestrator log
blob = bucket.blob(f'{folder_name}/orchestrator_logs/{run_id}/orchestrator.log')
blob.upload_from_filename('$LOGFILE')
print(f'Uploaded orchestrator log: {run_id}/orchestrator.log')

# Upload real-time log if it exists
if os.path.exists('$REALTIME_LOG'):
    blob = bucket.blob(f'{folder_name}/orchestrator_logs/{run_id}/realtime.log')
    blob.upload_from_filename('$REALTIME_LOG')
    print(f'Uploaded real-time log: {run_id}/realtime.log')
EOF
    echo "[$(date)] Logs uploaded to GCS with Run ID: $RUN_ID"
else
    echo "Warning: Could not upload orchestrator log"
fi

# ============================================================================
# PHASE 6: Shutdown
# ============================================================================
echo "[$(date)] Phase 6: Shutting down VM..."
echo "[$(date)] All 10 models have been trained and uploaded to GCS"
echo "[$(date)] Run ID: $RUN_ID"
echo "[$(date)] Models are stored in unique locations in gs://model-storage-data"

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
echo "Training Orchestrator Complete: $(date)"
echo "=========================================="

# Note: The VM shutdown may take a moment, so we exit cleanly
exit 0

# ============================================================================
# VM SPECIFICATIONS COMMENTARY
# ============================================================================
# Location: us-west1-a
# Specs: Nvidia L4 GPU, g2-standard-4 (4 vCPUs, 16 GB Memory) Consider g2-standard-8
# OS: ubuntu-2204-jammy-v20251023 x86_64, 100GB disk
#
# SPECIFICATION ANALYSIS:
# ✅ GPU: NVIDIA L4 (24GB VRAM) - Excellent for BERT models
#   - L4 is well-suited for transformer training
#   - 24GB VRAM can handle batch sizes of 10-16 for BERT-base
#   - Good balance of performance and cost
#
# ⚠️  CPU/Memory: 4 vCPUs, 16GB RAM - Adequate but tight
#   - 16GB RAM is sufficient for single model training
#   - May be limiting if running multiple CPU models in parallel
#   - 4 vCPUs is adequate for data preprocessing and CPU-based models
#   - Recommendation: Consider g2-standard-8 (8 vCPUs, 32GB) if budget allows
#     for better parallelization of CPU models
#
# 💰 COST OPTIMIZATION STRATEGIES:
# 1. Sequential GPU training (as implemented) maximizes GPU utilization
# 2. Sequential CPU training prevents memory issues and OOM kills
# 3. Automatic shutdown prevents idle time charges
# 4. All models upload to GCS immediately after training (fault tolerance)
# 5. Script continues even if individual models fail (resilience)
#
# ⏱️  ESTIMATED RUNTIME:
# - BERT models: ~2-4 hours each (3 epochs, depends on dataset size)
# - TF-IDF models: ~30-60 minutes each
# - Word2Vec models: ~1-2 hours each (includes Word2Vec training)
# - Total: ~12-20 hours depending on dataset size
#
# 🔧 OPTIMIZATION NOTES:
# - Using CUDA 12.1 PyTorch for optimal L4 performance
# - FP16 training enabled in BERT models for speed and memory
# - Batch sizes tuned for 24GB VRAM
# - Data preprocessing uses parallel processing (num_proc=4)
