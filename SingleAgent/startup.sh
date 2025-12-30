#!/bin/bash
# VM Startup Script - Environment Setup Only
# This script runs on VM startup to set up the environment for training
# Training is started manually via run_training.sh

# Don't exit on error - we want to continue even if some steps fail
# set -e  # Commented out to allow graceful error handling
set -o pipefail  # Exit on pipe failure

# Generate unique run ID for this training session (prevents GCS path conflicts)
RUN_ID="run_$(date +%Y%m%d_%H%M%S)_$$"
export TRAINING_RUN_ID="$RUN_ID"

# Set USER variable (override any existing value)
# Correct syntax: export USER=value (no $ before USER, no spaces around =)
export USER="s_jamieson_22"

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
# Wait for apt lock to be released if another process is using it
echo "[$(date)] Waiting for apt lock (if needed)..."
timeout=300  # 5 minutes max wait
while sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1 || sudo fuser /var/lib/dpkg/lock >/dev/null 2>&1; do
    echo "[$(date)] Apt lock detected, waiting 10 seconds..."
    sleep 10
    timeout=$((timeout - 10))
    if [ $timeout -le 0 ]; then
        echo "[$(date)] WARNING: Timeout waiting for apt lock. Continuing anyway..."
        break
    fi
done

sudo apt-get update -y || {
    echo "[$(date)] WARNING: apt-get update failed (may already be updated). Continuing..."
}

sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip git build-essential dkms curl coreutils || {
    echo "[$(date)] WARNING: Some packages may have failed to install. Continuing..."
}

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
# Wait for apt lock if needed
while sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1 || sudo fuser /var/lib/dpkg/lock >/dev/null 2>&1; do
    echo "[$(date)] Apt lock detected, waiting 5 seconds..."
    sleep 5
done

sudo apt-get update -y || {
    echo "[$(date)] WARNING: apt-get update failed. Continuing..."
}

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

# Ensure proper ownership of directories (in case script runs as root)
# This prevents permission errors when run_training.sh runs as regular user
chown -R "$USER:$USER" Data Logs Results 2>/dev/null || true
chmod -R u+w Data Logs Results 2>/dev/null || true

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
# PHASE 4: Training Preparation (Manual Training)
# ============================================================================
echo "[$(date)] Phase 4: Environment setup complete. Ready for manual training."
echo "[$(date)] Training Run ID: $RUN_ID"
echo "[$(date)] To train all 10 models, run manually:"
echo "  cd /home/$USER/FinalYearProject/SingleAgent"
echo "  bash run_training.sh"
echo ""
echo "[$(date)] Models to be trained:"
echo "  GPU Models (3): BERT Body, BERT Subject, BERT Body+Subject"
echo "  CPU Models (7): TF-IDF LR, TF-IDF NB, TF-IDF DT, TF-IDF RF, TF-IDF MLP, Word2Vec MLP, Word2Vec RF"
echo "[$(date)] Each model will be stored in a unique GCS location to prevent data loss"
echo "[$(date)] Real-time logs available at: $REALTIME_LOG"

# Change to SingleAgent directory where run_training.sh is located
# run_training.sh is triggered from: /home/$USER/FinalYearProject/SingleAgent
cd /home/$USER/FinalYearProject/SingleAgent

# Ensure proper ownership of entire project directory (in case script runs as root)
# This prevents permission errors when run_training.sh runs as regular user
chown -R "$USER:$USER" /home/$USER/FinalYearProject 2>/dev/null || true
chmod -R u+w /home/$USER/FinalYearProject 2>/dev/null || true

# Make run_training.sh executable
chmod +x run_training.sh

echo "[$(date)] ========================================"
echo "[$(date)] Setup Complete: $(date)"
echo "[$(date)] Environment is ready for training"
echo "[$(date)] Run ID: $RUN_ID"
echo "[$(date)] ========================================"
echo ""
echo "To start training, SSH into the VM and run:"
echo "  cd /home/$USER/FinalYearProject/SingleAgent"
echo "  bash run_training.sh"
echo ""
echo "Monitor progress with:"
echo "  tail -f $REALTIME_LOG"
echo ""

# Exit cleanly - VM stays running for manual training
# Note: Training will be started manually via run_training.sh
# VM shutdown should be done manually after training completes
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
