# 🚀 Language Mirror Pro - Google Cloud Training Guide

Train your custom 50M+ parameter language tutor model on Google Cloud Vertex AI with GPUs!

## 📋 Overview

| Phase | Task | Time |
|-------|------|------|
| 1 | Setup GCP Account | 10 min |
| 2 | Prepare Dataset | 5 min |
| 3 | Upload to GCS | 5 min |
| 4 | Train on Vertex AI | 2-4 hours |
| 5 | Deploy Model | 10 min |

**Total: ~3-5 hours** (mostly waiting for training)

---

## Phase 1: Setup Google Cloud

### 1.1 Create GCP Account (if needed)
1. Go to https://cloud.google.com
2. Click "Get started for free"
3. You get **$300 free credits** for 90 days!

### 1.2 Install Google Cloud CLI

**Mac:**
```bash
brew install google-cloud-sdk
```

**Or download from:** https://cloud.google.com/sdk/docs/install

### 1.3 Login and Setup Project
```bash
# Login
gcloud auth login

# Create project
gcloud projects create language-mirror-pro --name="Language Mirror Pro"

# Set as default
gcloud config set project language-mirror-pro

# Enable billing (required for training)
# Go to: https://console.cloud.google.com/billing

# Enable required APIs
gcloud services enable aiplatform.googleapis.com storage.googleapis.com
```

### 1.4 Create Storage Bucket
```bash
gsutil mb -l us-central1 gs://language-mirror-pro-training
```

---

## Phase 2: Prepare Dataset

### 2.1 Download the scripts
Download these files to your project:
- `prepare_dataset.py` - Generates training data
- `train_gcp.py` - Local training script
- `train_vertex_ai.py` - Cloud training script

### 2.2 Generate Training Data
```bash
cd ~/Projects/language-mirror-pro

# Create scripts directory
mkdir -p scripts/gcp
mv ~/Downloads/prepare_dataset.py scripts/gcp/
mv ~/Downloads/train_gcp.py scripts/gcp/
mv ~/Downloads/train_vertex_ai.py scripts/gcp/

# Activate venv
source venv/bin/activate

# Install requirements
pip install google-cloud-aiplatform google-cloud-storage tqdm

# Generate dataset (100K conversations)
python scripts/gcp/prepare_dataset.py
```

This creates:
```
datasets/
  processed/
    train.json    (80K examples)
    val.json      (10K examples)
    test.json     (10K examples)
```

---

## Phase 3: Upload Data to GCS

```bash
# Upload training data
gsutil -m cp -r datasets/processed gs://language-mirror-pro-training/data/

# Verify
gsutil ls gs://language-mirror-pro-training/data/
```

---

## Phase 4: Train on Vertex AI

### Option A: Quick Training (T4 GPU - $0.35/hr)
```bash
python scripts/gcp/train_vertex_ai.py \
    --project language-mirror-pro \
    --gpu NVIDIA_TESLA_T4 \
    --epochs 10 \
    --batch-size 32
```

### Option B: Faster Training (A100 GPU - $3.67/hr)
```bash
python scripts/gcp/train_vertex_ai.py \
    --project language-mirror-pro \
    --gpu NVIDIA_TESLA_A100 \
    --epochs 10 \
    --batch-size 64
```

### Option C: Local Training (Your Mac M4)
```bash
python scripts/gcp/train_gcp.py --local --epochs 10 --batch_size 16
```

### Monitor Training
- Go to: https://console.cloud.google.com/vertex-ai/training/custom-jobs
- Or watch logs in terminal

---

## Phase 5: Download & Deploy Trained Model

### 5.1 Download Model
```bash
# Download from GCS
gsutil cp gs://language-mirror-pro-training/models/best_model.pt models/

# Or use the output directory if trained locally
cp trained_model/best_model.pt models/language_mirror_trained.pt
```

### 5.2 Update Backend to Use New Model

Edit `backend/main.py` to load the trained model:

```python
# Add this import
from scripts.gcp.train_gcp import LanguageTutorTransformer, SimpleTokenizer, TrainingConfig

# Load trained model
config = TrainingConfig()
tokenizer = SimpleTokenizer(config.vocab_size)
model = LanguageTutorTransformer.load("models/language_mirror_trained.pt", config.device)
model.eval()
```

### 5.3 Test the Model
```bash
cd ~/Projects/language-mirror-pro/backend
python main.py
```

---

## 💰 Cost Estimate

| GPU Type | Cost/Hour | 10 Epochs (~2hr) |
|----------|-----------|------------------|
| T4 | $0.35 | ~$0.70 |
| V100 | $2.48 | ~$5.00 |
| A100 | $3.67 | ~$7.35 |

**With $300 free credits, you can train many times!**

---

## 🔧 Troubleshooting

### "Quota exceeded"
```bash
# Request GPU quota increase
# Go to: https://console.cloud.google.com/iam-admin/quotas
# Search for "NVIDIA" and request increase
```

### "Permission denied"
```bash
gcloud auth application-default login
```

### "Bucket not found"
```bash
gsutil mb -l us-central1 gs://YOUR_PROJECT_ID-training
```

---

## 📊 Model Architecture

```
LanguageTutorTransformer (50M+ parameters)
├── Embedding Layer (vocab_size=32000, d_model=512)
├── Positional Encoding
├── Transformer Encoder (6 layers)
│   ├── Multi-Head Attention (8 heads)
│   └── Feed-Forward (2048 dim)
├── Transformer Decoder (6 layers)
│   ├── Masked Multi-Head Attention
│   ├── Cross-Attention
│   └── Feed-Forward
└── Output Linear (512 -> 32000)
```

---

## 🎯 Expected Results

After 10 epochs:
- Train Loss: ~1.5
- Val Loss: ~1.8
- The model will generate contextual responses in all 5 languages!

---

## 🚀 Quick Commands Summary

```bash
# 1. Setup
gcloud auth login
gcloud config set project language-mirror-pro
gcloud services enable aiplatform.googleapis.com storage.googleapis.com
gsutil mb -l us-central1 gs://language-mirror-pro-training

# 2. Prepare Data
python scripts/gcp/prepare_dataset.py

# 3. Upload Data
gsutil -m cp -r datasets/processed gs://language-mirror-pro-training/data/

# 4. Train (pick one)
python scripts/gcp/train_gcp.py --local --epochs 5      # Local (Mac)
python scripts/gcp/train_vertex_ai.py --project language-mirror-pro  # Cloud (GPU)

# 5. Deploy
gsutil cp gs://language-mirror-pro-training/models/best_model.pt models/
python backend/main.py
```

---

Good luck with your hackathon! 🎉
