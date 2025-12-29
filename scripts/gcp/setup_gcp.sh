#!/bin/bash
# ============================================================================
# 🚀 Language Mirror Pro - Google Cloud Setup & Training
# ============================================================================
#
# This script sets up everything needed to train on Google Cloud
#
# Prerequisites:
# 1. Google Cloud account with billing enabled
# 2. gcloud CLI installed (https://cloud.google.com/sdk/docs/install)
#
# Usage:
#   chmod +x setup_gcp.sh
#   ./setup_gcp.sh
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "🚀 Language Mirror Pro - GCP Setup"
echo "=============================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found!${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set project variables
PROJECT_ID="${GCP_PROJECT_ID:-language-mirror-pro}"
REGION="us-central1"
BUCKET_NAME="${PROJECT_ID}-training"
SERVICE_ACCOUNT="language-mirror-training"

echo ""
echo "📋 Configuration:"
echo "   Project ID: $PROJECT_ID"
echo "   Region: $REGION"
echo "   Bucket: $BUCKET_NAME"
echo ""

# Login to GCP
echo "🔐 Authenticating with Google Cloud..."
gcloud auth login --quiet 2>/dev/null || true

# Create or select project
echo "📁 Setting up project..."
gcloud projects describe $PROJECT_ID &>/dev/null || gcloud projects create $PROJECT_ID

gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable \
    aiplatform.googleapis.com \
    compute.googleapis.com \
    storage.googleapis.com \
    containerregistry.googleapis.com \
    cloudbuild.googleapis.com

# Create storage bucket
echo "🪣 Creating storage bucket..."
gsutil mb -l $REGION gs://$BUCKET_NAME 2>/dev/null || echo "Bucket already exists"

# Create service account
echo "👤 Creating service account..."
gcloud iam service-accounts create $SERVICE_ACCOUNT \
    --display-name="Language Mirror Training" 2>/dev/null || echo "Service account exists"

# Grant permissions
echo "🔑 Granting permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user" --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.admin" --quiet

echo ""
echo -e "${GREEN}✅ GCP Setup Complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Upload training data: gsutil cp -r datasets/processed gs://$BUCKET_NAME/data/"
echo "2. Run training: python train_vertex_ai.py"
echo ""
