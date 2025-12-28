#!/bin/bash
# ============================================================================
# Language Mirror Pro - Quick Start Script
# ============================================================================

set -e

echo "=============================================="
echo "🪞 Language Mirror Pro - Setup"
echo "=============================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📦 Python version: $python_version"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install torch numpy fastapi uvicorn pydantic tqdm

echo ""
echo "=============================================="
echo "✅ Setup Complete!"
echo "=============================================="
echo ""
echo "🧪 Test the model:"
echo "   python -m ai_core.models.transformer"
echo ""
echo "🎮 Test the environment:"
echo "   python -m ai_core.training.environment"
echo ""
echo "🏋️  Train the model:"
echo "   python scripts/train.py --num_updates 100"
echo ""
echo "🚀 Start the server:"
echo "   cd backend && python main.py"
echo ""
echo "=============================================="
