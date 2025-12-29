"""
🚀 Language Mirror Pro - Vertex AI Training
============================================
Submit training job to Google Cloud Vertex AI

This script:
1. Packages the training code
2. Uploads data to GCS
3. Submits a training job to Vertex AI
4. Downloads the trained model

Requirements:
- Google Cloud SDK installed
- Vertex AI API enabled
- Storage bucket created

Usage:
    python train_vertex_ai.py --project YOUR_PROJECT_ID
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Check for Google Cloud packages
try:
    from google.cloud import storage
    from google.cloud import aiplatform
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    print("⚠️ Google Cloud packages not installed.")
    print("   Run: pip install google-cloud-aiplatform google-cloud-storage")


def upload_to_gcs(local_path: str, bucket_name: str, gcs_path: str):
    """Upload file or directory to GCS"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    local_path = Path(local_path)
    
    if local_path.is_file():
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))
        print(f"  Uploaded: {local_path} -> gs://{bucket_name}/{gcs_path}")
    else:
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                blob_path = f"{gcs_path}/{relative_path}"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(file_path))
                print(f"  Uploaded: {file_path}")


def download_from_gcs(bucket_name: str, gcs_path: str, local_path: str):
    """Download file from GCS"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    blob = bucket.blob(gcs_path)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"  Downloaded: gs://{bucket_name}/{gcs_path} -> {local_path}")


def submit_vertex_ai_job(
    project_id: str,
    region: str,
    bucket_name: str,
    machine_type: str = "n1-standard-8",
    accelerator_type: str = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
    epochs: int = 10,
    batch_size: int = 32
):
    """Submit training job to Vertex AI"""
    
    aiplatform.init(project=project_id, location=region)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"language_mirror_training_{timestamp}"
    
    # Training script
    training_script = """
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from pathlib import Path
from google.cloud import storage
import math

# Download data from GCS
def download_data():
    client = storage.Client()
    bucket = client.bucket(os.environ['BUCKET_NAME'])
    
    data_dir = Path('/tmp/data')
    data_dir.mkdir(exist_ok=True)
    
    for split in ['train.json', 'val.json']:
        blob = bucket.blob(f"data/{split}")
        blob.download_to_filename(str(data_dir / split))
    
    return data_dir

# Simple tokenizer
class SimpleTokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self._init_vocab()
    
    def _init_vocab(self):
        special = ["<pad>", "<sos>", "<eos>", "<unk>"]
        lang_tokens = [f"<lang:{l}>" for l in ["italian", "japanese", "spanish", "french", "german"]]
        chars = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\\\"()-")
        chars += list("àèìòùáéíóúâêîôûäëïöüñçß¿¡")
        hiragana = [chr(i) for i in range(0x3040, 0x309F)]
        katakana = [chr(i) for i in range(0x30A0, 0x30FF)]
        
        all_tokens = special + lang_tokens + chars + hiragana + katakana
        for i, t in enumerate(all_tokens[:self.vocab_size]):
            self.token_to_id[t] = i
            self.id_to_token[i] = t
    
    @property
    def pad_token_id(self): return 0
    @property
    def sos_token_id(self): return 1
    @property
    def eos_token_id(self): return 2
    
    def encode(self, text, max_len=256):
        tokens = [self.sos_token_id]
        if text.startswith("<lang:"):
            end = text.find(">") + 1
            if text[:end] in self.token_to_id:
                tokens.append(self.token_to_id[text[:end]])
            text = text[end:].strip()
        for c in text:
            tokens.append(self.token_to_id.get(c, 3))
        tokens.append(self.eos_token_id)
        tokens = tokens[:max_len]
        while len(tokens) < max_len:
            tokens.append(0)
        return tokens
    
    def decode(self, ids):
        return "".join([self.id_to_token.get(i, "") for i in ids if i > 3])

# Dataset
class ConvDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=256):
        with open(path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        item = self.data[i]
        inp = self.tokenizer.encode(item["input"], self.max_len)
        out = self.tokenizer.encode(item["output"], self.max_len)
        labels = out[1:] + [0]
        return {
            "input_ids": torch.tensor(inp),
            "output_ids": torch.tensor(out),
            "labels": torch.tensor(labels)
        }

# Model
class LanguageTutor(nn.Module):
    def __init__(self, vocab_size=32000, d_model=512, n_heads=8, n_layers=6, d_ff=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        pe = torch.zeros(256, d_model)
        pos = torch.arange(256).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=n_heads,
            num_encoder_layers=n_layers, num_decoder_layers=n_layers,
            dim_feedforward=d_ff, dropout=0.1, batch_first=True
        )
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embed(src) + self.pe[:, :src.size(1)]
        tgt = self.embed(tgt) + self.pe[:, :tgt.size(1)]
        tgt_len = tgt.size(1)
        causal_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(src.device)
        out = self.transformer(src, tgt, tgt_mask=causal_mask, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
        return self.output(out)

# Train
def train():
    print("🚀 Starting training on Vertex AI")
    
    # Download data
    data_dir = download_data()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    tokenizer = SimpleTokenizer()
    
    train_ds = ConvDataset(str(data_dir / "train.json"), tokenizer)
    val_ds = ConvDataset(str(data_dir / "val.json"), tokenizer)
    
    batch_size = int(os.environ.get('BATCH_SIZE', 32))
    epochs = int(os.environ.get('EPOCHS', 10))
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    
    model = LanguageTutor().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dl:
            inp = batch["input_ids"].to(device)
            out = batch["output_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(inp, out, src_mask=(inp == 0), tgt_mask=(out == 0))
            loss = criterion(logits.view(-1, model.vocab_size), labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dl:
                inp = batch["input_ids"].to(device)
                out = batch["output_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(inp, out, src_mask=(inp == 0), tgt_mask=(out == 0))
                val_loss += criterion(logits.view(-1, model.vocab_size), labels.view(-1)).item()
        
        val_loss /= len(val_dl)
        print(f"Epoch {epoch+1}: train_loss={total_loss/len(train_dl):.4f}, val_loss={val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "/tmp/best_model.pt")
            print("  Saved best model!")
    
    # Upload model to GCS
    client = storage.Client()
    bucket = client.bucket(os.environ['BUCKET_NAME'])
    blob = bucket.blob("models/best_model.pt")
    blob.upload_from_filename("/tmp/best_model.pt")
    print(f"✅ Model uploaded to gs://{os.environ['BUCKET_NAME']}/models/best_model.pt")

if __name__ == "__main__":
    train()
"""
    
    # Create training job
    job = aiplatform.CustomJob.from_local_script(
        display_name=job_name,
        script_path="train_script.py",
        container_uri="gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest",
        requirements=["google-cloud-storage"],
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        environment_variables={
            "BUCKET_NAME": bucket_name,
            "EPOCHS": str(epochs),
            "BATCH_SIZE": str(batch_size)
        }
    )
    
    print(f"\n🚀 Submitting training job: {job_name}")
    job.run(sync=False)
    
    print(f"\n✅ Job submitted!")
    print(f"   Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs")
    
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="GCP Project ID")
    parser.add_argument("--region", default="us-central1", help="GCP Region")
    parser.add_argument("--bucket", help="GCS Bucket name")
    parser.add_argument("--machine", default="n1-standard-8", help="Machine type")
    parser.add_argument("--gpu", default="NVIDIA_TESLA_T4", help="GPU type")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--upload-data", action="store_true", help="Upload data to GCS first")
    args = parser.parse_args()
    
    if not GCP_AVAILABLE:
        print("❌ Google Cloud packages required!")
        print("   pip install google-cloud-aiplatform google-cloud-storage")
        return
    
    bucket_name = args.bucket or f"{args.project}-training"
    
    print("=" * 60)
    print("🚀 Language Mirror Pro - Vertex AI Training")
    print("=" * 60)
    print(f"\n📋 Configuration:")
    print(f"   Project: {args.project}")
    print(f"   Region: {args.region}")
    print(f"   Bucket: {bucket_name}")
    print(f"   Machine: {args.machine}")
    print(f"   GPU: {args.gpu} x {args.gpu_count}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    
    # Upload data if requested
    if args.upload_data:
        print("\n📤 Uploading training data to GCS...")
        upload_to_gcs("datasets/processed", bucket_name, "data")
    
    # Submit job
    submit_vertex_ai_job(
        project_id=args.project,
        region=args.region,
        bucket_name=bucket_name,
        machine_type=args.machine,
        accelerator_type=args.gpu,
        accelerator_count=args.gpu_count,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
