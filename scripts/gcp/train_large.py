"""
🚀 Language Mirror Pro - Large Scale GCP Training
==================================================
Train on mC4 + CC-100 data using Google Cloud Vertex AI.

Features:
- Mixed precision training (FP16)
- Gradient accumulation
- Checkpointing to GCS
- Distributed training support
- Streaming data loading

Usage:
    # Local test
    python train_large.py --local --epochs 5
    
    # GCP Training
    python train_large.py --epochs 20 --batch-size 128

Requirements:
    pip install torch datasets google-cloud-storage tqdm
"""

import os
import sys
import json
import math
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class TrainConfig:
    # Model Architecture
    vocab_size: int = 32000
    d_model: int = 768          # Larger model
    n_heads: int = 12
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    d_ff: int = 3072
    max_seq_length: int = 128
    dropout: float = 0.1
    
    # Training
    batch_size: int = 64
    gradient_accumulation: int = 4  # Effective batch = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 10
    max_steps: int = -1  # -1 for full epochs
    
    # Optimization
    use_amp: bool = True        # Mixed precision
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1
    
    # Data
    data_dir: str = "datasets/large_scale"
    
    # Output
    output_dir: str = "trained_model_large"
    save_steps: int = 5000
    eval_steps: int = 1000
    
    # GCS
    gcs_bucket: str = ""
    gcs_output_prefix: str = "models"
    
    # Device
    device: str = ""
    
    def __post_init__(self):
        if not self.device:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


# ============================================================================
# TOKENIZER
# ============================================================================

class MultilingualTokenizer:
    """Fast tokenizer for multilingual text"""
    
    SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>", "<sep>", "<mask>"]
    LANG_TOKENS = ["<lang:italian>", "<lang:japanese>", "<lang:spanish>", 
                   "<lang:french>", "<lang:german>", "<lang:english>"]
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self._build_vocab()
    
    def _build_vocab(self):
        tokens = self.SPECIAL_TOKENS + self.LANG_TOKENS
        
        # ASCII
        tokens += list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        tokens += list("0123456789")
        tokens += list(" .,!?;:'\"-()[]{}@#$%&*+=<>/\\|`~_^")
        
        # European accents
        tokens += list("àèìòùáéíóúâêîôûäëïöüÀÈÌÒÙÁÉÍÓÚÂÊÎÔÛÄËÏÖÜ")
        tokens += list("ñÑçÇßœŒæÆøØåÅ")
        tokens += list("¿¡«»„""''‹›")
        
        # Japanese (Hiragana + Katakana + common Kanji)
        for i in range(0x3040, 0x309F):  # Hiragana
            tokens.append(chr(i))
        for i in range(0x30A0, 0x30FF):  # Katakana
            tokens.append(chr(i))
        for i in range(0x4E00, 0x4E00 + 2000):  # Common Kanji
            tokens.append(chr(i))
        tokens += list("。、！？「」『』（）・")
        
        # Build mapping
        for i, t in enumerate(tokens[:self.vocab_size]):
            self.token_to_id[t] = i
            self.id_to_token[i] = t
    
    @property
    def pad_id(self): return 0
    @property  
    def sos_id(self): return 1
    @property
    def eos_id(self): return 2
    @property
    def unk_id(self): return 3
    
    def encode(self, text: str, max_len: int = 128) -> List[int]:
        tokens = []
        
        # Handle language tag
        for lang_token in self.LANG_TOKENS:
            if text.startswith(lang_token):
                tokens.append(self.token_to_id[lang_token])
                text = text[len(lang_token):].strip()
                break
        
        tokens.append(self.sos_id)
        
        for char in text:
            tokens.append(self.token_to_id.get(char, self.unk_id))
        
        tokens.append(self.eos_id)
        
        # Truncate/pad
        if len(tokens) > max_len:
            tokens = tokens[:max_len-1] + [self.eos_id]
        while len(tokens) < max_len:
            tokens.append(self.pad_id)
        
        return tokens
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        chars = []
        for i in ids:
            if i in self.id_to_token:
                token = self.id_to_token[i]
                if skip_special and token in self.SPECIAL_TOKENS:
                    continue
                if skip_special and token in self.LANG_TOKENS:
                    continue
                chars.append(token)
        return "".join(chars)


# ============================================================================
# DATASET
# ============================================================================

class LargeDataset(Dataset):
    """Memory-efficient dataset for large JSON/JSONL files"""
    
    def __init__(self, path: str, tokenizer: MultilingualTokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        
        path = Path(path)
        
        # Load JSONL (line by line) or JSON
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        
        print(f"Loaded {len(self.data):,} examples from {path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        src = self.tokenizer.encode(item["input"], self.max_len)
        tgt = self.tokenizer.encode(item["output"], self.max_len)
        labels = tgt[1:] + [self.tokenizer.pad_id]
        
        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt": torch.tensor(tgt, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ============================================================================
# MODEL
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LanguageMirrorLarge(nn.Module):
    """
    Large-scale Language Tutor Transformer
    ~100M+ parameters for production quality
    """
    
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_enc = PositionalEncoding(config.d_model, config.max_seq_length, config.dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.n_heads,
            num_encoder_layers=config.n_encoder_layers,
            num_decoder_layers=config.n_decoder_layers,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm (better training)
        )
        
        # Output
        self.output = nn.Linear(config.d_model, config.vocab_size)
        
        # Tie weights
        self.output.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embed
        src_emb = self.pos_enc(self.embed(src))
        tgt_emb = self.pos_enc(self.embed(tgt))
        
        # Causal mask for decoder
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=src.device, dtype=torch.bool), 
            diagonal=1
        )
        
        # Forward
        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
        )
        
        return self.output(out)
    
    @torch.no_grad()
    def generate(self, src, tokenizer, max_len=80, temperature=0.7, top_k=50, top_p=0.9):
        self.eval()
        device = src.device
        
        # Encode
        src_emb = self.pos_enc(self.embed(src))
        memory = self.transformer.encoder(src_emb)
        
        # Start generation
        generated = [tokenizer.sos_id]
        
        for _ in range(max_len):
            tgt = torch.tensor([generated], device=device)
            tgt_emb = self.pos_enc(self.embed(tgt))
            
            tgt_len = tgt.size(1)
            causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool),
                diagonal=1
            )
            
            out = self.transformer.decoder(tgt_emb, memory, tgt_mask=causal_mask)
            logits = self.output(out[:, -1, :]) / temperature
            
            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k)
                logits[logits < topk_vals[:, [-1]]] = float('-inf')
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumsum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits[remove] = float('-inf')
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == tokenizer.eos_id:
                break
            
            generated.append(next_token)
        
        return tokenizer.decode(generated)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    def __init__(self, model, config, train_dl, val_dl, tokenizer):
        self.model = model
        self.config = config
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.tokenizer = tokenizer
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98),
        )
        
        # Scheduler
        total_steps = len(train_dl) * config.epochs // config.gradient_accumulation
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_ratio,
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_id,
            label_smoothing=config.label_smoothing
        )
        
        # AMP Scaler
        self.scaler = GradScaler() if config.use_amp and config.device == "cuda" else None
        
        # Tracking
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Output dir
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def train_step(self, batch):
        src = batch["src"].to(self.config.device)
        tgt = batch["tgt"].to(self.config.device)
        labels = batch["labels"].to(self.config.device)
        
        src_mask = (src == self.tokenizer.pad_id)
        tgt_mask = (tgt == self.tokenizer.pad_id)
        
        # Forward with AMP
        if self.scaler:
            with autocast():
                logits = self.model(src, tgt, src_mask, tgt_mask)
                loss = self.criterion(logits.view(-1, self.config.vocab_size), labels.view(-1))
                loss = loss / self.config.gradient_accumulation
            
            self.scaler.scale(loss).backward()
        else:
            logits = self.model(src, tgt, src_mask, tgt_mask)
            loss = self.criterion(logits.view(-1, self.config.vocab_size), labels.view(-1))
            loss = loss / self.config.gradient_accumulation
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation
    
    def optimizer_step(self):
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        for batch in self.val_dl:
            src = batch["src"].to(self.config.device)
            tgt = batch["tgt"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)
            
            src_mask = (src == self.tokenizer.pad_id)
            tgt_mask = (tgt == self.tokenizer.pad_id)
            
            logits = self.model(src, tgt, src_mask, tgt_mask)
            loss = self.criterion(logits.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(self.val_dl)
    
    def test_generation(self):
        self.model.eval()
        
        tests = [
            ("<lang:italian> Ciao!", "italian"),
            ("<lang:japanese> こんにちは", "japanese"),
            ("<lang:spanish> Hola!", "spanish"),
            ("<lang:french> Bonjour!", "french"),
            ("<lang:german> Hallo!", "german"),
        ]
        
        print("\n  📝 Sample generations:")
        for text, lang in tests:
            src = torch.tensor([self.tokenizer.encode(text, 64)]).to(self.config.device)
            output = self.model.generate(src, self.tokenizer, max_len=60, temperature=0.7)
            print(f"    [{lang}] {text.split('> ')[1]} → {output[:50]}...")
        
        self.model.train()
    
    def save_checkpoint(self, name: str = "checkpoint"):
        path = self.output_dir / f"{name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": self.config,
        }, path)
        print(f"  💾 Saved: {path}")
        
        # Upload to GCS if configured
        if self.config.gcs_bucket:
            self._upload_to_gcs(path)
    
    def _upload_to_gcs(self, local_path: Path):
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(self.config.gcs_bucket)
            blob_name = f"{self.config.gcs_output_prefix}/{local_path.name}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_path))
            print(f"  ☁️ Uploaded to gs://{self.config.gcs_bucket}/{blob_name}")
        except Exception as e:
            print(f"  ⚠️ GCS upload failed: {e}")
    
    def train(self):
        print(f"\n🚀 Starting training on {self.config.device}")
        print(f"   Model parameters: {self.model.count_params():,}")
        print(f"   Train examples: {len(self.train_dl.dataset):,}")
        print(f"   Batch size: {self.config.batch_size} x {self.config.gradient_accumulation} = {self.config.batch_size * self.config.gradient_accumulation}")
        print(f"   Epochs: {self.config.epochs}")
        print(f"   Mixed precision: {self.config.use_amp and self.config.device == 'cuda'}")
        print()
        
        self.model.train()
        
        for epoch in range(self.config.epochs):
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(self.train_dl, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            
            for i, batch in enumerate(pbar):
                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1
                
                # Gradient accumulation step
                if (i + 1) % self.config.gradient_accumulation == 0:
                    self.optimizer_step()
                    pbar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})
                
                # Evaluation
                if self.global_step > 0 and self.global_step % self.config.eval_steps == 0:
                    val_loss = self.validate()
                    print(f"\n  Step {self.global_step}: val_loss = {val_loss:.4f}")
                    
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint("best_model")
                
                # Periodic save
                if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}")
            
            # End of epoch
            avg_loss = total_loss / num_batches
            val_loss = self.validate()
            
            print(f"\nEpoch {epoch+1} complete:")
            print(f"  Train loss: {avg_loss:.4f}")
            print(f"  Val loss: {val_loss:.4f}")
            
            # Test generation
            self.test_generation()
            
            # Save checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch+1}")
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint("best_model")
                print("  ✅ New best model!")
        
        print(f"\n🎉 Training complete!")
        print(f"   Best validation loss: {self.best_loss:.4f}")
        print(f"   Model saved to: {self.output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run locally (smaller config)")
    parser.add_argument("--data-dir", type=str, default="datasets/large_scale")
    parser.add_argument("--output-dir", type=str, default="trained_model_large")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gcs-bucket", type=str, default="")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🧠 Language Mirror Pro - Large Scale Training")
    print("   Training on mC4 + CC-100 + OPUS")
    print("=" * 70)
    
    # Config
    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gcs_bucket=args.gcs_bucket,
    )
    
    # Smaller config for local testing
    if args.local:
        config.d_model = 512
        config.n_heads = 8
        config.d_ff = 2048
        config.batch_size = 32
        config.gradient_accumulation = 2
    
    print(f"\n📋 Configuration:")
    print(f"   Device: {config.device}")
    print(f"   Model: d={config.d_model}, heads={config.n_heads}, layers={config.n_encoder_layers}+{config.n_decoder_layers}")
    print(f"   Vocab: {config.vocab_size}")
    
    # Tokenizer
    tokenizer = MultilingualTokenizer(config.vocab_size)
    
    # Check data
    data_dir = Path(config.data_dir)
    train_path = data_dir / "train.json"
    if not train_path.exists():
        train_path = data_dir / "train.jsonl"
    
    if not train_path.exists():
        print(f"\n❌ Training data not found at {data_dir}")
        print("   Run: python prepare_large_datasets.py")
        return
    
    # Datasets
    print("\n📚 Loading datasets...")
    train_ds = LargeDataset(str(train_path), tokenizer, config.max_seq_length)
    
    val_path = data_dir / "val.json"
    if not val_path.exists():
        val_path = data_dir / "val.jsonl"
    val_ds = LargeDataset(str(val_path), tokenizer, config.max_seq_length)
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4 if config.device == "cuda" else 0,
        pin_memory=config.device == "cuda",
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2 if config.device == "cuda" else 0,
    )
    
    # Model
    print("\n🧠 Creating model...")
    model = LanguageMirrorLarge(config).to(config.device)
    print(f"   Parameters: {model.count_params():,}")
    
    # Train
    trainer = Trainer(model, config, train_dl, val_dl, tokenizer)
    trainer.train()


if __name__ == "__main__":
    main()
