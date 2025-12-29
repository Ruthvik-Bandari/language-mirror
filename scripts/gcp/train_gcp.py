"""
🚀 Language Mirror Pro - Google Cloud Training
===============================================
Train the custom Transformer model on Vertex AI with GPUs.

Requirements:
- Google Cloud account with billing enabled
- gcloud CLI installed and configured
- Vertex AI API enabled

Usage:
1. Run locally: python train_gcp.py --local
2. Run on Vertex AI: python train_gcp.py --cloud
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    # Model
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_length: int = 256
    dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 10
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    
    # Data
    data_dir: str = "datasets/processed"
    output_dir: str = "trained_model"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Languages
    languages: List[str] = None
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["italian", "japanese", "spanish", "french", "german"]


# ============================================================================
# TOKENIZER (Simple for training)
# ============================================================================

class SimpleTokenizer:
    """Character-level tokenizer with special tokens"""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Special tokens
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        
        self._init_vocab()
    
    def _init_vocab(self):
        # Special tokens
        special = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        
        # Language tokens
        lang_tokens = [f"<lang:{l}>" for l in ["italian", "japanese", "spanish", "french", "german"]]
        
        # Basic ASCII + extended characters
        chars = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        chars += list(".,!?;:'\"()-_/\\@#$%^&*+=<>[]{}|`~")
        chars += list("àèìòùáéíóúâêîôûäëïöüñçß")  # European accents
        chars += list("¿¡")  # Spanish
        
        # Japanese hiragana (basic)
        hiragana = [chr(i) for i in range(0x3040, 0x309F)]
        # Japanese katakana (basic)  
        katakana = [chr(i) for i in range(0x30A0, 0x30FF)]
        # Common kanji
        kanji = [chr(i) for i in range(0x4E00, 0x4E00 + 500)]
        
        all_tokens = special + lang_tokens + chars + hiragana + katakana + kanji
        
        for i, token in enumerate(all_tokens[:self.vocab_size]):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
    
    @property
    def pad_token_id(self):
        return self.token_to_id[self.pad_token]
    
    @property
    def sos_token_id(self):
        return self.token_to_id[self.sos_token]
    
    @property
    def eos_token_id(self):
        return self.token_to_id[self.eos_token]
    
    def encode(self, text: str, max_length: int = 256) -> List[int]:
        """Encode text to token IDs"""
        # Check for language token at start
        tokens = []
        
        if text.startswith("<lang:"):
            # Extract language token
            end = text.find(">") + 1
            lang_token = text[:end]
            text = text[end:].strip()
            
            if lang_token in self.token_to_id:
                tokens.append(self.token_to_id[lang_token])
        
        # Add SOS
        tokens.append(self.sos_token_id)
        
        # Encode characters
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.token_to_id[self.unk_token])
        
        # Add EOS
        tokens.append(self.eos_token_id)
        
        # Truncate or pad
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.eos_token_id]
        
        while len(tokens) < max_length:
            tokens.append(self.pad_token_id)
        
        return tokens
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text"""
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                if skip_special and token in [self.pad_token, self.sos_token, self.eos_token]:
                    continue
                tokens.append(token)
        return "".join(tokens)


# ============================================================================
# DATASET
# ============================================================================

class ConversationDataset(Dataset):
    """Dataset for conversation training"""
    
    def __init__(self, data_path: str, tokenizer: SimpleTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Encode input (user message with language tag)
        input_ids = self.tokenizer.encode(item["input"], self.max_length)
        
        # Encode output (tutor response)
        output_ids = self.tokenizer.encode(item["output"], self.max_length)
        
        # Create labels (shift output for autoregressive training)
        labels = output_ids[1:] + [self.tokenizer.pad_token_id]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "output_ids": torch.tensor(output_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ============================================================================
# MODEL
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LanguageTutorTransformer(nn.Module):
    """
    Custom Transformer for Language Tutoring
    Encoder-Decoder architecture optimized for conversation
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length, config.dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.n_heads,
            num_encoder_layers=config.n_layers,
            num_decoder_layers=config.n_layers,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Output head
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        output_ids: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        output_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass"""
        # Embed
        src = self.embedding(input_ids)
        src = self.pos_encoding(src)
        
        tgt = self.embedding(output_ids)
        tgt = self.pos_encoding(tgt)
        
        # Create causal mask for decoder
        tgt_len = output_ids.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(input_ids.device)
        
        # Transformer forward
        output = self.transformer(
            src, tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=input_mask,
            tgt_key_padding_mask=output_mask
        )
        
        # Project to vocabulary
        logits = self.output_head(output)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        sos_token_id: int = 1,
        eos_token_id: int = 2
    ) -> torch.Tensor:
        """Generate response autoregressively"""
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Encode input
        src = self.embedding(input_ids)
        src = self.pos_encoding(src)
        memory = self.transformer.encoder(src)
        
        # Start with SOS token
        generated = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Embed generated tokens
                tgt = self.embedding(generated)
                tgt = self.pos_encoding(tgt)
                
                # Create causal mask
                tgt_len = generated.size(1)
                tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(device)
                
                # Decode
                output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
                logits = self.output_head(output[:, -1, :])
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-p sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if EOS
                if (next_token == eos_token_id).all():
                    break
        
        return generated
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict()
        }, path)
        print(f"✅ Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "LanguageTutorTransformer":
        """Load model"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        print(f"✅ Model loaded from {path}")
        return model


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """Training loop for the Language Tutor model"""
    
    def __init__(
        self,
        model: LanguageTutorTransformer,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        tokenizer: SimpleTokenizer
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        total_steps = len(train_dataloader) * config.epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        
        # Loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Best loss
        self.best_val_loss = float("inf")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            # Move to device
            input_ids = batch["input_ids"].to(self.config.device)
            output_ids = batch["output_ids"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)
            
            # Create padding masks
            input_mask = (input_ids == self.tokenizer.pad_token_id)
            output_mask = (output_ids == self.tokenizer.pad_token_id)
            
            # Forward
            logits = self.model(input_ids, output_ids, input_mask, output_mask)
            
            # Loss
            loss = self.criterion(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            # Update
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / len(self.train_dataloader)
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        for batch in self.val_dataloader:
            input_ids = batch["input_ids"].to(self.config.device)
            output_ids = batch["output_ids"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)
            
            input_mask = (input_ids == self.tokenizer.pad_token_id)
            output_mask = (output_ids == self.tokenizer.pad_token_id)
            
            logits = self.model(input_ids, output_ids, input_mask, output_mask)
            
            loss = self.criterion(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
            
            total_loss += loss.item()
        
        return total_loss / len(self.val_dataloader)
    
    def train(self):
        """Full training loop"""
        print(f"\n🚀 Starting training on {self.config.device}")
        print(f"   Model parameters: {self.model.count_parameters():,}")
        print(f"   Training examples: {len(self.train_dataloader.dataset):,}")
        print(f"   Validation examples: {len(self.val_dataloader.dataset):,}")
        print(f"   Epochs: {self.config.epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print()
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save(str(self.output_dir / "best_model.pt"))
                print(f"   ✅ New best model saved!")
            
            # Save checkpoint
            self.model.save(str(self.output_dir / f"checkpoint_epoch_{epoch+1}.pt"))
            
            # Test generation
            self.test_generation()
        
        print(f"\n🎉 Training complete!")
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        print(f"   Model saved to: {self.output_dir}")
    
    @torch.no_grad()
    def test_generation(self):
        """Test model generation"""
        self.model.eval()
        
        test_inputs = [
            "<lang:italian> Ciao!",
            "<lang:japanese> こんにちは",
            "<lang:spanish> Hola!",
            "<lang:french> Bonjour!",
            "<lang:german> Hallo!",
        ]
        
        print("\n   Sample generations:")
        for text in test_inputs[:2]:  # Just show 2
            input_ids = torch.tensor([self.tokenizer.encode(text, 64)]).to(self.config.device)
            
            output_ids = self.model.generate(
                input_ids,
                max_length=50,
                temperature=0.7,
                sos_token_id=self.tokenizer.sos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            output_text = self.tokenizer.decode(output_ids[0].tolist())
            print(f"   Input: {text}")
            print(f"   Output: {output_text[:100]}...")
            print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Language Mirror model")
    parser.add_argument("--local", action="store_true", help="Train locally")
    parser.add_argument("--cloud", action="store_true", help="Train on Google Cloud")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 Language Mirror Pro - Model Training")
    print("=" * 60)
    
    # Config
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    print(f"\n📋 Configuration:")
    print(f"   Device: {config.device}")
    print(f"   Model dim: {config.d_model}")
    print(f"   Layers: {config.n_layers}")
    print(f"   Heads: {config.n_heads}")
    print(f"   Vocab size: {config.vocab_size}")
    
    # Tokenizer
    print("\n🔤 Initializing tokenizer...")
    tokenizer = SimpleTokenizer(config.vocab_size)
    
    # Check if data exists
    data_dir = Path(config.data_dir)
    if not (data_dir / "train.json").exists():
        print("\n⚠️ Training data not found! Run prepare_dataset.py first:")
        print("   python prepare_dataset.py")
        return
    
    # Datasets
    print("\n📚 Loading datasets...")
    train_dataset = ConversationDataset(
        str(data_dir / "train.json"),
        tokenizer,
        config.max_seq_length
    )
    
    val_dataset = ConversationDataset(
        str(data_dir / "val.json"),
        tokenizer,
        config.max_seq_length
    )
    
    # Dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Model
    print("\n🧠 Creating model...")
    model = LanguageTutorTransformer(config)
    model.to(config.device)
    print(f"   Parameters: {model.count_parameters():,}")
    
    # Trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
