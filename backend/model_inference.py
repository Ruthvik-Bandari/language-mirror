"""
🧠 Language Mirror Pro - Model Inference
=========================================
Load and use the trained 76.9M parameter model for conversation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_length: int = 256
    dropout: float = 0.1
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"


class SimpleTokenizer:
    """Character-level tokenizer matching training"""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self._init_vocab()
    
    def _init_vocab(self):
        special = ["<pad>", "<sos>", "<eos>", "<unk>"]
        lang_tokens = [f"<lang:{l}>" for l in ["italian", "japanese", "spanish", "french", "german"]]
        
        chars = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        chars += list(".,!?;:'\"()-_/\\@#$%^&*+=<>[]{}|`~")
        chars += list("àèìòùáéíóúâêîôûäëïöüñçß¿¡")
        
        hiragana = [chr(i) for i in range(0x3040, 0x309F)]
        katakana = [chr(i) for i in range(0x30A0, 0x30FF)]
        kanji = [chr(i) for i in range(0x4E00, 0x4E00 + 500)]
        
        all_tokens = special + lang_tokens + chars + hiragana + katakana + kanji
        
        for i, token in enumerate(all_tokens[:self.vocab_size]):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
    
    @property
    def pad_token_id(self): return 0
    @property
    def sos_token_id(self): return 1
    @property
    def eos_token_id(self): return 2
    @property
    def unk_token_id(self): return 3
    
    def get_lang_token_id(self, language: str) -> int:
        """Get the token ID for a language"""
        lang_token = f"<lang:{language}>"
        return self.token_to_id.get(lang_token, self.unk_token_id)
    
    def encode(self, text: str, max_length: int = 256, add_special: bool = True) -> List[int]:
        """Encode text to token IDs"""
        tokens = []
        
        # Check for language token
        if text.startswith("<lang:"):
            end = text.find(">") + 1
            lang_token = text[:end]
            text = text[end:].strip()
            if lang_token in self.token_to_id:
                tokens.append(self.token_to_id[lang_token])
        
        if add_special:
            tokens.append(self.sos_token_id)
        
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.unk_token_id)
        
        if add_special:
            tokens.append(self.eos_token_id)
        
        # Truncate
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.eos_token_id]
        
        # Pad
        while len(tokens) < max_length:
            tokens.append(self.pad_token_id)
        
        return tokens
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text"""
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                if skip_special and token in ["<pad>", "<sos>", "<eos>", "<unk>"]:
                    continue
                if skip_special and token.startswith("<lang:"):
                    continue
                tokens.append(token)
        return "".join(tokens)


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


class LanguageTutorModel(nn.Module):
    """The trained 76.9M parameter model"""
    
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length, config.dropout)
        
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.n_heads,
            num_encoder_layers=config.n_layers,
            num_decoder_layers=config.n_layers,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.embedding(src)
        src_emb = self.pos_encoding(src_emb)
        
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        tgt_len = tgt.size(1)
        causal_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(src.device)
        
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask
        )
        
        return self.output_head(output)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        tokenizer: SimpleTokenizer,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """Generate response with proper sampling"""
        self.eval()
        device = input_ids.device
        
        # Encode input
        src = self.embedding(input_ids)
        src = self.pos_encoding(src)
        memory = self.transformer.encoder(src)
        
        # Start with SOS
        generated = [tokenizer.sos_token_id]
        
        with torch.no_grad():
            for _ in range(max_length):
                tgt = torch.tensor([generated], dtype=torch.long, device=device)
                tgt_emb = self.embedding(tgt)
                tgt_emb = self.pos_encoding(tgt_emb)
                
                tgt_len = tgt.size(1)
                causal_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(device)
                
                output = self.transformer.decoder(tgt_emb, memory, tgt_mask=causal_mask)
                logits = self.output_head(output[:, -1, :])
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Stop at EOS
                if next_token == tokenizer.eos_token_id:
                    break
                
                generated.append(next_token)
        
        return tokenizer.decode(generated)


class LanguageMirrorInference:
    """Main inference class for the backend"""
    
    def __init__(self, model_path: str = "trained_model/model_weights.pt"):
        self.config = InferenceConfig()
        self.tokenizer = SimpleTokenizer(self.config.vocab_size)
        self.model = None
        self.loaded = False
        
        # Response templates as fallback
        self.fallback_responses = {
            "italian": [
                ("Ciao! Come stai oggi?", "Hello! How are you today?"),
                ("Benissimo! Continua così!", "Very good! Keep it up!"),
                ("Interessante! Dimmi di più.", "Interesting! Tell me more."),
            ],
            "japanese": [
                ("こんにちは！元気ですか？", "Hello! How are you?"),
                ("すごいですね！続けてください！", "Amazing! Please continue!"),
                ("面白いですね！", "That's interesting!"),
            ],
            "spanish": [
                ("¡Hola! ¿Cómo estás?", "Hello! How are you?"),
                ("¡Muy bien! ¡Sigue así!", "Very good! Keep it up!"),
                ("¡Interesante! Cuéntame más.", "Interesting! Tell me more."),
            ],
            "french": [
                ("Bonjour! Comment allez-vous?", "Hello! How are you?"),
                ("Très bien! Continuez!", "Very good! Continue!"),
                ("Intéressant! Dites-m'en plus.", "Interesting! Tell me more."),
            ],
            "german": [
                ("Hallo! Wie geht es Ihnen?", "Hello! How are you?"),
                ("Sehr gut! Weiter so!", "Very good! Keep it up!"),
                ("Interessant! Erzählen Sie mehr.", "Interesting! Tell me more."),
            ],
        }
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load the trained model"""
        try:
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, weights_only=False, map_location=self.config.device)
                
                self.model = LanguageTutorModel(self.config)
                
                # Handle different checkpoint formats
                if "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.to(self.config.device)
                self.model.eval()
                self.loaded = True
                print(f"✅ Model loaded from {model_path}")
                print(f"   Device: {self.config.device}")
                print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            else:
                print(f"⚠️ Model not found at {model_path}")
                self.loaded = False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.loaded = False
    
    def generate_response(
        self,
        user_input: str,
        language: str = "italian",
        temperature: float = 0.8,
        use_model: bool = True
    ) -> Dict[str, str]:
        """Generate a response for the user input"""
        
        # Try model generation
        if self.loaded and use_model and self.model is not None:
            try:
                # Prepare input with language token
                full_input = f"<lang:{language}> {user_input}"
                input_ids = self.tokenizer.encode(full_input, max_length=128)
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.config.device)
                
                # Generate
                response = self.model.generate(
                    input_tensor,
                    self.tokenizer,
                    max_length=100,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.9
                )
                
                # Clean up response
                response = response.strip()
                
                if len(response) > 5:  # Valid response
                    return {
                        "response": response,
                        "translation": self._get_translation(response, language),
                        "source": "model"
                    }
            except Exception as e:
                print(f"Generation error: {e}")
        
        # Fallback to templates
        import random
        responses = self.fallback_responses.get(language, self.fallback_responses["italian"])
        response, translation = random.choice(responses)
        
        return {
            "response": response,
            "translation": translation,
            "source": "fallback"
        }
    
    def _get_translation(self, text: str, language: str) -> str:
        """Simple translation lookup (in production, use translation API)"""
        # For now, return a placeholder
        return f"[Translation of: {text[:50]}...]"


# Quick test
if __name__ == "__main__":
    print("🧠 Testing Language Mirror Inference...")
    
    inference = LanguageMirrorInference("trained_model/model_weights.pt")
    
    test_inputs = [
        ("Ciao!", "italian"),
        ("こんにちは", "japanese"),
        ("Hola!", "spanish"),
        ("Bonjour!", "french"),
        ("Hallo!", "german"),
    ]
    
    for user_input, language in test_inputs:
        result = inference.generate_response(user_input, language, temperature=0.8)
        print(f"\n[{language.upper()}]")
        print(f"  Input: {user_input}")
        print(f"  Output: {result['response']}")
        print(f"  Source: {result['source']}")
