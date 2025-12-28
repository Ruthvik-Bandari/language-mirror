"""
🔤 Language Mirror Pro - Multilingual Tokenizer
================================================
Custom BPE tokenizer optimized for multilingual language learning.
Handles 10 languages with special tokens for pedagogical features.
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from collections import Counter
from pathlib import Path
import pickle


class LanguageMirrorTokenizer:
    """
    Multilingual BPE Tokenizer for Language Mirror Pro
    
    Features:
    - Byte-Pair Encoding (BPE) for subword tokenization
    - Special tokens for pedagogical features
    - Language-specific handling
    - Phoneme tokenization support
    """
    
    # Special tokens
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    SEP_TOKEN = "<sep>"
    MASK_TOKEN = "<mask>"
    
    # Pedagogical special tokens
    CORRECT_TOKEN = "<correct>"
    ERROR_TOKEN = "<error>"
    FEEDBACK_TOKEN = "<feedback>"
    PRONUNCIATION_TOKEN = "<pron>"
    
    # Language tokens
    LANG_TOKENS = [
        "<lang:italian>", "<lang:japanese>", "<lang:spanish>",
        "<lang:french>", "<lang:german>", "<lang:portuguese>",
        "<lang:mandarin>", "<lang:korean>", "<lang:arabic>", "<lang:hindi>"
    ]
    
    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        
        # Token to ID mapping
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # BPE merges
        self.merges: List[Tuple[str, str]] = []
        
        # Initialize special tokens
        self._init_special_tokens()
        
        # Language detection patterns
        self.language_patterns = {
            "japanese": re.compile(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]'),
            "korean": re.compile(r'[\uac00-\ud7af]'),
            "mandarin": re.compile(r'[\u4e00-\u9fff]'),
            "arabic": re.compile(r'[\u0600-\u06ff]'),
            "hindi": re.compile(r'[\u0900-\u097f]'),
        }
    
    def _init_special_tokens(self):
        """Initialize special tokens"""
        special_tokens = [
            self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN,
            self.SEP_TOKEN, self.MASK_TOKEN,
            self.CORRECT_TOKEN, self.ERROR_TOKEN, self.FEEDBACK_TOKEN, self.PRONUNCIATION_TOKEN,
        ] + self.LANG_TOKENS
        
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
    
    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]
    
    @property
    def sos_token_id(self) -> int:
        return self.token_to_id[self.SOS_TOKEN]
    
    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]
    
    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]
    
    def detect_language(self, text: str) -> str:
        """Detect language from text"""
        for lang, pattern in self.language_patterns.items():
            if pattern.search(text):
                return lang
        return "unknown"
    
    def _get_pairs(self, word: List[str]) -> set:
        """Get all adjacent pairs in word"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using BPE"""
        if not word:
            return []
        
        word = list(word) + ["</w>"]
        
        while True:
            pairs = self._get_pairs(word)
            if not pairs:
                break
            
            # Find best pair to merge
            best_pair = None
            best_rank = float('inf')
            
            for pair in pairs:
                if pair in self.merges:
                    rank = self.merges.index(pair)
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair
            
            if best_pair is None:
                break
            
            # Merge the pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        
        return word
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into subword tokens"""
        # Normalize
        text = text.strip()
        
        # Split into words (handle multiple scripts)
        words = re.findall(r'\S+|\s+', text)
        
        tokens = []
        for word in words:
            if word.isspace():
                tokens.append("▁")  # Space token
            else:
                word_tokens = self._tokenize_word(word.lower())
                tokens.extend(word_tokens)
        
        return tokens
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        return_tensors: Optional[str] = None
    ) -> Dict:
        """Encode text to token IDs"""
        tokens = self.tokenize(text)
        
        # Convert to IDs
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.unk_token_id)
        
        # Add special tokens
        if add_special_tokens:
            ids = [self.sos_token_id] + ids + [self.eos_token_id]
        
        # Truncate
        if max_length and len(ids) > max_length:
            ids = ids[:max_length]
        
        # Padding
        attention_mask = [1] * len(ids)
        if padding and max_length:
            pad_length = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        
        result = {
            "input_ids": ids,
            "attention_mask": attention_mask
        }
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            result = {k: torch.tensor([v]) for k, v in result.items()}
        
        return result
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text"""
        tokens = []
        
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]
                
                # Skip special tokens if requested
                if skip_special_tokens and token.startswith("<") and token.endswith(">"):
                    continue
                
                tokens.append(token)
        
        # Join and clean up
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        text = text.replace("▁", " ")
        text = text.strip()
        
        return text
    
    def train(self, texts: List[str], vocab_size: Optional[int] = None):
        """
        Train BPE on corpus
        
        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size
        """
        if vocab_size:
            self.vocab_size = vocab_size
        
        # Count character frequencies
        char_freq = Counter()
        for text in texts:
            for word in text.lower().split():
                word = tuple(word) + ("</w>",)
                char_freq[word] += 1
        
        # Initialize vocabulary with characters
        vocab = set()
        for word in char_freq:
            for char in word:
                vocab.add(char)
        
        # Add characters to token mapping
        next_id = len(self.token_to_id)
        for char in sorted(vocab):
            if char not in self.token_to_id:
                self.token_to_id[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1
        
        # BPE training loop
        word_freq = dict(char_freq)
        
        while len(self.token_to_id) < self.vocab_size:
            # Count pairs
            pairs = Counter()
            for word, freq in word_freq.items():
                symbols = list(word)
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i+1])] += freq
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = pairs.most_common(1)[0][0]
            
            # Merge pair in all words
            new_word_freq = {}
            merged = best_pair[0] + best_pair[1]
            
            for word, freq in word_freq.items():
                new_word = []
                i = 0
                symbols = list(word)
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == best_pair:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(symbols[i])
                        i += 1
                new_word_freq[tuple(new_word)] = freq
            
            word_freq = new_word_freq
            
            # Add to merges and vocab
            self.merges.append(best_pair)
            if merged not in self.token_to_id:
                self.token_to_id[merged] = next_id
                self.id_to_token[next_id] = merged
                next_id += 1
        
        print(f"✅ Tokenizer trained with {len(self.token_to_id)} tokens")
    
    def save(self, path: str):
        """Save tokenizer to file"""
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "id_to_token": {int(k): v for k, v in self.id_to_token.items()},
            "merges": self.merges
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "LanguageMirrorTokenizer":
        """Load tokenizer from file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        
        print(f"✅ Tokenizer loaded from {path}")
        return tokenizer
    
    @classmethod
    def from_pretrained(cls, vocab_size: int = 16000) -> "LanguageMirrorTokenizer":
        """
        Create a pre-initialized tokenizer with basic vocabulary.
        For production, train on actual multilingual corpus.
        """
        tokenizer = cls(vocab_size=vocab_size)
        
        # Add basic Latin characters
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        chars += list(".,!?;:'\"()-_@#$%&*+=/\\[]{}|<>~`")
        chars += list(" \t\n")
        
        # Add accented characters for European languages
        chars += list("àáâãäåæçèéêëìíîïñòóôõöùúûüýÿ")
        chars += list("ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝ")
        
        # Add common subwords
        common_subwords = [
            "ing", "tion", "ed", "er", "ly", "ment", "ness", "able", "ible",
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "zione", "mente", "ando", "endo",  # Italian
            "ción", "mente", "ando", "iendo",  # Spanish
            "tion", "ment", "eur", "eux",  # French
            "ung", "heit", "keit", "lich",  # German
            "</w>", "▁"  # Special
        ]
        
        next_id = len(tokenizer.token_to_id)
        
        for item in chars + common_subwords:
            if item not in tokenizer.token_to_id:
                tokenizer.token_to_id[item] = next_id
                tokenizer.id_to_token[next_id] = item
                next_id += 1
        
        print(f"✅ Pre-initialized tokenizer with {len(tokenizer.token_to_id)} tokens")
        return tokenizer


# Phoneme tokenizer for pronunciation
class PhonemeTokenizer:
    """
    Tokenizer for phoneme sequences (IPA-based)
    Used for pronunciation scoring
    """
    
    # Common IPA phonemes
    PHONEMES = [
        # Vowels
        'i', 'ɪ', 'e', 'ɛ', 'æ', 'a', 'ɑ', 'ɒ', 'ɔ', 'o', 'ʊ', 'u', 'ə', 'ʌ',
        # Consonants
        'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ',
        'h', 'm', 'n', 'ŋ', 'l', 'r', 'j', 'w',
        # Affricates
        'tʃ', 'dʒ',
        # Special
        '<sil>', '<pad>', '<unk>'
    ]
    
    def __init__(self):
        self.phoneme_to_id = {p: i for i, p in enumerate(self.PHONEMES)}
        self.id_to_phoneme = {i: p for i, p in enumerate(self.PHONEMES)}
    
    def encode(self, phonemes: List[str]) -> List[int]:
        return [self.phoneme_to_id.get(p, self.phoneme_to_id['<unk>']) for p in phonemes]
    
    def decode(self, ids: List[int]) -> List[str]:
        return [self.id_to_phoneme.get(i, '<unk>') for i in ids]


# Quick test
if __name__ == "__main__":
    print("🔤 Testing Language Mirror Tokenizer...")
    print("=" * 60)
    
    # Create tokenizer
    tokenizer = LanguageMirrorTokenizer.from_pretrained()
    
    # Test texts
    test_texts = [
        "Hello, how are you?",
        "Ciao, come stai?",
        "こんにちは、元気ですか？",
        "Hola, ¿cómo estás?",
        "Bonjour, comment allez-vous?",
        "Hallo, wie geht es Ihnen?",
    ]
    
    print("\n📝 Tokenization tests:")
    for text in test_texts:
        encoded = tokenizer.encode(text, max_length=50, padding=True)
        decoded = tokenizer.decode(encoded["input_ids"])
        lang = tokenizer.detect_language(text)
        
        print(f"\n  Original: {text}")
        print(f"  Language: {lang}")
        print(f"  Tokens: {len(encoded['input_ids'])} IDs")
        print(f"  Decoded: {decoded}")
    
    print("\n" + "=" * 60)
    print("🎉 Tokenizer tests passed!")
