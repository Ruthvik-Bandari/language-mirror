"""
📚 Language Mirror Pro - Large Scale Dataset Preparation
=========================================================
Download and prepare mC4 and CC-100 for training on Google Cloud.

Datasets:
- mC4: Multilingual C4 (used to train mT5) - 101 languages
- CC-100: CommonCrawl monolingual data - 100 languages
- OPUS: Parallel translations

These are MASSIVE datasets - we stream and sample efficiently.

Requirements:
    pip install datasets huggingface_hub google-cloud-storage tqdm

Usage:
    python prepare_large_datasets.py --upload-to-gcs --bucket YOUR_BUCKET
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Generator
from tqdm import tqdm
from itertools import islice

try:
    from datasets import load_dataset, interleave_datasets
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("❌ Install: pip install datasets huggingface_hub")

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

# ============================================================================
# CONFIG
# ============================================================================

OUTPUT_DIR = Path("datasets/large_scale")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Languages and their codes
LANGUAGES = {
    "italian": {"mc4": "it", "cc100": "it", "name": "Italian"},
    "japanese": {"mc4": "ja", "cc100": "ja", "name": "Japanese"},
    "spanish": {"mc4": "es", "cc100": "es", "name": "Spanish"},
    "french": {"mc4": "fr", "cc100": "fr", "name": "French"},
    "german": {"mc4": "de", "cc100": "de", "name": "German"},
}

# How many examples per language per dataset
SAMPLES_PER_LANG = {
    "mc4": 100000,      # 100K per language from mC4
    "cc100": 100000,    # 100K per language from CC-100
    "opus": 50000,      # 50K per language from OPUS
}


# ============================================================================
# mC4 DATASET
# ============================================================================

def download_mc4(lang_code: str, lang_name: str, num_samples: int) -> List[Dict]:
    """
    Download mC4 (multilingual C4) dataset
    
    mC4 is the multilingual version of C4 (Colossal Clean Crawled Corpus)
    Used to train mT5, contains 101 languages
    """
    print(f"  📥 mC4 ({lang_name})...")
    
    examples = []
    
    try:
        # Stream to avoid downloading entire dataset
        dataset = load_dataset(
            "mc4",
            lang_code,
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        count = 0
        for item in tqdm(dataset, total=num_samples, desc=f"    mC4-{lang_code}"):
            text = item.get("text", "")
            
            # Filter: reasonable length, not too short, not too long
            if 20 < len(text) < 500:
                # Clean text
                text = text.strip()
                text = " ".join(text.split())  # Normalize whitespace
                
                # Skip if too many special chars or numbers
                alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
                if alpha_ratio < 0.7:
                    continue
                
                examples.append({
                    "text": text,
                    "language": lang_name,
                    "source": "mc4"
                })
                
                count += 1
                if count >= num_samples:
                    break
        
        print(f"    ✅ Got {len(examples)} examples")
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
    
    return examples


# ============================================================================
# CC-100 DATASET  
# ============================================================================

def download_cc100(lang_code: str, lang_name: str, num_samples: int) -> List[Dict]:
    """
    Download CC-100 (CommonCrawl) dataset
    
    CC-100 is monolingual data from CommonCrawl, used to train XLM-R
    High quality, 100 languages
    """
    print(f"  📥 CC-100 ({lang_name})...")
    
    examples = []
    
    try:
        dataset = load_dataset(
            "cc100",
            lang=lang_code,
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        count = 0
        for item in tqdm(dataset, total=num_samples, desc=f"    CC100-{lang_code}"):
            text = item.get("text", "")
            
            # Filter by length
            if 30 < len(text) < 400:
                text = text.strip()
                text = " ".join(text.split())
                
                # Quality filter
                if len(text) > 30:
                    examples.append({
                        "text": text,
                        "language": lang_name,
                        "source": "cc100"
                    })
                    
                    count += 1
                    if count >= num_samples:
                        break
        
        print(f"    ✅ Got {len(examples)} examples")
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
    
    return examples


# ============================================================================
# OPUS PARALLEL DATA
# ============================================================================

def download_opus(lang_code: str, lang_name: str, num_samples: int) -> List[Dict]:
    """
    Download OPUS parallel translations (English <-> Target)
    
    This gives us translation pairs for supervised training
    """
    print(f"  📥 OPUS ({lang_name})...")
    
    examples = []
    
    try:
        pair = f"en-{lang_code}"
        
        dataset = load_dataset(
            "opus100",
            pair,
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        count = 0
        for item in tqdm(dataset, total=num_samples, desc=f"    OPUS-{pair}"):
            trans = item.get("translation", {})
            en_text = trans.get("en", "")
            tgt_text = trans.get(lang_code, "")
            
            if en_text and tgt_text and 10 < len(tgt_text) < 300:
                examples.append({
                    "en_text": en_text.strip(),
                    "target_text": tgt_text.strip(),
                    "language": lang_name,
                    "source": "opus"
                })
                
                count += 1
                if count >= num_samples:
                    break
        
        print(f"    ✅ Got {len(examples)} examples")
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
    
    return examples


# ============================================================================
# DATA CONVERSION
# ============================================================================

def convert_to_training_format(
    mc4_data: List[Dict],
    cc100_data: List[Dict],
    opus_data: List[Dict]
) -> List[Dict]:
    """
    Convert all data to unified training format:
    
    Format: {"input": "<lang:X> text", "output": "response", "language": "X"}
    
    Training strategies:
    1. Monolingual continuation (mC4, CC-100)
    2. Translation (OPUS)
    3. Conversation simulation
    """
    print("\n🔧 Converting to training format...")
    
    training_data = []
    
    # 1. Monolingual data -> Text completion / continuation
    print("  Converting monolingual data...")
    for item in tqdm(mc4_data + cc100_data, desc="  Monolingual"):
        text = item["text"]
        lang = item["language"]
        
        # Split text into input/output for continuation training
        words = text.split()
        if len(words) > 6:
            split_point = random.randint(3, len(words) - 3)
            input_text = " ".join(words[:split_point])
            output_text = " ".join(words[split_point:])
            
            training_data.append({
                "input": f"<lang:{lang}> {input_text}",
                "output": output_text,
                "translation": "",
                "language": lang,
                "source": item["source"],
                "task": "continuation"
            })
    
    # 2. Translation data -> Direct translation training
    print("  Converting translation data...")
    for item in tqdm(opus_data, desc="  Translation"):
        lang = item["language"]
        
        # English -> Target language
        training_data.append({
            "input": f"<lang:{lang}> {item['en_text']}",
            "output": item["target_text"],
            "translation": item["en_text"],
            "language": lang,
            "source": "opus",
            "task": "translation"
        })
        
        # Also add target language echo (input in target, output in target)
        # This helps model respond naturally in the language
        if random.random() < 0.3:
            target_words = item["target_text"].split()
            if len(target_words) > 4:
                partial = " ".join(target_words[:len(target_words)//2])
                training_data.append({
                    "input": f"<lang:{lang}> {partial}",
                    "output": item["target_text"],
                    "translation": item["en_text"],
                    "language": lang,
                    "source": "opus_echo",
                    "task": "echo"
                })
    
    # 3. Create conversation-style examples from monolingual
    print("  Creating conversation examples...")
    conversation_templates = {
        "italian": [
            ("Ciao!", "Come stai?", "Come va?"),
            ("Grazie", "Prego!", "Di niente!"),
            ("Buongiorno", "Buongiorno! Come stai?"),
        ],
        "japanese": [
            ("こんにちは", "こんにちは！元気ですか？"),
            ("ありがとう", "どういたしまして！"),
            ("おはよう", "おはようございます！"),
        ],
        "spanish": [
            ("Hola", "¡Hola! ¿Cómo estás?"),
            ("Gracias", "¡De nada!"),
            ("Buenos días", "¡Buenos días! ¿Qué tal?"),
        ],
        "french": [
            ("Bonjour", "Bonjour ! Comment allez-vous ?"),
            ("Merci", "Je vous en prie !"),
            ("Salut", "Salut ! Ça va ?"),
        ],
        "german": [
            ("Hallo", "Hallo! Wie geht es Ihnen?"),
            ("Danke", "Bitte schön!"),
            ("Guten Tag", "Guten Tag! Wie geht's?"),
        ],
    }
    
    for lang, templates in conversation_templates.items():
        for template in templates:
            input_text = template[0]
            output_text = random.choice(template[1:]) if len(template) > 2 else template[1]
            
            # Add multiple times for emphasis
            for _ in range(1000):  # Repeat important patterns
                training_data.append({
                    "input": f"<lang:{lang}> {input_text}",
                    "output": output_text,
                    "translation": "",
                    "language": lang,
                    "source": "conversation",
                    "task": "conversation"
                })
    
    random.shuffle(training_data)
    return training_data


# ============================================================================
# DATA SPLITS & SAVING
# ============================================================================

def create_splits(data: List[Dict], train_ratio: float = 0.9):
    """Create train/val/test splits, balanced by language"""
    print("\n📊 Creating balanced splits...")
    
    # Group by language
    by_lang = {}
    for item in data:
        lang = item["language"]
        if lang not in by_lang:
            by_lang[lang] = []
        by_lang[lang].append(item)
    
    train, val, test = [], [], []
    
    for lang, items in by_lang.items():
        random.shuffle(items)
        n = len(items)
        
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + 0.05) * n)
        
        train.extend(items[:train_end])
        val.extend(items[train_end:val_end])
        test.extend(items[val_end:])
        
        print(f"  {lang}: {len(items[:train_end])} train, {len(items[train_end:val_end])} val, {len(items[val_end:])} test")
    
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    return train, val, test


def save_locally(train, val, test, output_dir: Path):
    """Save datasets locally as JSON"""
    print("\n💾 Saving locally...")
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save as JSONL for efficiency with large files
    for name, data in [("train", train), ("val", val), ("test", test)]:
        filepath = output_dir / f"{name}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  ✅ {name}: {len(data):,} examples -> {filepath}")
    
    # Also save as single JSON for compatibility
    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False)
    with open(output_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False)
    with open(output_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False)


def upload_to_gcs(local_dir: Path, bucket_name: str, gcs_prefix: str = "data"):
    """Upload dataset to Google Cloud Storage"""
    if not GCS_AVAILABLE:
        print("❌ google-cloud-storage not installed")
        return
    
    print(f"\n☁️ Uploading to gs://{bucket_name}/{gcs_prefix}/...")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    for filepath in local_dir.glob("*.json*"):
        blob_name = f"{gcs_prefix}/{filepath.name}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(filepath))
        print(f"  ✅ Uploaded: {blob_name}")
    
    print(f"  📁 Data available at: gs://{bucket_name}/{gcs_prefix}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc4-samples", type=int, default=100000, help="Samples per lang from mC4")
    parser.add_argument("--cc100-samples", type=int, default=100000, help="Samples per lang from CC-100")
    parser.add_argument("--opus-samples", type=int, default=50000, help="Samples per lang from OPUS")
    parser.add_argument("--output-dir", type=str, default="datasets/large_scale")
    parser.add_argument("--upload-to-gcs", action="store_true", help="Upload to GCS")
    parser.add_argument("--bucket", type=str, help="GCS bucket name")
    args = parser.parse_args()
    
    print("=" * 70)
    print("📚 Language Mirror Pro - Large Scale Dataset Preparation")
    print("   Downloading mC4 + CC-100 + OPUS for 5 languages")
    print("=" * 70)
    
    if not HF_AVAILABLE:
        print("\n❌ Install required: pip install datasets huggingface_hub")
        return
    
    all_mc4 = []
    all_cc100 = []
    all_opus = []
    
    # Download for each language
    for lang_name, codes in LANGUAGES.items():
        print(f"\n🌍 Processing {lang_name.upper()}...")
        
        # mC4
        mc4_data = download_mc4(codes["mc4"], lang_name, args.mc4_samples)
        all_mc4.extend(mc4_data)
        
        # CC-100
        cc100_data = download_cc100(codes["cc100"], lang_name, args.cc100_samples)
        all_cc100.extend(cc100_data)
        
        # OPUS (translations)
        opus_data = download_opus(codes["mc4"], lang_name, args.opus_samples)
        all_opus.extend(opus_data)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Download Summary:")
    print(f"   mC4: {len(all_mc4):,} examples")
    print(f"   CC-100: {len(all_cc100):,} examples")
    print(f"   OPUS: {len(all_opus):,} examples")
    print(f"   Total raw: {len(all_mc4) + len(all_cc100) + len(all_opus):,}")
    print("=" * 70)
    
    # Convert to training format
    training_data = convert_to_training_format(all_mc4, all_cc100, all_opus)
    print(f"\n📈 Total training examples: {len(training_data):,}")
    
    # Create splits
    train, val, test = create_splits(training_data)
    
    # Save
    output_dir = Path(args.output_dir)
    save_locally(train, val, test, output_dir)
    
    # Upload to GCS if requested
    if args.upload_to_gcs:
        if not args.bucket:
            print("\n❌ --bucket required for GCS upload")
        else:
            upload_to_gcs(output_dir, args.bucket)
    
    # Sample output
    print("\n📋 Sample examples:")
    for lang in ["italian", "japanese", "spanish"]:
        sample = next((x for x in train if x["language"] == lang), None)
        if sample:
            print(f"\n[{lang.upper()}] ({sample['task']})")
            print(f"  Input: {sample['input'][:70]}...")
            print(f"  Output: {sample['output'][:70]}...")
    
    print("\n" + "=" * 70)
    print("✅ Dataset preparation complete!")
    print(f"📁 Output: {output_dir}")
    print("\nNext: Train on GCP with:")
    print(f"  python train_large.py --data-dir {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
