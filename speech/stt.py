"""
🎤 Language Mirror Pro - Speech-to-Text
========================================
Local speech recognition using OpenAI's Whisper model.
No API keys needed - runs entirely on your machine!

Supports:
- Multiple languages
- Real-time transcription
- Pronunciation analysis
"""

import os
import io
import tempfile
import numpy as np
from typing import Optional, Dict, Tuple, List
from pathlib import Path

# We'll use whisper if available, otherwise provide fallback
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ Whisper not installed. Run: pip install openai-whisper")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


class SpeechToText:
    """
    Local Speech-to-Text using Whisper.
    
    Models available:
    - tiny: ~39M parameters, fastest
    - base: ~74M parameters, good balance
    - small: ~244M parameters, better accuracy
    - medium: ~769M parameters, high accuracy
    - large: ~1.5B parameters, best accuracy
    """
    
    # Language codes for Whisper
    LANGUAGE_CODES = {
        "italian": "it",
        "japanese": "ja",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "portuguese": "pt",
        "mandarin": "zh",
        "korean": "ko",
        "arabic": "ar",
        "hindi": "hi",
        "english": "en"
    }
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto"
    ):
        """
        Initialize Whisper model.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            device: Device to use (auto, cpu, cuda, mps)
        """
        self.model_size = model_size
        self.model = None
        self.device = device
        
        if WHISPER_AVAILABLE:
            self._load_model()
        else:
            print("⚠️ Whisper not available. Using fallback mode.")
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            import torch
            
            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            
            print(f"🎤 Loading Whisper {self.model_size} model on {self.device}...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print(f"✅ Whisper model loaded!")
            
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            self.model = None
    
    def transcribe(
        self,
        audio_path: str = None,
        audio_data: bytes = None,
        language: str = None,
        task: str = "transcribe"  # "transcribe" or "translate"
    ) -> Dict:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            audio_data: Raw audio bytes
            language: Target language (optional, auto-detected if not provided)
            task: "transcribe" for same language, "translate" for English translation
            
        Returns:
            Dict with transcription results
        """
        if not WHISPER_AVAILABLE or self.model is None:
            return self._fallback_transcribe(audio_path, audio_data)
        
        try:
            # Handle audio data
            if audio_data:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_data)
                    audio_path = f.name
            
            # Get language code
            lang_code = None
            if language:
                lang_code = self.LANGUAGE_CODES.get(language.lower(), language)
            
            # Transcribe
            options = {
                "task": task,
                "fp16": self.device == "cuda"
            }
            if lang_code:
                options["language"] = lang_code
            
            result = self.model.transcribe(audio_path, **options)
            
            # Clean up temp file
            if audio_data and os.path.exists(audio_path):
                os.remove(audio_path)
            
            return {
                "success": True,
                "text": result["text"].strip(),
                "language": result.get("language", lang_code),
                "segments": result.get("segments", []),
                "confidence": self._estimate_confidence(result)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }
    
    def _estimate_confidence(self, result: Dict) -> float:
        """Estimate transcription confidence from segments"""
        if not result.get("segments"):
            return 0.8
        
        # Average no_speech_prob (lower is better)
        no_speech_probs = [s.get("no_speech_prob", 0) for s in result["segments"]]
        avg_no_speech = sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else 0
        
        return max(0.0, min(1.0, 1.0 - avg_no_speech))
    
    def _fallback_transcribe(self, audio_path: str, audio_data: bytes) -> Dict:
        """Fallback when Whisper is not available"""
        return {
            "success": False,
            "error": "Whisper not available. Install with: pip install openai-whisper",
            "text": "",
            "fallback": True
        }
    
    def transcribe_realtime(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        language: str = None
    ) -> Dict:
        """
        Transcribe audio chunk for real-time processing.
        
        Args:
            audio_chunk: Numpy array of audio samples
            sample_rate: Audio sample rate
            language: Target language
            
        Returns:
            Dict with transcription results
        """
        if not WHISPER_AVAILABLE or self.model is None:
            return {"success": False, "text": "", "error": "Whisper not available"}
        
        try:
            # Resample if needed
            if sample_rate != 16000:
                # Simple resampling (for production, use librosa or scipy)
                ratio = 16000 / sample_rate
                audio_chunk = np.interp(
                    np.arange(0, len(audio_chunk), 1/ratio),
                    np.arange(len(audio_chunk)),
                    audio_chunk
                )
            
            # Pad or trim to 30 seconds
            audio_chunk = whisper.pad_or_trim(audio_chunk.astype(np.float32))
            
            # Get mel spectrogram
            mel = whisper.log_mel_spectrogram(audio_chunk).to(self.device)
            
            # Detect language if not provided
            if not language:
                _, probs = self.model.detect_language(mel)
                language = max(probs, key=probs.get)
            
            # Decode
            options = whisper.DecodingOptions(
                language=self.LANGUAGE_CODES.get(language, language),
                fp16=self.device == "cuda"
            )
            result = whisper.decode(self.model, mel, options)
            
            return {
                "success": True,
                "text": result.text.strip(),
                "language": language,
                "confidence": 1.0 - result.no_speech_prob
            }
            
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }


class PronunciationAnalyzer:
    """
    Analyze pronunciation quality by comparing expected vs actual phonemes.
    """
    
    def __init__(self, stt: SpeechToText = None):
        self.stt = stt or SpeechToText(model_size="base")
    
    def analyze(
        self,
        audio_path: str = None,
        audio_data: bytes = None,
        expected_text: str = "",
        language: str = "italian"
    ) -> Dict:
        """
        Analyze pronunciation quality.
        
        Args:
            audio_path: Path to audio file
            audio_data: Raw audio bytes
            expected_text: What the user was supposed to say
            language: Target language
            
        Returns:
            Dict with pronunciation analysis
        """
        # Transcribe audio
        result = self.stt.transcribe(
            audio_path=audio_path,
            audio_data=audio_data,
            language=language
        )
        
        if not result["success"]:
            return {
                "success": False,
                "error": result.get("error", "Transcription failed"),
                "score": 0.0
            }
        
        transcribed = result["text"].lower().strip()
        expected = expected_text.lower().strip()
        
        # Calculate similarity score
        score = self._calculate_similarity(transcribed, expected)
        
        # Generate feedback
        feedback = self._generate_feedback(transcribed, expected, score, language)
        
        return {
            "success": True,
            "transcribed": result["text"],
            "expected": expected_text,
            "score": score,
            "feedback": feedback,
            "confidence": result.get("confidence", 0.8)
        }
    
    def _calculate_similarity(self, transcribed: str, expected: str) -> float:
        """Calculate similarity between transcribed and expected text"""
        if not expected:
            return 0.8  # Default score if no expected text
        
        # Simple Levenshtein-based similarity
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(transcribed, expected)
        max_len = max(len(transcribed), len(expected), 1)
        similarity = 1.0 - (distance / max_len)
        
        return max(0.0, min(1.0, similarity))
    
    def _generate_feedback(
        self,
        transcribed: str,
        expected: str,
        score: float,
        language: str
    ) -> str:
        """Generate pronunciation feedback"""
        if score >= 0.9:
            feedback_templates = {
                "italian": "Eccellente! La tua pronuncia è perfetta!",
                "japanese": "素晴らしい！発音が完璧です！",
                "spanish": "¡Excelente! Tu pronunciación es perfecta!",
                "french": "Excellent! Votre prononciation est parfaite!",
                "german": "Ausgezeichnet! Ihre Aussprache ist perfekt!"
            }
        elif score >= 0.7:
            feedback_templates = {
                "italian": "Molto bene! Quasi perfetto, continua così!",
                "japanese": "とても良いです！もう少しで完璧です！",
                "spanish": "¡Muy bien! Casi perfecto, sigue así!",
                "french": "Très bien! Presque parfait, continuez!",
                "german": "Sehr gut! Fast perfekt, weiter so!"
            }
        elif score >= 0.5:
            feedback_templates = {
                "italian": "Buon tentativo! Prova a pronunciare più lentamente.",
                "japanese": "良い試みです！もう少しゆっくり発音してみてください。",
                "spanish": "¡Buen intento! Intenta pronunciar más despacio.",
                "french": "Bon essai! Essayez de prononcer plus lentement.",
                "german": "Guter Versuch! Versuchen Sie, langsamer zu sprechen."
            }
        else:
            feedback_templates = {
                "italian": "Riproviamo insieme! Ascolta e ripeti.",
                "japanese": "もう一度一緒に練習しましょう！",
                "spanish": "¡Intentemos de nuevo juntos! Escucha y repite.",
                "french": "Réessayons ensemble! Écoutez et répétez.",
                "german": "Versuchen wir es noch einmal! Hören Sie zu und wiederholen Sie."
            }
        
        return feedback_templates.get(language, feedback_templates["italian"])


# Quick test
if __name__ == "__main__":
    print("🎤 Testing Speech-to-Text...")
    print("=" * 60)
    
    stt = SpeechToText(model_size="tiny")  # Use tiny for quick test
    
    if WHISPER_AVAILABLE and stt.model:
        print("✅ Whisper model loaded successfully!")
        print(f"   Model size: {stt.model_size}")
        print(f"   Device: {stt.device}")
    else:
        print("⚠️ Whisper not available - install with: pip install openai-whisper")
    
    print("\n" + "=" * 60)
    print("🎉 Speech-to-Text module ready!")
