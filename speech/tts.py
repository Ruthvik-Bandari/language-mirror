"""
🔊 Language Mirror Pro - Text-to-Speech
========================================
High-quality TTS using Microsoft Edge TTS.
Free, no API keys, native accents for multiple languages!

Features:
- Native accents (Italian, Japanese, Spanish, etc.)
- Regional dialects
- Adjustable speed and pitch
- Async streaming support
"""

import os
import io
import asyncio
import tempfile
import base64
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass

# Try to import edge_tts
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("⚠️ edge-tts not installed. Run: pip install edge-tts")


@dataclass
class VoiceConfig:
    """Voice configuration"""
    voice_id: str
    name: str
    language: str
    locale: str
    gender: str


# Voice mapping for each language with regional variants
VOICE_MAP = {
    "italian": {
        "standard": VoiceConfig("it-IT-ElsaNeural", "Elsa", "italian", "it-IT", "Female"),
        "male": VoiceConfig("it-IT-DiegoNeural", "Diego", "italian", "it-IT", "Male"),
        "sicilian": VoiceConfig("it-IT-IsabellaNeural", "Isabella", "italian", "it-IT", "Female"),
        "roman": VoiceConfig("it-IT-DiegoNeural", "Diego", "italian", "it-IT", "Male"),
    },
    "japanese": {
        "standard": VoiceConfig("ja-JP-NanamiNeural", "Nanami", "japanese", "ja-JP", "Female"),
        "male": VoiceConfig("ja-JP-KeitaNeural", "Keita", "japanese", "ja-JP", "Male"),
        "osaka": VoiceConfig("ja-JP-NanamiNeural", "Nanami", "japanese", "ja-JP", "Female"),
    },
    "spanish": {
        "standard": VoiceConfig("es-ES-ElviraNeural", "Elvira", "spanish", "es-ES", "Female"),
        "male": VoiceConfig("es-ES-AlvaroNeural", "Alvaro", "spanish", "es-ES", "Male"),
        "mexican": VoiceConfig("es-MX-DaliaNeural", "Dalia", "spanish", "es-MX", "Female"),
        "argentinian": VoiceConfig("es-AR-ElenaNeural", "Elena", "spanish", "es-AR", "Female"),
    },
    "french": {
        "standard": VoiceConfig("fr-FR-DeniseNeural", "Denise", "french", "fr-FR", "Female"),
        "male": VoiceConfig("fr-FR-HenriNeural", "Henri", "french", "fr-FR", "Male"),
        "parisian": VoiceConfig("fr-FR-DeniseNeural", "Denise", "french", "fr-FR", "Female"),
        "quebec": VoiceConfig("fr-CA-SylvieNeural", "Sylvie", "french", "fr-CA", "Female"),
    },
    "german": {
        "standard": VoiceConfig("de-DE-KatjaNeural", "Katja", "german", "de-DE", "Female"),
        "male": VoiceConfig("de-DE-ConradNeural", "Conrad", "german", "de-DE", "Male"),
        "bavarian": VoiceConfig("de-DE-KatjaNeural", "Katja", "german", "de-DE", "Female"),
        "austrian": VoiceConfig("de-AT-IngridNeural", "Ingrid", "german", "de-AT", "Female"),
    },
    "portuguese": {
        "standard": VoiceConfig("pt-BR-FranciscaNeural", "Francisca", "portuguese", "pt-BR", "Female"),
        "male": VoiceConfig("pt-BR-AntonioNeural", "Antonio", "portuguese", "pt-BR", "Male"),
        "european": VoiceConfig("pt-PT-RaquelNeural", "Raquel", "portuguese", "pt-PT", "Female"),
    },
    "mandarin": {
        "standard": VoiceConfig("zh-CN-XiaoxiaoNeural", "Xiaoxiao", "mandarin", "zh-CN", "Female"),
        "male": VoiceConfig("zh-CN-YunxiNeural", "Yunxi", "mandarin", "zh-CN", "Male"),
    },
    "korean": {
        "standard": VoiceConfig("ko-KR-SunHiNeural", "SunHi", "korean", "ko-KR", "Female"),
        "male": VoiceConfig("ko-KR-InJoonNeural", "InJoon", "korean", "ko-KR", "Male"),
    },
    "arabic": {
        "standard": VoiceConfig("ar-SA-ZariyahNeural", "Zariyah", "arabic", "ar-SA", "Female"),
        "male": VoiceConfig("ar-SA-HamedNeural", "Hamed", "arabic", "ar-SA", "Male"),
    },
    "hindi": {
        "standard": VoiceConfig("hi-IN-SwaraNeural", "Swara", "hindi", "hi-IN", "Female"),
        "male": VoiceConfig("hi-IN-MadhurNeural", "Madhur", "hindi", "hi-IN", "Male"),
    },
    "english": {
        "standard": VoiceConfig("en-US-JennyNeural", "Jenny", "english", "en-US", "Female"),
        "male": VoiceConfig("en-US-GuyNeural", "Guy", "english", "en-US", "Male"),
        "british": VoiceConfig("en-GB-SoniaNeural", "Sonia", "english", "en-GB", "Female"),
        "australian": VoiceConfig("en-AU-NatashaNeural", "Natasha", "english", "en-AU", "Female"),
    },
}


class TextToSpeech:
    """
    High-quality Text-to-Speech using Edge TTS.
    
    Features:
    - Native accents for 10+ languages
    - Regional dialect support
    - Adjustable speech rate and pitch
    - Async audio generation
    """
    
    def __init__(self):
        self.voice_map = VOICE_MAP
        
        if not EDGE_TTS_AVAILABLE:
            print("⚠️ edge-tts not available. Install with: pip install edge-tts")
    
    def get_available_voices(self, language: str = None) -> List[Dict]:
        """Get available voices for a language"""
        if language:
            voices = self.voice_map.get(language.lower(), {})
            return [
                {
                    "id": v.voice_id,
                    "name": v.name,
                    "dialect": dialect,
                    "gender": v.gender
                }
                for dialect, v in voices.items()
            ]
        else:
            # Return all voices
            all_voices = []
            for lang, voices in self.voice_map.items():
                for dialect, v in voices.items():
                    all_voices.append({
                        "id": v.voice_id,
                        "name": v.name,
                        "language": lang,
                        "dialect": dialect,
                        "gender": v.gender
                    })
            return all_voices
    
    def get_voice(self, language: str, dialect: str = "standard") -> VoiceConfig:
        """Get voice config for language and dialect"""
        language = language.lower()
        dialect = dialect.lower() if dialect else "standard"
        
        voices = self.voice_map.get(language, self.voice_map["italian"])
        return voices.get(dialect, voices.get("standard"))
    
    async def synthesize_async(
        self,
        text: str,
        language: str = "italian",
        dialect: str = "standard",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        output_format: str = "audio-24khz-48kbitrate-mono-mp3"
    ) -> bytes:
        """
        Synthesize speech asynchronously.
        
        Args:
            text: Text to speak
            language: Target language
            dialect: Regional dialect
            rate: Speech rate (e.g., "+10%", "-20%")
            pitch: Voice pitch (e.g., "+5Hz", "-10Hz")
            output_format: Audio format
            
        Returns:
            Audio bytes
        """
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError("edge-tts not installed")
        
        voice_config = self.get_voice(language, dialect)
        
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice_config.voice_id,
            rate=rate,
            pitch=pitch
        )
        
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        return audio_data
    
    def synthesize(
        self,
        text: str,
        language: str = "italian",
        dialect: str = "standard",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        output_path: str = None
    ) -> bytes:
        """
        Synthesize speech synchronously.
        
        Args:
            text: Text to speak
            language: Target language
            dialect: Regional dialect
            rate: Speech rate
            pitch: Voice pitch
            output_path: Optional path to save audio file
            
        Returns:
            Audio bytes
        """
        if not EDGE_TTS_AVAILABLE:
            return self._fallback_synthesize(text, language)
        
        # Run async function synchronously
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        audio_data = loop.run_until_complete(
            self.synthesize_async(text, language, dialect, rate, pitch)
        )
        
        if output_path:
            with open(output_path, "wb") as f:
                f.write(audio_data)
        
        return audio_data
    
    def synthesize_to_base64(
        self,
        text: str,
        language: str = "italian",
        dialect: str = "standard",
        rate: str = "+0%"
    ) -> str:
        """
        Synthesize speech and return as base64 string.
        Useful for web API responses.
        """
        audio_data = self.synthesize(text, language, dialect, rate)
        return base64.b64encode(audio_data).decode("utf-8")
    
    def _fallback_synthesize(self, text: str, language: str) -> bytes:
        """Fallback when edge-tts is not available"""
        print(f"⚠️ TTS fallback: Would speak '{text}' in {language}")
        return b""
    
    async def stream_synthesize(
        self,
        text: str,
        language: str = "italian",
        dialect: str = "standard",
        rate: str = "+0%"
    ):
        """
        Stream audio chunks for real-time playback.
        
        Yields:
            Audio chunk bytes
        """
        if not EDGE_TTS_AVAILABLE:
            yield b""
            return
        
        voice_config = self.get_voice(language, dialect)
        
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice_config.voice_id,
            rate=rate
        )
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]


class SpeechService:
    """
    Combined speech service for Language Mirror Pro.
    Handles both STT and TTS.
    """
    
    def __init__(self):
        self.tts = TextToSpeech()
        
        # Import STT if available
        try:
            from speech.stt import SpeechToText, PronunciationAnalyzer
            self.stt = SpeechToText(model_size="base")
            self.pronunciation = PronunciationAnalyzer(self.stt)
            self.stt_available = True
        except ImportError:
            self.stt = None
            self.pronunciation = None
            self.stt_available = False
    
    def speak(
        self,
        text: str,
        language: str = "italian",
        dialect: str = "standard"
    ) -> bytes:
        """Generate speech from text"""
        return self.tts.synthesize(text, language, dialect)
    
    def speak_base64(
        self,
        text: str,
        language: str = "italian",
        dialect: str = "standard"
    ) -> str:
        """Generate speech and return as base64"""
        return self.tts.synthesize_to_base64(text, language, dialect)
    
    def transcribe(
        self,
        audio_path: str = None,
        audio_data: bytes = None,
        language: str = None
    ) -> Dict:
        """Transcribe speech to text"""
        if not self.stt_available:
            return {"success": False, "error": "STT not available"}
        return self.stt.transcribe(audio_path, audio_data, language)
    
    def analyze_pronunciation(
        self,
        audio_path: str = None,
        audio_data: bytes = None,
        expected_text: str = "",
        language: str = "italian"
    ) -> Dict:
        """Analyze pronunciation quality"""
        if not self.pronunciation:
            return {"success": False, "error": "Pronunciation analyzer not available"}
        return self.pronunciation.analyze(audio_path, audio_data, expected_text, language)


# Quick test
if __name__ == "__main__":
    print("🔊 Testing Text-to-Speech...")
    print("=" * 60)
    
    tts = TextToSpeech()
    
    if EDGE_TTS_AVAILABLE:
        print("✅ Edge TTS available!")
        print("\n📋 Available voices:")
        
        for lang in ["italian", "japanese", "spanish"]:
            voices = tts.get_available_voices(lang)
            print(f"\n  {lang.capitalize()}:")
            for v in voices:
                print(f"    - {v['name']} ({v['dialect']}, {v['gender']})")
        
        # Test synthesis
        print("\n🎵 Testing synthesis...")
        try:
            audio = tts.synthesize(
                "Ciao! Benvenuto a Language Mirror!",
                language="italian",
                dialect="standard"
            )
            print(f"✅ Generated {len(audio)} bytes of audio")
            
            # Save test file
            test_path = "/tmp/test_tts.mp3"
            with open(test_path, "wb") as f:
                f.write(audio)
            print(f"✅ Saved to {test_path}")
            
        except Exception as e:
            print(f"❌ Synthesis failed: {e}")
    else:
        print("⚠️ Edge TTS not available - install with: pip install edge-tts")
    
    print("\n" + "=" * 60)
    print("🎉 Text-to-Speech module ready!")
