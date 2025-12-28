"""
🎤🔊 Language Mirror Pro - Speech Module
========================================
Speech-to-Text and Text-to-Speech capabilities.
"""

from .stt import SpeechToText, PronunciationAnalyzer, WHISPER_AVAILABLE
from .tts import TextToSpeech, SpeechService, EDGE_TTS_AVAILABLE, VOICE_MAP

__all__ = [
    "SpeechToText",
    "PronunciationAnalyzer", 
    "TextToSpeech",
    "SpeechService",
    "WHISPER_AVAILABLE",
    "EDGE_TTS_AVAILABLE",
    "VOICE_MAP"
]
