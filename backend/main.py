"""
🚀 Language Mirror Pro - Voice-Driven Language Tutor
=====================================================
Complete backend using:
- Google Gemini API for AI conversations
- ElevenLabs for native accent voice synthesis
- FastAPI for the backend

ElevenLabs Challenge: Voice-driven, conversational, intelligent
"""

import os
import uuid
import base64
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import google.generativeai as genai
import httpx

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDecqgkr6NJBKjbwujohuPzCPnEl4G4fd4")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "fcba80850c9e78662167a5914a0f66d4c3c05bb632829d9109a0bbc5db57ca19")

# ElevenLabs voices for each language (native speakers)
ELEVENLABS_VOICES = {
    "italian": {
        "voice_id": "21m00Tcm4TlvDq8ikWAM",  # Rachel - can do Italian
        "name": "Italian Tutor",
        "model_id": "eleven_multilingual_v2"
    },
    "japanese": {
        "voice_id": "21m00Tcm4TlvDq8ikWAM",
        "name": "Japanese Tutor", 
        "model_id": "eleven_multilingual_v2"
    },
    "spanish": {
        "voice_id": "21m00Tcm4TlvDq8ikWAM",
        "name": "Spanish Tutor",
        "model_id": "eleven_multilingual_v2"
    },
    "french": {
        "voice_id": "21m00Tcm4TlvDq8ikWAM",
        "name": "French Tutor",
        "model_id": "eleven_multilingual_v2"
    },
    "german": {
        "voice_id": "21m00Tcm4TlvDq8ikWAM",
        "name": "German Tutor",
        "model_id": "eleven_multilingual_v2"
    },
}

# Language configurations
LANGUAGE_CONFIG = {
    "italian": {
        "name": "Italian",
        "native_name": "Italiano",
        "greeting": "Ciao! Sono il tuo tutor di italiano.",
        "flag": "🇮🇹"
    },
    "japanese": {
        "name": "Japanese",
        "native_name": "日本語",
        "greeting": "こんにちは！私はあなたの日本語の先生です。",
        "flag": "🇯🇵"
    },
    "spanish": {
        "name": "Spanish",
        "native_name": "Español",
        "greeting": "¡Hola! Soy tu tutor de español.",
        "flag": "🇪🇸"
    },
    "french": {
        "name": "French",
        "native_name": "Français",
        "greeting": "Bonjour ! Je suis votre tuteur de français.",
        "flag": "🇫🇷"
    },
    "german": {
        "name": "German",
        "native_name": "Deutsch",
        "greeting": "Hallo! Ich bin Ihr Deutschlehrer.",
        "flag": "🇩🇪"
    },
}


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ConversationRequest(BaseModel):
    text: str = Field(..., description="User input text")
    language: str = Field(default="italian", description="Target language")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    include_audio: bool = Field(default=True, description="Include audio response")


class ConversationResponse(BaseModel):
    tutor_response: str
    translation: str
    grammar_note: Optional[str] = None
    pronunciation_tip: Optional[str] = None
    audio_base64: Optional[str] = None
    session_id: str
    language: str


class HealthResponse(BaseModel):
    status: str
    gemini_available: bool
    elevenlabs_available: bool
    supported_languages: List[str]
    version: str


class VoiceListResponse(BaseModel):
    voices: List[Dict]


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

class ConversationSession:
    def __init__(self, language: str, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.language = language
        self.history: List[Dict] = []
        self.created_at = datetime.now()
    
    def add_turn(self, user_text: str, tutor_response: str):
        self.history.append({
            "user": user_text,
            "tutor": tutor_response,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_context(self, max_turns: int = 5) -> str:
        """Get recent conversation context for Gemini"""
        recent = self.history[-max_turns:] if self.history else []
        context = ""
        for turn in recent:
            context += f"Student: {turn['user']}\nTutor: {turn['tutor']}\n"
        return context


sessions: Dict[str, ConversationSession] = {}


# ============================================================================
# GEMINI AI
# ============================================================================

class GeminiTutor:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.available = True
        print("✅ Gemini AI initialized")
    
    def create_prompt(self, user_text: str, language: str, context: str = "") -> str:
        lang_config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["italian"])
        
        prompt = f"""You are a friendly and encouraging {lang_config['name']} language tutor. 
Your name is Language Mirror and you help students learn {lang_config['name']}.

RULES:
1. ALWAYS respond in {lang_config['name']} first, then provide English translation
2. Keep responses conversational and natural (2-3 sentences max)
3. If the student makes a grammar mistake, gently correct it
4. Be encouraging and supportive
5. Adapt to the student's level
6. Ask follow-up questions to keep the conversation going

CONVERSATION CONTEXT:
{context}

STUDENT'S MESSAGE: {user_text}

Respond in this EXACT format:
RESPONSE: [Your response in {lang_config['name']}]
TRANSLATION: [English translation]
GRAMMAR_NOTE: [Any grammar correction or tip, or "None" if no issues]
PRONUNCIATION_TIP: [Any pronunciation tip, or "None"]
"""
        return prompt
    
    async def generate_response(self, user_text: str, language: str, context: str = "") -> Dict:
        """Generate response using Gemini"""
        try:
            prompt = self.create_prompt(user_text, language, context)
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            return self.parse_response(response.text, language)
            
        except Exception as e:
            print(f"Gemini error: {e}")
            return self.fallback_response(language)
    
    def parse_response(self, text: str, language: str) -> Dict:
        """Parse Gemini's response"""
        result = {
            "tutor_response": "",
            "translation": "",
            "grammar_note": None,
            "pronunciation_tip": None
        }
        
        lines = text.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            if line.startswith("RESPONSE:"):
                result["tutor_response"] = line.replace("RESPONSE:", "").strip()
            elif line.startswith("TRANSLATION:"):
                result["translation"] = line.replace("TRANSLATION:", "").strip()
            elif line.startswith("GRAMMAR_NOTE:"):
                note = line.replace("GRAMMAR_NOTE:", "").strip()
                if note.lower() != "none":
                    result["grammar_note"] = note
            elif line.startswith("PRONUNCIATION_TIP:"):
                tip = line.replace("PRONUNCIATION_TIP:", "").strip()
                if tip.lower() != "none":
                    result["pronunciation_tip"] = tip
        
        # Fallback if parsing failed
        if not result["tutor_response"]:
            result["tutor_response"] = text.split("\n")[0]
            result["translation"] = "Translation not available"
        
        return result
    
    def fallback_response(self, language: str) -> Dict:
        """Fallback responses if Gemini fails"""
        fallbacks = {
            "italian": {
                "tutor_response": "Scusa, non ho capito. Puoi ripetere?",
                "translation": "Sorry, I didn't understand. Can you repeat?"
            },
            "japanese": {
                "tutor_response": "すみません、わかりませんでした。もう一度言ってください。",
                "translation": "Sorry, I didn't understand. Please say it again."
            },
            "spanish": {
                "tutor_response": "Perdona, no entendí. ¿Puedes repetir?",
                "translation": "Sorry, I didn't understand. Can you repeat?"
            },
            "french": {
                "tutor_response": "Pardon, je n'ai pas compris. Pouvez-vous répéter?",
                "translation": "Sorry, I didn't understand. Can you repeat?"
            },
            "german": {
                "tutor_response": "Entschuldigung, ich habe das nicht verstanden. Können Sie wiederholen?",
                "translation": "Sorry, I didn't understand. Can you repeat?"
            },
        }
        
        fb = fallbacks.get(language, fallbacks["italian"])
        return {
            "tutor_response": fb["tutor_response"],
            "translation": fb["translation"],
            "grammar_note": None,
            "pronunciation_tip": None
        }


# ============================================================================
# ELEVENLABS TTS
# ============================================================================

class ElevenLabsTTS:
    def __init__(self):
        self.api_key = ELEVENLABS_API_KEY
        self.base_url = "https://api.elevenlabs.io/v1"
        self.available = True
        print("✅ ElevenLabs TTS initialized")
    
    async def synthesize(self, text: str, language: str) -> Optional[bytes]:
        """Convert text to speech using ElevenLabs"""
        try:
            voice_config = ELEVENLABS_VOICES.get(language, ELEVENLABS_VOICES["italian"])
            
            url = f"{self.base_url}/text-to-speech/{voice_config['voice_id']}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "model_id": voice_config["model_id"],
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.5,
                    "use_speaker_boost": True
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers, timeout=30.0)
                
                if response.status_code == 200:
                    return response.content
                else:
                    print(f"ElevenLabs error: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            print(f"ElevenLabs error: {e}")
            return None
    
    async def synthesize_to_base64(self, text: str, language: str) -> Optional[str]:
        """Convert text to speech and return as base64"""
        audio_bytes = await self.synthesize(text, language)
        if audio_bytes:
            return base64.b64encode(audio_bytes).decode('utf-8')
        return None
    
    async def get_voices(self) -> List[Dict]:
        """Get available voices from ElevenLabs"""
        try:
            url = f"{self.base_url}/voices"
            headers = {"xi-api-key": self.api_key}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("voices", [])
                return []
        except Exception as e:
            print(f"Error fetching voices: {e}")
            return []


# ============================================================================
# FASTAPI APP
# ============================================================================

# Global instances
gemini_tutor: GeminiTutor = None
elevenlabs_tts: ElevenLabsTTS = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown"""
    global gemini_tutor, elevenlabs_tts
    
    print("=" * 60)
    print("🚀 Language Mirror Pro - Voice-Driven Language Tutor")
    print("=" * 60)
    
    # Initialize services
    gemini_tutor = GeminiTutor()
    elevenlabs_tts = ElevenLabsTTS()
    
    print("=" * 60)
    print("✅ All services ready!")
    print("=" * 60)
    
    yield
    
    print("👋 Shutting down...")


app = FastAPI(
    title="Language Mirror Pro API",
    description="Voice-driven language tutor using Gemini + ElevenLabs",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        gemini_available=gemini_tutor.available if gemini_tutor else False,
        elevenlabs_available=elevenlabs_tts.available if elevenlabs_tts else False,
        supported_languages=list(LANGUAGE_CONFIG.keys()),
        version="3.0.0"
    )


@app.get("/api/languages")
async def get_languages():
    """Get supported languages"""
    languages = []
    for code, config in LANGUAGE_CONFIG.items():
        languages.append({
            "code": code,
            "name": config["name"],
            "native_name": config["native_name"],
            "flag": config["flag"],
            "greeting": config["greeting"]
        })
    return {"languages": languages}


@app.post("/api/conversation", response_model=ConversationResponse)
async def conversation(request: ConversationRequest):
    """Main conversation endpoint - voice-driven interaction"""
    global gemini_tutor, elevenlabs_tts
    
    # Get or create session
    session_id = request.session_id
    if session_id and session_id in sessions:
        session = sessions[session_id]
    else:
        session = ConversationSession(request.language)
        session_id = session.session_id
        sessions[session_id] = session
    
    # Get conversation context
    context = session.get_context()
    
    # Generate response using Gemini
    result = await gemini_tutor.generate_response(
        user_text=request.text,
        language=request.language,
        context=context
    )
    
    # Update session
    session.add_turn(request.text, result["tutor_response"])
    
    # Generate audio using ElevenLabs
    audio_base64 = None
    if request.include_audio and elevenlabs_tts:
        audio_base64 = await elevenlabs_tts.synthesize_to_base64(
            result["tutor_response"],
            request.language
        )
    
    return ConversationResponse(
        tutor_response=result["tutor_response"],
        translation=result["translation"],
        grammar_note=result["grammar_note"],
        pronunciation_tip=result["pronunciation_tip"],
        audio_base64=audio_base64,
        session_id=session_id,
        language=request.language
    )


@app.get("/api/voices", response_model=VoiceListResponse)
async def get_voices():
    """Get available ElevenLabs voices"""
    voices = await elevenlabs_tts.get_voices()
    return {"voices": voices}


@app.post("/api/tts")
async def text_to_speech(text: str, language: str = "italian"):
    """Generate speech from text"""
    audio_base64 = await elevenlabs_tts.synthesize_to_base64(text, language)
    
    if audio_base64:
        return {
            "success": True,
            "audio_base64": audio_base64,
            "format": "mp3"
        }
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to generate audio"}
        )


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session history"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "session_id": session.session_id,
        "language": session.language,
        "history": session.history,
        "created_at": session.created_at.isoformat()
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "deleted"}


@app.post("/api/start-lesson")
async def start_lesson(language: str = "italian"):
    """Start a new lesson with a greeting"""
    session = ConversationSession(language)
    sessions[session.session_id] = session
    
    lang_config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["italian"])
    greeting = lang_config["greeting"]
    
    # Generate audio for greeting
    audio_base64 = await elevenlabs_tts.synthesize_to_base64(greeting, language)
    
    return {
        "session_id": session.session_id,
        "language": language,
        "greeting": greeting,
        "audio_base64": audio_base64
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
