"""
🚀 Language Mirror Pro - Production Backend with Trained Model
==============================================================
FastAPI backend using the custom trained 76.9M parameter model.
"""

import os
import sys
import uuid
import random
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Import the trained model inference
from model_inference import LanguageMirrorInference, SimpleTokenizer

# Import TTS
try:
    from speech.tts import TextToSpeech, EDGE_TTS_AVAILABLE
    tts = TextToSpeech()
    TTS_READY = EDGE_TTS_AVAILABLE
except ImportError:
    tts = None
    TTS_READY = False
    print("⚠️ TTS not available. Install: pip install edge-tts")


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ConversationRequest(BaseModel):
    text: str = Field(..., description="User input text")
    language: str = Field(default="italian", description="Target language")
    dialect: Optional[str] = Field(default=None, description="Regional dialect")
    scenario: Optional[str] = Field(default="general", description="Conversation scenario")
    session_id: Optional[str] = Field(default=None, description="Session ID for context")


class ConversationResponse(BaseModel):
    tutor_response: str
    translation: str
    grammar_feedback: Optional[str] = None
    pronunciation_score: Optional[float] = None
    suggested_responses: List[str] = []
    session_id: str
    audio: Optional[str] = None
    model_source: str = "trained_model"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_params: str
    tts_available: bool
    version: str
    timestamp: str


# ============================================================================
# LANGUAGE CONFIG
# ============================================================================

LANGUAGE_CONFIG = {
    "italian": {
        "name": "Italian",
        "dialects": ["standard", "sicilian", "roman", "milanese"],
        "suggestions": ["Grazie!", "Come si dice...?", "Non capisco", "Perfetto!"],
    },
    "japanese": {
        "name": "Japanese", 
        "dialects": ["standard", "osaka", "kyoto"],
        "suggestions": ["ありがとう", "もう一度", "わかりません", "すごい!"],
    },
    "spanish": {
        "name": "Spanish",
        "dialects": ["castilian", "mexican", "argentinian"],
        "suggestions": ["¡Gracias!", "¿Cómo se dice...?", "No entiendo", "¡Perfecto!"],
    },
    "french": {
        "name": "French",
        "dialects": ["parisian", "quebec", "belgian"],
        "suggestions": ["Merci!", "Comment dit-on...?", "Je ne comprends pas", "Parfait!"],
    },
    "german": {
        "name": "German",
        "dialects": ["standard", "bavarian", "austrian"],
        "suggestions": ["Danke!", "Wie sagt man...?", "Ich verstehe nicht", "Perfekt!"],
    },
}


# ============================================================================
# SESSION MANAGEMENT  
# ============================================================================

class ConversationSession:
    def __init__(self, language: str, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.language = language
        self.history: List[Dict] = []
        self.turn_count = 0
        self.created_at = datetime.now()
    
    def add_turn(self, user_text: str, tutor_response: str):
        self.history.append({
            "turn": self.turn_count,
            "user": user_text,
            "tutor": tutor_response,
            "timestamp": datetime.now().isoformat()
        })
        self.turn_count += 1


sessions: Dict[str, ConversationSession] = {}


# ============================================================================
# FASTAPI APP
# ============================================================================

# Global model inference instance
model_inference: LanguageMirrorInference = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global model_inference
    
    print("🚀 Starting Language Mirror Pro API...")
    print("=" * 50)
    
    # Load trained model
    model_path = os.getenv("MODEL_PATH", "trained_model/best_model.pt")
    
    # Check multiple possible paths
    possible_paths = [
        model_path,
        "../trained_model/best_model.pt",
        "models/best_model.pt",
        "../models/best_model.pt",
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            model_path = path
            break
    
    model_inference = LanguageMirrorInference(model_path)
    
    print(f"🔊 TTS: {'✅ Ready' if TTS_READY else '❌ Not available'}")
    print("=" * 50)
    
    yield
    
    print("👋 Shutting down...")


app = FastAPI(
    title="Language Mirror Pro API",
    description="Custom 76.9M parameter language tutor model",
    version="2.0.0",
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
# ROUTES
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_inference.loaded if model_inference else False,
        model_params="76.9M",
        tts_available=TTS_READY,
        version="2.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/languages")
async def get_languages():
    """Get supported languages"""
    languages = []
    for code, data in LANGUAGE_CONFIG.items():
        languages.append({
            "code": code,
            "name": data["name"],
            "dialects": data["dialects"],
            "available": True
        })
    return {"languages": languages}


@app.post("/api/conversation", response_model=ConversationResponse)
async def conversation(request: ConversationRequest):
    """Main conversation endpoint using trained model"""
    global model_inference
    
    # Get or create session
    session_id = request.session_id
    if session_id and session_id in sessions:
        session = sessions[session_id]
    else:
        session = ConversationSession(request.language)
        session_id = session.session_id
        sessions[session_id] = session
    
    # Generate response using trained model
    result = model_inference.generate_response(
        user_input=request.text,
        language=request.language,
        temperature=0.8,
        use_model=True
    )
    
    tutor_response = result["response"]
    translation = result["translation"]
    source = result["source"]
    
    # Update session
    session.add_turn(request.text, tutor_response)
    
    # Generate audio if TTS available
    audio_base64 = None
    if TTS_READY and tts:
        try:
            audio_base64 = tts.synthesize_to_base64(
                tutor_response,
                request.language,
                request.dialect or "standard"
            )
        except Exception as e:
            print(f"TTS error: {e}")
    
    # Get suggestions
    lang_config = LANGUAGE_CONFIG.get(request.language, LANGUAGE_CONFIG["italian"])
    suggestions = lang_config.get("suggestions", [])[:4]
    
    # Pronunciation score (simulated for now)
    pronunciation_score = round(random.uniform(0.65, 0.95), 2)
    
    return ConversationResponse(
        tutor_response=tutor_response,
        translation=translation,
        pronunciation_score=pronunciation_score,
        suggested_responses=suggestions,
        session_id=session_id,
        audio=audio_base64,
        model_source=source
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
        "turn_count": session.turn_count,
        "history": session.history
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete session"""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "deleted"}


@app.post("/api/tts")
async def text_to_speech(text: str, language: str = "italian", dialect: str = "standard"):
    """Generate speech from text"""
    if not TTS_READY:
        return JSONResponse(status_code=501, content={"error": "TTS not available"})
    
    try:
        audio_base64 = tts.synthesize_to_base64(text, language, dialect)
        return {"success": True, "audio": audio_base64, "format": "mp3"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
