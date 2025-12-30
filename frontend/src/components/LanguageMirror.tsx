import { useState, useRef, useEffect } from 'react';

// Types
interface Message {
  id: string;
  type: 'user' | 'tutor';
  text: string;
  translation?: string;
  grammarNote?: string;
  pronunciationTip?: string;
  audioBase64?: string;
  timestamp: Date;
}

interface Language {
  code: string;
  name: string;
  native_name: string;
  flag: string;
  greeting: string;
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Voice-driven Language Tutor Component
export default function LanguageMirror() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('italian');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [languages, setLanguages] = useState<Language[]>([]);
  
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const recognitionRef = useRef<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch available languages
  useEffect(() => {
    fetch(`${API_URL}/api/languages`)
      .then(res => res.json())
      .then(data => setLanguages(data.languages))
      .catch(err => console.error('Failed to fetch languages:', err));
  }, []);

  // Initialize speech recognition
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      
      // Set language for speech recognition
      const langCodes: Record<string, string> = {
        'italian': 'it-IT',
        'japanese': 'ja-JP',
        'spanish': 'es-ES',
        'french': 'fr-FR',
        'german': 'de-DE'
      };
      recognitionRef.current.lang = langCodes[selectedLanguage] || 'it-IT';

      recognitionRef.current.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInputText(transcript);
        setIsListening(false);
        // Auto-send after voice input
        handleSendMessage(transcript);
      };

      recognitionRef.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }
  }, [selectedLanguage]);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Start lesson with greeting
  const startLesson = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/start-lesson?language=${selectedLanguage}`, {
        method: 'POST'
      });
      const data = await response.json();
      
      setSessionId(data.session_id);
      
      const tutorMessage: Message = {
        id: Date.now().toString(),
        type: 'tutor',
        text: data.greeting,
        audioBase64: data.audio_base64,
        timestamp: new Date()
      };
      
      setMessages([tutorMessage]);
      
      // Auto-play greeting
      if (data.audio_base64) {
        playAudio(data.audio_base64);
      }
    } catch (error) {
      console.error('Failed to start lesson:', error);
    }
    setIsLoading(false);
  };

  // Send message to tutor
  const handleSendMessage = async (text?: string) => {
    const messageText = text || inputText.trim();
    if (!messageText || isLoading) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      text: messageText,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/conversation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: messageText,
          language: selectedLanguage,
          session_id: sessionId,
          include_audio: true
        })
      });

      const data = await response.json();
      
      if (!sessionId) {
        setSessionId(data.session_id);
      }

      const tutorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'tutor',
        text: data.tutor_response,
        translation: data.translation,
        grammarNote: data.grammar_note,
        pronunciationTip: data.pronunciation_tip,
        audioBase64: data.audio_base64,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, tutorMessage]);
      
      // Auto-play response
      if (data.audio_base64) {
        playAudio(data.audio_base64);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
    }
    
    setIsLoading(false);
  };

  // Play audio from base64
  const playAudio = (base64Audio: string) => {
    setIsPlaying(true);
    const audio = new Audio(`data:audio/mpeg;base64,${base64Audio}`);
    audioRef.current = audio;
    
    audio.onended = () => setIsPlaying(false);
    audio.onerror = () => setIsPlaying(false);
    
    audio.play().catch(err => {
      console.error('Audio playback error:', err);
      setIsPlaying(false);
    });
  };

  // Stop audio
  const stopAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setIsPlaying(false);
  };

  // Start listening
  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      setIsListening(true);
      recognitionRef.current.start();
    }
  };

  // Stop listening
  const stopListening = () => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  };

  // Handle Enter key
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Get language flag
  const getFlag = (code: string) => {
    const lang = languages.find(l => l.code === code);
    return lang?.flag || '🌍';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-800">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            🪞 Language Mirror
          </h1>
          <p className="text-purple-200">
            Voice-driven AI Language Tutor
          </p>
          <p className="text-purple-300 text-sm mt-1">
            Powered by Google Gemini + ElevenLabs
          </p>
        </header>

        {/* Language Selector */}
        <div className="flex justify-center gap-2 mb-6">
          {languages.map(lang => (
            <button
              key={lang.code}
              onClick={() => {
                setSelectedLanguage(lang.code);
                setMessages([]);
                setSessionId(null);
              }}
              className={`px-4 py-2 rounded-full text-lg transition-all ${
                selectedLanguage === lang.code
                  ? 'bg-white text-purple-900 shadow-lg scale-110'
                  : 'bg-white/20 text-white hover:bg-white/30'
              }`}
            >
              {lang.flag} {lang.name}
            </button>
          ))}
        </div>

        {/* Start Lesson Button */}
        {messages.length === 0 && (
          <div className="text-center mb-8">
            <button
              onClick={startLesson}
              disabled={isLoading}
              className="px-8 py-4 bg-gradient-to-r from-green-500 to-emerald-500 text-white text-xl font-bold rounded-full shadow-lg hover:shadow-xl transition-all transform hover:scale-105 disabled:opacity-50"
            >
              {isLoading ? '⏳ Starting...' : `🎤 Start ${getFlag(selectedLanguage)} Lesson`}
            </button>
          </div>
        )}

        {/* Chat Messages */}
        {messages.length > 0 && (
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4 mb-4 h-96 overflow-y-auto">
            {messages.map(message => (
              <div
                key={message.id}
                className={`mb-4 ${
                  message.type === 'user' ? 'text-right' : 'text-left'
                }`}
              >
                <div
                  className={`inline-block max-w-[80%] p-4 rounded-2xl ${
                    message.type === 'user'
                      ? 'bg-purple-600 text-white'
                      : 'bg-white text-gray-800'
                  }`}
                >
                  <p className="text-lg">{message.text}</p>
                  
                  {message.translation && (
                    <p className="text-sm mt-2 opacity-70 border-t pt-2">
                      📝 {message.translation}
                    </p>
                  )}
                  
                  {message.grammarNote && (
                    <p className="text-sm mt-2 text-orange-600 bg-orange-50 p-2 rounded">
                      ✏️ {message.grammarNote}
                    </p>
                  )}
                  
                  {message.pronunciationTip && (
                    <p className="text-sm mt-2 text-blue-600 bg-blue-50 p-2 rounded">
                      🗣️ {message.pronunciationTip}
                    </p>
                  )}
                  
                  {/* Play audio button */}
                  {message.audioBase64 && message.type === 'tutor' && (
                    <button
                      onClick={() => 
                        isPlaying ? stopAudio() : playAudio(message.audioBase64!)
                      }
                      className="mt-2 px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm hover:bg-purple-200 transition"
                    >
                      {isPlaying ? '⏹️ Stop' : '🔊 Play'}
                    </button>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}

        {/* Input Area */}
        {messages.length > 0 && (
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-4">
            <div className="flex gap-3">
              {/* Voice Input Button */}
              <button
                onClick={isListening ? stopListening : startListening}
                disabled={isLoading}
                className={`p-4 rounded-full transition-all ${
                  isListening
                    ? 'bg-red-500 text-white animate-pulse'
                    : 'bg-white text-purple-700 hover:bg-purple-100'
                }`}
              >
                {isListening ? '⏹️' : '🎤'}
              </button>

              {/* Text Input */}
              <input
                type="text"
                value={inputText}
                onChange={e => setInputText(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={isListening ? "Listening..." : "Type or speak..."}
                className="flex-1 px-4 py-3 rounded-xl bg-white text-gray-800 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-400"
                disabled={isLoading || isListening}
              />

              {/* Send Button */}
              <button
                onClick={() => handleSendMessage()}
                disabled={!inputText.trim() || isLoading}
                className="p-4 bg-purple-600 text-white rounded-full hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {isLoading ? '⏳' : '📤'}
              </button>
            </div>

            {/* Status */}
            <div className="text-center mt-3 text-purple-200 text-sm">
              {isListening && '🎙️ Listening... Speak now!'}
              {isLoading && '🧠 Thinking...'}
              {isPlaying && '🔊 Playing audio...'}
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="text-center mt-8 text-purple-300 text-sm">
          <p>Built with Google Cloud Gemini + ElevenLabs</p>
          <p>AI Accelerate Hackathon 2025</p>
        </footer>
      </div>
    </div>
  );
}
