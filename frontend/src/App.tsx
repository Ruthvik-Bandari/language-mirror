import { useState, useRef, useEffect } from 'react';
import { QueryClientProvider, useMutation } from '@tanstack/react-query';
import { 
  Languages, ChevronRight, Sparkles, Volume2, GraduationCap,
  Send, Mic, Globe, VolumeX, RefreshCw, Settings, MessageCircle,
  Coffee, Map, ShoppingBag, Hotel, Lightbulb, ArrowLeft
} from 'lucide-react';
import { queryClient, api, Message, ConversationResponse } from './api';

const LANGUAGES = [
  { code: 'italian', name: 'Italian', flag: '🇮🇹', dialects: ['Standard', 'Sicilian', 'Roman', 'Milanese'] },
  { code: 'japanese', name: 'Japanese', flag: '🇯🇵', dialects: ['Standard', 'Osaka', 'Kyoto'] },
  { code: 'spanish', name: 'Spanish', flag: '🇪🇸', dialects: ['Castilian', 'Mexican', 'Argentinian'] },
  { code: 'french', name: 'French', flag: '🇫🇷', dialects: ['Parisian', 'Quebec', 'Belgian'] },
  { code: 'german', name: 'German', flag: '🇩🇪', dialects: ['Standard', 'Bavarian', 'Austrian'] },
];

const SCENARIOS = [
  { id: 'free', name: 'Free Chat', icon: MessageCircle, description: 'Practice anything' },
  { id: 'restaurant', name: 'Restaurant', icon: Coffee, description: 'Order food & drinks' },
  { id: 'directions', name: 'Directions', icon: Map, description: 'Navigate the city' },
  { id: 'shopping', name: 'Shopping', icon: ShoppingBag, description: 'Buy things' },
  { id: 'hotel', name: 'Hotel', icon: Hotel, description: 'Book & check-in' },
];

// Welcome Screen
function WelcomeScreen({ onStart }: { onStart: () => void }) {
  return (
    <div className="screen gradient-bg items-center justify-center" style={{ padding: '1.5rem' }}>
      <div className="container text-center">
        {/* Logo */}
        <div 
          className="icon-box gradient-cyan-purple animate-scale-in" 
          style={{ margin: '0 auto 2rem', animationDelay: '0.2s' }}
        >
          <Languages size={48} color="white" />
        </div>

        {/* Title */}
        <h1 
          className="text-5xl font-bold mb-4 animate-fade-in" 
          style={{ animationDelay: '0.3s' }}
        >
          Language <span className="gradient-text">Mirror</span>
        </h1>

        <p 
          className="text-xl mb-2 animate-fade-in" 
          style={{ color: '#cbd5e1', animationDelay: '0.4s' }}
        >
          AI Language Tutor with Native Accents
        </p>

        <p 
          className="text-sm mb-10 animate-fade-in" 
          style={{ color: '#94a3b8', animationDelay: '0.5s' }}
        >
          Powered by Custom Reinforcement Learning
        </p>

        {/* Features */}
        <div 
          className="grid grid-cols-3 gap-4 mb-10 animate-fade-in" 
          style={{ animationDelay: '0.6s' }}
        >
          {[
            { icon: Sparkles, label: 'Custom AI', gradient: 'gradient-yellow-orange' },
            { icon: Volume2, label: 'Native Voice', gradient: 'gradient-cyan-blue' },
            { icon: GraduationCap, label: 'Adaptive', gradient: 'gradient-purple-pink' },
          ].map((feature, i) => (
            <div key={i} className="text-center">
              <div className={`feature-box ${feature.gradient}`} style={{ margin: '0 auto 0.5rem' }}>
                <feature.icon size={28} color="white" />
              </div>
              <p className="text-sm" style={{ color: '#cbd5e1' }}>{feature.label}</p>
            </div>
          ))}
        </div>

        {/* Start Button */}
        <button 
          className="btn btn-primary w-full animate-fade-in" 
          onClick={onStart}
          style={{ animationDelay: '0.7s' }}
        >
          Start Learning
          <ChevronRight size={20} />
        </button>

        {/* Badge */}
        <div className="badge mt-8 animate-fade-in" style={{ animationDelay: '0.9s' }}>
          <Sparkles size={16} color="#facc15" />
          Custom RL Model • No External APIs
        </div>
      </div>
    </div>
  );
}

// Setup Screen
function SetupScreen({ 
  onBack, 
  onStart,
  language,
  setLanguage,
  dialect,
  setDialect,
  scenario,
  setScenario
}: { 
  onBack: () => void;
  onStart: () => void;
  language: typeof LANGUAGES[0];
  setLanguage: (l: typeof LANGUAGES[0]) => void;
  dialect: string;
  setDialect: (d: string) => void;
  scenario: string;
  setScenario: (s: string) => void;
}) {
  return (
    <div className="screen gradient-bg overflow-auto" style={{ padding: '1.5rem' }}>
      <div style={{ maxWidth: '32rem', margin: '0 auto', width: '100%' }}>
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <button 
            onClick={onBack} 
            className="flex items-center gap-2"
            style={{ color: '#94a3b8', background: 'none', border: 'none', cursor: 'pointer' }}
          >
            <ArrowLeft size={20} /> Back
          </button>
          <h1 className="text-xl font-semibold">Setup</h1>
          <div style={{ width: '4rem' }} />
        </div>

        {/* Language Selection */}
        <div className="mb-8 animate-fade-in">
          <h2 className="text-lg font-medium mb-4">Choose Language</h2>
          <div className="grid grid-cols-2 gap-3">
            {LANGUAGES.map((lang) => (
              <button
                key={lang.code}
                onClick={() => { setLanguage(lang); setDialect(lang.dialects[0]); }}
                className={`card ${language.code === lang.code ? 'selected' : ''}`}
                style={{ textAlign: 'center' }}
              >
                <span style={{ fontSize: '2rem', display: 'block', marginBottom: '0.5rem' }}>{lang.flag}</span>
                <span className="font-medium">{lang.name}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Dialect Selection */}
        <div className="mb-8 animate-fade-in" style={{ animationDelay: '0.1s' }}>
          <h2 className="text-lg font-medium mb-4">Choose Dialect</h2>
          <div className="flex flex-wrap gap-2">
            {language.dialects.map((d) => (
              <button
                key={d}
                onClick={() => setDialect(d)}
                className={`chip ${dialect === d ? 'selected' : ''}`}
              >
                {d}
              </button>
            ))}
          </div>
        </div>

        {/* Scenario Selection */}
        <div className="mb-8 animate-fade-in" style={{ animationDelay: '0.2s' }}>
          <h2 className="text-lg font-medium mb-4">Choose Scenario</h2>
          <div className="space-y-3">
            {SCENARIOS.map((s) => (
              <button
                key={s.id}
                onClick={() => setScenario(s.id)}
                className={`card ${scenario === s.id ? 'selected' : ''}`}
                style={{ display: 'flex', alignItems: 'center', gap: '1rem', width: '100%' }}
              >
                <div className={`scenario-icon ${scenario === s.id ? 'selected' : ''}`}>
                  <s.icon size={24} color="white" />
                </div>
                <div className="text-left">
                  <p className="font-medium">{s.name}</p>
                  <p className="text-sm" style={{ color: '#94a3b8' }}>{s.description}</p>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Start Button */}
        <button 
          className="btn btn-primary w-full animate-fade-in mb-8" 
          onClick={onStart}
          style={{ animationDelay: '0.3s' }}
        >
          Start Conversation
        </button>
      </div>
    </div>
  );
}

// Chat Screen
function ChatScreen({
  language,
  dialect,
  scenario,
  onSettings,
}: {
  language: typeof LANGUAGES[0];
  dialect: string;
  scenario: string;
  onSettings: () => void;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [showTranslation, setShowTranslation] = useState(true);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const mutation = useMutation({
    mutationFn: api.sendMessage,
    onSuccess: (data: ConversationResponse) => {
      const tutorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'tutor',
        text: data.tutor_response,
        translation: data.translation,
        grammarFeedback: data.grammar_feedback,
        pronunciationScore: data.pronunciation_score,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, tutorMessage]);
      setSessionId(data.session_id);
    },
    onError: () => {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'tutor',
        text: 'Mi scusi, ho avuto un problema. Può ripetere?',
        translation: 'Sorry, I had a problem. Can you repeat?',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    },
  });

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    if (!inputText.trim() || mutation.isPending) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      text: inputText,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);

    mutation.mutate({
      text: inputText,
      language: language.code,
      dialect,
      scenario,
      sessionId: sessionId || undefined,
    });

    setInputText('');
  };

  const handleReset = () => {
    setMessages([]);
    setSessionId(null);
  };

  return (
    <div className="screen gradient-bg-dark">
      {/* Header */}
      <div className="header">
        <button className="btn-icon" onClick={onSettings}>
          <Settings size={24} />
        </button>
        <div className="text-center">
          <h1 className="font-semibold flex items-center justify-center gap-2">
            <span style={{ fontSize: '1.25rem' }}>{language.flag}</span>
            {language.name}
            <span className="text-sm" style={{ color: '#94a3b8' }}>({dialect})</span>
          </h1>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowTranslation(!showTranslation)}
            className={`btn-icon ${showTranslation ? 'active' : ''}`}
          >
            <Globe size={20} />
          </button>
          <button
            onClick={() => setAudioEnabled(!audioEnabled)}
            className={`btn-icon ${audioEnabled ? 'active' : ''}`}
          >
            {audioEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
          </button>
          <button onClick={handleReset} className="btn-icon">
            <RefreshCw size={20} />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">
              <MessageCircle size={40} color="#475569" />
            </div>
            <p style={{ color: '#94a3b8' }}>Start a conversation!</p>
            <p className="text-sm mt-2" style={{ color: '#64748b' }}>
              Try saying hello in {language.name}
            </p>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            style={{ animation: 'fadeIn 0.3s ease-out' }}
          >
            <div className={`message ${message.type === 'user' ? 'message-user' : 'message-tutor'}`}>
              <p>{message.text}</p>
              
              {message.type === 'tutor' && message.translation && showTranslation && (
                <p className="text-sm mt-2 pt-2 border-t" style={{ color: '#94a3b8' }}>
                  {message.translation}
                </p>
              )}
              
              {message.grammarFeedback && (
                <div className="grammar-tip">
                  <div className="flex items-center gap-2 text-sm font-medium mb-1" style={{ color: '#facc15' }}>
                    <Lightbulb size={16} />
                    Grammar Tip
                  </div>
                  <p className="text-sm" style={{ color: '#fef08a' }}>{message.grammarFeedback}</p>
                </div>
              )}
              
              {message.pronunciationScore !== undefined && message.pronunciationScore > 0 && (
                <div className="flex items-center gap-2 mt-2">
                  <span className="text-xs" style={{ color: '#94a3b8' }}>Pronunciation:</span>
                  <div className="pronunciation-bar flex-1">
                    <div
                      className={`pronunciation-fill ${
                        message.pronunciationScore >= 0.8 ? 'good' :
                        message.pronunciationScore >= 0.6 ? 'medium' : 'poor'
                      }`}
                      style={{ width: `${message.pronunciationScore * 100}%` }}
                    />
                  </div>
                  <span className="text-xs" style={{ color: '#94a3b8' }}>
                    {Math.round(message.pronunciationScore * 100)}%
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}

        {mutation.isPending && (
          <div className="flex justify-start">
            <div className="message message-tutor">
              <div className="loading-dots">
                <div className="loading-dot" />
                <div className="loading-dot" />
                <div className="loading-dot" />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="chat-input-area">
        <button className="btn-icon" style={{ padding: '1rem' }}>
          <Mic size={24} />
        </button>

        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder={`Type in ${language.name}...`}
          className="input"
        />

        <button
          onClick={handleSend}
          disabled={!inputText.trim() || mutation.isPending}
          className={inputText.trim() && !mutation.isPending ? 'btn-icon gradient-cyan-purple' : 'btn-icon'}
          style={{ padding: '1rem', color: inputText.trim() ? 'white' : undefined }}
        >
          <Send size={24} />
        </button>
      </div>
    </div>
  );
}

// Main App
function AppContent() {
  const [screen, setScreen] = useState<'welcome' | 'setup' | 'chat'>('welcome');
  const [language, setLanguage] = useState(LANGUAGES[0]);
  const [dialect, setDialect] = useState('Standard');
  const [scenario, setScenario] = useState('free');

  if (screen === 'welcome') {
    return <WelcomeScreen onStart={() => setScreen('setup')} />;
  }

  if (screen === 'setup') {
    return (
      <SetupScreen
        onBack={() => setScreen('welcome')}
        onStart={() => setScreen('chat')}
        language={language}
        setLanguage={setLanguage}
        dialect={dialect}
        setDialect={setDialect}
        scenario={scenario}
        setScenario={setScenario}
      />
    );
  }

  return (
    <ChatScreen
      language={language}
      dialect={dialect}
      scenario={scenario}
      onSettings={() => setScreen('setup')}
    />
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}

export default App;
