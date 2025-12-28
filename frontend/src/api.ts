import { QueryClient } from '@tanstack/react-query';

export const API_URL = 'http://localhost:8000';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5,
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});

export interface Message {
  id: string;
  type: 'user' | 'tutor';
  text: string;
  translation?: string;
  grammarFeedback?: string;
  pronunciationScore?: number;
  timestamp: Date;
}

export interface ConversationRequest {
  text: string;
  language: string;
  dialect?: string;
  scenario?: string;
  sessionId?: string;
}

export interface ConversationResponse {
  tutor_response: string;
  translation: string;
  grammar_feedback?: string;
  pronunciation_score?: number;
  suggested_responses: string[];
  session_id: string;
}

export const api = {
  async healthCheck() {
    const response = await fetch(`${API_URL}/`);
    if (!response.ok) throw new Error('API not available');
    return response.json();
  },

  async sendMessage(request: ConversationRequest): Promise<ConversationResponse> {
    const response = await fetch(`${API_URL}/api/conversation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: request.text,
        language: request.language,
        dialect: request.dialect || 'standard',
        scenario: request.scenario || 'free',
        session_id: request.sessionId,
      }),
    });
    if (!response.ok) throw new Error('Failed to send message');
    return response.json();
  },
};
