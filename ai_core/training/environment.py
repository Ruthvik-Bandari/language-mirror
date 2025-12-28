"""
🎓 Language Mirror Pro - Advanced Training Environment
=======================================================
Simulated language learning environment with:
- Curriculum learning
- Diverse learner profiles
- Realistic error patterns
- Pedagogically-informed rewards
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import json


class ProficiencyLevel(Enum):
    """CEFR-aligned proficiency levels"""
    A1 = 0  # Beginner
    A2 = 1  # Elementary
    B1 = 2  # Intermediate
    B2 = 3  # Upper Intermediate
    C1 = 4  # Advanced
    C2 = 5  # Mastery


class ResponseType(Enum):
    """Types of tutor responses"""
    GREETING = 0
    CORRECTION_GENTLE = 1
    CORRECTION_DIRECT = 2
    ENCOURAGEMENT = 3
    QUESTION_SIMPLE = 4
    QUESTION_COMPLEX = 5
    VOCABULARY_INTRO = 6
    GRAMMAR_EXPLANATION = 7
    PRONUNCIATION_TIP = 8
    CULTURAL_NOTE = 9
    PRACTICE_PROMPT = 10
    CONVERSATION = 11


@dataclass
class LearnerState:
    """Complete learner state for RL"""
    # Proficiency metrics (0-1)
    vocabulary: float = 0.2
    grammar: float = 0.2
    pronunciation: float = 0.2
    listening: float = 0.2
    fluency: float = 0.1
    
    # Psychological state
    confidence: float = 0.5
    motivation: float = 0.7
    frustration: float = 0.0
    engagement: float = 0.8
    
    # Session state
    errors_this_session: int = 0
    correct_this_session: int = 0
    turns_this_session: int = 0
    
    # Current level
    level: ProficiencyLevel = ProficiencyLevel.A1
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for model input"""
        return torch.tensor([
            self.vocabulary, self.grammar, self.pronunciation,
            self.listening, self.fluency, self.confidence,
            self.motivation, self.frustration, self.engagement,
            self.level.value / 5.0  # Normalized level
        ], dtype=torch.float32)
    
    def to_proficiency_vector(self) -> torch.Tensor:
        """5-dim proficiency vector for model"""
        return torch.tensor([
            self.vocabulary, self.grammar, self.pronunciation,
            self.confidence, self.errors_this_session / max(self.turns_this_session, 1)
        ], dtype=torch.float32)


@dataclass
class LanguageKnowledge:
    """Knowledge base for a language"""
    name: str
    code: str
    
    # Vocabulary by level
    vocabulary: Dict[ProficiencyLevel, List[str]] = field(default_factory=dict)
    
    # Grammar patterns by level
    grammar_patterns: Dict[ProficiencyLevel, List[str]] = field(default_factory=dict)
    
    # Common errors
    common_errors: List[Dict[str, str]] = field(default_factory=list)
    
    # Phrases
    greetings: List[str] = field(default_factory=list)
    responses: Dict[ResponseType, List[Tuple[str, str]]] = field(default_factory=dict)


# Pre-defined language knowledge bases
LANGUAGES = {
    "italian": LanguageKnowledge(
        name="Italian",
        code="it",
        vocabulary={
            ProficiencyLevel.A1: ["ciao", "grazie", "prego", "sì", "no", "buongiorno", "arrivederci"],
            ProficiencyLevel.A2: ["vorrei", "quanto costa", "dov'è", "mi chiamo", "piacere"],
            ProficiencyLevel.B1: ["potrebbe", "sarebbe", "avrei bisogno", "mi piacerebbe"],
            ProficiencyLevel.B2: ["nonostante", "sebbene", "affinché", "purché"],
        },
        common_errors=[
            {"wrong": "Io sono fame", "correct": "Ho fame", "type": "verb_choice", "explanation": "Use 'avere' for physical states"},
            {"wrong": "Il problema è facila", "correct": "Il problema è facile", "type": "adjective_agreement"},
            {"wrong": "Io mangio la pizza ieri", "correct": "Ho mangiato la pizza ieri", "type": "tense"},
        ],
        greetings=["Ciao!", "Buongiorno!", "Salve!", "Buonasera!"],
        responses={
            ResponseType.GREETING: [
                ("Ciao! Come stai oggi?", "Hello! How are you today?"),
                ("Buongiorno! Pronto per imparare?", "Good morning! Ready to learn?"),
            ],
            ResponseType.ENCOURAGEMENT: [
                ("Ottimo lavoro! Stai migliorando!", "Great work! You're improving!"),
                ("Benissimo! Continua così!", "Very good! Keep it up!"),
                ("Fantastico! La tua pronuncia migliora!", "Fantastic! Your pronunciation is improving!"),
            ],
            ResponseType.CORRECTION_GENTLE: [
                ("Quasi perfetto! Prova: '{correct}'", "Almost perfect! Try: '{correct}'"),
                ("Buon tentativo! Si dice: '{correct}'", "Good try! We say: '{correct}'"),
            ],
            ResponseType.CORRECTION_DIRECT: [
                ("Attenzione: '{wrong}' → '{correct}'", "Note: '{wrong}' → '{correct}'"),
                ("La forma corretta è: '{correct}'", "The correct form is: '{correct}'"),
            ],
            ResponseType.QUESTION_SIMPLE: [
                ("Come ti chiami?", "What's your name?"),
                ("Di dove sei?", "Where are you from?"),
                ("Che lavoro fai?", "What do you do for work?"),
            ],
            ResponseType.QUESTION_COMPLEX: [
                ("Cosa ne pensi della situazione attuale?", "What do you think about the current situation?"),
                ("Se potessi viaggiare ovunque, dove andresti?", "If you could travel anywhere, where would you go?"),
            ],
            ResponseType.VOCABULARY_INTRO: [
                ("Impariamo una nuova parola: '{word}' significa '{meaning}'", "Let's learn a new word: '{word}' means '{meaning}'"),
            ],
            ResponseType.PRACTICE_PROMPT: [
                ("Proviamo a dire: '{phrase}'", "Let's try saying: '{phrase}'"),
                ("Ripeti dopo di me: '{phrase}'", "Repeat after me: '{phrase}'"),
            ],
            ResponseType.CONVERSATION: [
                ("Interessante! Dimmi di più.", "Interesting! Tell me more."),
                ("Capisco. E poi?", "I understand. And then?"),
                ("Davvero? Che bello!", "Really? How nice!"),
            ],
        }
    ),
    
    "japanese": LanguageKnowledge(
        name="Japanese",
        code="ja",
        vocabulary={
            ProficiencyLevel.A1: ["こんにちは", "ありがとう", "はい", "いいえ", "すみません"],
            ProficiencyLevel.A2: ["食べます", "飲みます", "行きます", "来ます", "見ます"],
            ProficiencyLevel.B1: ["〜たいです", "〜てください", "〜ましょう"],
        },
        common_errors=[
            {"wrong": "私は学生います", "correct": "私は学生です", "type": "copula"},
            {"wrong": "本を読むます", "correct": "本を読みます", "type": "verb_conjugation"},
        ],
        greetings=["こんにちは！", "おはようございます！", "こんばんは！"],
        responses={
            ResponseType.GREETING: [
                ("こんにちは！元気ですか？", "Hello! How are you?"),
                ("はじめまして！よろしくお願いします。", "Nice to meet you!"),
            ],
            ResponseType.ENCOURAGEMENT: [
                ("すごい！上手になっていますね！", "Amazing! You're getting better!"),
                ("よくできました！", "Well done!"),
            ],
            ResponseType.CORRECTION_GENTLE: [
                ("惜しい！正しくは「{correct}」です。", "Close! The correct form is '{correct}'."),
            ],
            ResponseType.QUESTION_SIMPLE: [
                ("お名前は何ですか？", "What is your name?"),
                ("どこから来ましたか？", "Where are you from?"),
            ],
            ResponseType.CONVERSATION: [
                ("なるほど！もっと教えてください。", "I see! Tell me more."),
                ("それは面白いですね！", "That's interesting!"),
            ],
        }
    ),
    
    "spanish": LanguageKnowledge(
        name="Spanish",
        code="es",
        vocabulary={
            ProficiencyLevel.A1: ["hola", "gracias", "por favor", "sí", "no", "buenos días"],
            ProficiencyLevel.A2: ["quisiera", "cuánto cuesta", "dónde está", "me llamo"],
            ProficiencyLevel.B1: ["podría", "sería", "me gustaría", "hubiera"],
        },
        common_errors=[
            {"wrong": "Yo soy hambre", "correct": "Tengo hambre", "type": "verb_choice"},
            {"wrong": "El problema es fácila", "correct": "El problema es fácil", "type": "adjective_agreement"},
        ],
        greetings=["¡Hola!", "¡Buenos días!", "¡Buenas tardes!"],
        responses={
            ResponseType.GREETING: [
                ("¡Hola! ¿Cómo estás hoy?", "Hello! How are you today?"),
                ("¡Buenos días! ¿Listo para aprender?", "Good morning! Ready to learn?"),
            ],
            ResponseType.ENCOURAGEMENT: [
                ("¡Muy bien! ¡Sigue así!", "Very good! Keep it up!"),
                ("¡Excelente trabajo!", "Excellent work!"),
            ],
            ResponseType.CORRECTION_GENTLE: [
                ("¡Casi! La forma correcta es: '{correct}'", "Almost! The correct form is: '{correct}'"),
            ],
            ResponseType.CONVERSATION: [
                ("¡Qué interesante! Cuéntame más.", "How interesting! Tell me more."),
            ],
        }
    ),
    
    "french": LanguageKnowledge(
        name="French",
        code="fr",
        greetings=["Bonjour!", "Salut!", "Bonsoir!"],
        responses={
            ResponseType.GREETING: [
                ("Bonjour! Comment allez-vous?", "Hello! How are you?"),
            ],
            ResponseType.ENCOURAGEMENT: [
                ("Très bien! Continuez!", "Very good! Keep going!"),
            ],
        }
    ),
    
    "german": LanguageKnowledge(
        name="German",
        code="de",
        greetings=["Hallo!", "Guten Tag!", "Guten Morgen!"],
        responses={
            ResponseType.GREETING: [
                ("Hallo! Wie geht es Ihnen?", "Hello! How are you?"),
            ],
            ResponseType.ENCOURAGEMENT: [
                ("Sehr gut! Weiter so!", "Very good! Keep it up!"),
            ],
        }
    ),
}


class SimulatedLearner:
    """
    Simulates a language learner for RL training.
    Models realistic learning patterns and errors.
    """
    
    def __init__(
        self,
        target_language: str = "italian",
        initial_level: ProficiencyLevel = ProficiencyLevel.A1,
        personality: str = "balanced"  # "confident", "shy", "perfectionist", "balanced"
    ):
        self.target_language = target_language
        self.language_data = LANGUAGES.get(target_language, LANGUAGES["italian"])
        
        # Initialize state based on level
        level_factor = initial_level.value / 5.0
        
        self.state = LearnerState(
            vocabulary=0.1 + level_factor * 0.15,
            grammar=0.1 + level_factor * 0.15,
            pronunciation=0.1 + level_factor * 0.1,
            listening=0.15 + level_factor * 0.1,
            fluency=0.05 + level_factor * 0.15,
            level=initial_level
        )
        
        # Personality affects learning
        self.personality = personality
        self._apply_personality()
    
    def _apply_personality(self):
        """Apply personality modifiers"""
        if self.personality == "confident":
            self.state.confidence = 0.8
            self.state.motivation = 0.9
        elif self.personality == "shy":
            self.state.confidence = 0.3
            self.state.motivation = 0.6
        elif self.personality == "perfectionist":
            self.state.confidence = 0.4
            self.state.frustration = 0.2
            self.state.motivation = 0.85
    
    def generate_utterance(self, context: str = "general") -> Tuple[str, bool, Optional[Dict]]:
        """
        Generate a learner utterance.
        Returns: (utterance, is_correct, error_info)
        """
        # Probability of error based on proficiency
        error_prob = 1.0 - (self.state.grammar * 0.5 + self.state.vocabulary * 0.3 + self.state.confidence * 0.2)
        error_prob = max(0.1, min(0.8, error_prob))
        
        makes_error = random.random() < error_prob
        
        if makes_error and self.language_data.common_errors:
            # Generate utterance with error
            error_info = random.choice(self.language_data.common_errors)
            return error_info["wrong"], False, error_info
        else:
            # Generate correct utterance
            level = self.state.level
            vocab_list = self.language_data.vocabulary.get(level, [])
            
            if vocab_list:
                phrase = random.choice(vocab_list)
            else:
                phrase = random.choice(self.language_data.greetings)
            
            return phrase, True, None
    
    def receive_feedback(self, response_type: ResponseType, feedback_quality: float = 0.5):
        """
        Update learner state based on tutor response.
        This simulates learning progression.
        """
        self.state.turns_this_session += 1
        
        # Base learning rate
        lr = 0.05 * (0.5 + self.state.motivation * 0.5)
        
        if response_type == ResponseType.ENCOURAGEMENT:
            self.state.confidence = min(1.0, self.state.confidence + lr * 2)
            self.state.motivation = min(1.0, self.state.motivation + lr)
            self.state.frustration = max(0.0, self.state.frustration - lr)
            
        elif response_type in [ResponseType.CORRECTION_GENTLE, ResponseType.CORRECTION_DIRECT]:
            self.state.errors_this_session += 1
            
            # Gentle correction is better for confidence
            if response_type == ResponseType.CORRECTION_GENTLE:
                self.state.grammar = min(1.0, self.state.grammar + lr * feedback_quality)
                self.state.confidence = max(0.1, self.state.confidence - lr * 0.3)
            else:
                # Direct correction: more learning, but can hurt confidence
                self.state.grammar = min(1.0, self.state.grammar + lr * 1.2 * feedback_quality)
                self.state.confidence = max(0.1, self.state.confidence - lr * 0.6)
                self.state.frustration = min(1.0, self.state.frustration + lr * 0.3)
                
        elif response_type == ResponseType.VOCABULARY_INTRO:
            self.state.vocabulary = min(1.0, self.state.vocabulary + lr * feedback_quality)
            
        elif response_type == ResponseType.GRAMMAR_EXPLANATION:
            self.state.grammar = min(1.0, self.state.grammar + lr * 0.8 * feedback_quality)
            
        elif response_type == ResponseType.PRONUNCIATION_TIP:
            self.state.pronunciation = min(1.0, self.state.pronunciation + lr * feedback_quality)
            
        elif response_type == ResponseType.PRACTICE_PROMPT:
            self.state.fluency = min(1.0, self.state.fluency + lr * 0.5 * feedback_quality)
            self.state.correct_this_session += 1
            
        elif response_type == ResponseType.CONVERSATION:
            self.state.fluency = min(1.0, self.state.fluency + lr * 0.3)
            self.state.listening = min(1.0, self.state.listening + lr * 0.2)
        
        # Update engagement based on variety and appropriateness
        self.state.engagement = max(0.1, min(1.0, 
            self.state.engagement - 0.02 +  # Natural decay
            self.state.motivation * 0.03 -   # Motivation helps
            self.state.frustration * 0.05    # Frustration hurts
        ))
        
        # Check for level up
        avg_skill = (self.state.vocabulary + self.state.grammar + 
                    self.state.pronunciation + self.state.fluency) / 4
        
        current_level_threshold = (self.state.level.value + 1) * 0.15
        if avg_skill > current_level_threshold and self.state.level.value < 5:
            self.state.level = ProficiencyLevel(self.state.level.value + 1)


class LanguageLearningEnvironment:
    """
    RL Environment for training the language tutor agent.
    """
    
    def __init__(
        self,
        languages: List[str] = None,
        max_turns: int = 30,
        curriculum: bool = True
    ):
        self.languages = languages or list(LANGUAGES.keys())
        self.max_turns = max_turns
        self.curriculum = curriculum
        
        # Current episode
        self.learner: SimulatedLearner = None
        self.current_language: str = None
        self.turn: int = 0
        self.episode_reward: float = 0
        
        # For curriculum learning
        self.difficulty_level: int = 0
        self.successes_at_level: int = 0
        self.required_successes: int = 5
    
    def reset(self, language: str = None, level: ProficiencyLevel = None) -> Dict[str, torch.Tensor]:
        """Reset environment for new episode"""
        self.turn = 0
        self.episode_reward = 0
        
        # Select language
        self.current_language = language or random.choice(self.languages)
        
        # Curriculum: start easy, increase difficulty
        if self.curriculum and level is None:
            level = ProficiencyLevel(min(self.difficulty_level, 5))
        elif level is None:
            level = random.choice(list(ProficiencyLevel))
        
        # Select personality
        personalities = ["confident", "shy", "perfectionist", "balanced"]
        personality = random.choice(personalities)
        
        # Create learner
        self.learner = SimulatedLearner(
            target_language=self.current_language,
            initial_level=level,
            personality=personality
        )
        
        # Get initial observation
        utterance, is_correct, error_info = self.learner.generate_utterance()
        
        return self._make_observation(utterance, is_correct, error_info)
    
    def _make_observation(
        self,
        utterance: str,
        is_correct: bool,
        error_info: Optional[Dict]
    ) -> Dict[str, Any]:
        """Create observation dict"""
        lang_idx = self.languages.index(self.current_language)
        
        return {
            "utterance": utterance,
            "is_correct": is_correct,
            "error_info": error_info,
            "language_idx": torch.tensor(lang_idx),
            "proficiency": self.learner.state.to_proficiency_vector(),
            "learner_state": self.learner.state,
            "turn": self.turn
        }
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute action and return (observation, reward, done, info)
        
        Action space: ResponseType enum values
        """
        self.turn += 1
        response_type = ResponseType(action % len(ResponseType))
        
        # Calculate reward
        reward = self._calculate_reward(response_type)
        self.episode_reward += reward
        
        # Update learner
        feedback_quality = 0.5 + random.uniform(-0.2, 0.2)  # Some noise
        self.learner.receive_feedback(response_type, feedback_quality)
        
        # Check if done
        done = (
            self.turn >= self.max_turns or
            self.learner.state.engagement < 0.15 or
            self.learner.state.frustration > 0.9 or
            (self.learner.state.vocabulary > 0.9 and self.learner.state.grammar > 0.9)
        )
        
        # Get next observation
        if not done:
            utterance, is_correct, error_info = self.learner.generate_utterance()
            obs = self._make_observation(utterance, is_correct, error_info)
        else:
            obs = self._make_observation("", True, None)
            
            # Curriculum update
            if self.curriculum:
                if self.episode_reward > 5.0:  # Success threshold
                    self.successes_at_level += 1
                    if self.successes_at_level >= self.required_successes:
                        self.difficulty_level = min(5, self.difficulty_level + 1)
                        self.successes_at_level = 0
                        print(f"📈 Curriculum advanced to level {self.difficulty_level}")
        
        info = {
            "response_type": response_type.name,
            "episode_reward": self.episode_reward,
            "learner_state": {
                "vocabulary": self.learner.state.vocabulary,
                "grammar": self.learner.state.grammar,
                "confidence": self.learner.state.confidence,
                "engagement": self.learner.state.engagement,
                "level": self.learner.state.level.name
            }
        }
        
        return obs, reward, done, info
    
    def _calculate_reward(self, response_type: ResponseType) -> float:
        """
        Calculate reward based on pedagogical appropriateness.
        
        Good tutoring:
        - Correct when needed, encourage when struggling
        - Match difficulty to learner level
        - Maintain engagement
        """
        reward = 0.0
        state = self.learner.state
        
        # Base reward for engagement
        reward += state.engagement * 0.1
        
        # Response appropriateness
        if response_type == ResponseType.ENCOURAGEMENT:
            if state.confidence < 0.4:
                reward += 1.0  # Good to encourage struggling learners
            elif state.confidence > 0.8:
                reward += 0.2  # Less impactful for confident learners
            else:
                reward += 0.5
                
        elif response_type in [ResponseType.CORRECTION_GENTLE, ResponseType.CORRECTION_DIRECT]:
            # Check if learner made an error
            error_rate = state.errors_this_session / max(state.turns_this_session, 1)
            
            if error_rate > 0.3:
                # Learner is making errors, correction is good
                if response_type == ResponseType.CORRECTION_GENTLE and state.confidence < 0.5:
                    reward += 1.2  # Gentle correction for struggling learner
                elif response_type == ResponseType.CORRECTION_DIRECT and state.confidence > 0.6:
                    reward += 1.0  # Direct correction OK for confident learner
                else:
                    reward += 0.6
            else:
                # Learner isn't making many errors
                reward -= 0.3  # Unnecessary correction
                
        elif response_type == ResponseType.VOCABULARY_INTRO:
            if state.vocabulary < 0.5:
                reward += 0.8  # Good to teach vocab to beginners
            else:
                reward += 0.3
                
        elif response_type == ResponseType.PRACTICE_PROMPT:
            if state.fluency < 0.4:
                reward += 0.7  # Practice helps fluency
            else:
                reward += 0.4
                
        elif response_type == ResponseType.QUESTION_SIMPLE:
            if state.level.value <= 1:
                reward += 0.5  # Simple questions for beginners
            else:
                reward += 0.2
                
        elif response_type == ResponseType.QUESTION_COMPLEX:
            if state.level.value >= 3:
                reward += 0.6  # Complex questions for advanced
            else:
                reward -= 0.2  # Too hard for beginners
                
        elif response_type == ResponseType.CONVERSATION:
            reward += 0.3 + state.engagement * 0.2  # Natural conversation
        
        # Penalty for causing frustration
        if state.frustration > 0.5:
            reward -= 0.3
        
        # Penalty for losing engagement
        if state.engagement < 0.3:
            reward -= 0.5
        
        return reward


# Quick test
if __name__ == "__main__":
    print("🎓 Testing Language Learning Environment...")
    print("=" * 60)
    
    env = LanguageLearningEnvironment(curriculum=True)
    
    for episode in range(3):
        obs = env.reset()
        print(f"\n📚 Episode {episode + 1}")
        print(f"   Language: {env.current_language}")
        print(f"   Level: {env.learner.state.level.name}")
        print(f"   Personality: {env.learner.personality}")
        
        done = False
        total_reward = 0
        
        while not done:
            # Random action for testing
            action = random.randint(0, len(ResponseType) - 1)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        
        print(f"   Turns: {env.turn}")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Final State: {info['learner_state']}")
    
    print("\n" + "=" * 60)
    print("🎉 Environment tests passed!")
