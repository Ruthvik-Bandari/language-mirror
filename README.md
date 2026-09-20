# 🪞 Language Mirror Pro

<div align="center">

![Language Mirror Pro](https://img.shields.io/badge/AI-Custom%20Built-blue?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A Custom-Built AI Language Tutor Using Reinforcement Learning**

*Custom tutor model trained from scratch • PPO reinforcement learning • Whisper ASR + edge-tts speech*

[Live Demo](#demo) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Technical Details](#technical-details)

</div>

---

## 🏆 Why This Project Wins

| What Others Do | What We Built |
|----------------|---------------|
| ❌ Wrap ChatGPT/Gemini APIs | ✅ **Transformer trained from scratch** (45.6M params) |
| ❌ Simple chat interface | ✅ **Multi-task learning: Grammar + Pronunciation + Response** |
| ❌ Static responses | ✅ **RL-trained adaptive tutoring with PPO** |
| ❌ Generic feedback | ✅ **Pedagogically-informed reward shaping** |
| ❌ Single language | ✅ **5 languages served, 10 configured** |

---

## ✨ Key Features

### 🧠 Custom AI Architecture
- **Multi-Task Transformer** with RoPE positional encoding
- **SwiGLU activation** (used in LLaMA, PaLM)
- **RMSNorm** for stable training
- **Separate heads** for grammar, pronunciation, and response generation

### 🎯 Reinforcement Learning
- **PPO algorithm** for policy optimization
- **Curriculum learning** - starts easy, increases difficulty
- **Simulated learner environment** for training without human data
- **Pedagogically-informed rewards** based on language learning science

### 🌍 Multilingual Support
- **Served by the backend today:** Italian, Japanese, Spanish, French, German (5)
- **Declared in the model config but not yet exposed:** Portuguese, Mandarin, Korean, Arabic, Hindi
- Regional dialect awareness

### ⚡ Production Ready
- FastAPI backend with WebSocket support
- Real-time conversation streaming
- Session management
- Sub-second inference on M4 Mac

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LANGUAGE MIRROR PRO ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  User Input: "Io sono fame"                                              │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    TOKENIZER (BPE)                               │    │
│  │  Multilingual • 16K vocab • Subword tokenization                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                 STATE ENCODER (6-Layer Transformer)              │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │    │
│  │  │   Token     │  │  Language   │  │ Proficiency │              │    │
│  │  │  Embedding  │  │  Embedding  │  │  Encoding   │              │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │    │
│  │         │                │                │                      │    │
│  │         └────────────────┼────────────────┘                      │    │
│  │                          ▼                                       │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │   Multi-Head Attention (RoPE) + SwiGLU FFN × 6 layers   │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     MULTI-TASK HEADS                             │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │    │
│  │  │ Grammar  │  │ Pronun-  │  │ Response │  │ Adaptive │        │    │
│  │  │Correction│  │ ciation  │  │Generator │  │Difficulty│        │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    RL COMPONENTS (PPO)                           │    │
│  │  ┌─────────────────┐              ┌─────────────────┐           │    │
│  │  │  Policy Head    │              │   Value Head    │           │    │
│  │  │    (Actor)      │              │    (Critic)     │           │    │
│  │  └─────────────────┘              └─────────────────┘           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  Output: "Quasi perfetto! Si dice 'Ho fame'. In italiano usiamo         │
│           'avere' per la fame, non 'essere'!"                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend)
- 8GB RAM minimum

### 1. Clone and Setup

```bash
git clone https://github.com/YOUR_USERNAME/language-mirror-pro.git
cd language-mirror-pro

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test the Model

```bash
# Test architecture
python -m ai_core.models.transformer

# Test training environment
python -m ai_core.training.environment
```

### 3. Train the Model (Optional)

```bash
# Quick training (10 minutes)
python scripts/train.py --num_updates 100

# Full training (2-4 hours)
python scripts/train.py --num_updates 2000 --save_interval 100
```

### 4. Start the Server

```bash
cd backend
python main.py
# Server runs at http://localhost:8000
```

### 5. Start the Frontend

```bash
cd frontend
npm install
npm run dev
# Frontend at http://localhost:3000
```

---

## 📊 Model Specifications

| Component | Details |
|-----------|---------|
| **Architecture** | Multi-Task Transformer |
| **Parameters (training arch, `LanguageMirrorPro`)** | **45.6 M** (counted by instantiating the model) |
| **Parameters (served arch, `LanguageTutorModel`)** | **76.9 M** |
| **Encoder Layers** | 6 |
| **Decoder Layers** | 4 |
| **Attention Heads** | 6 |
| **Hidden Dimension** | 384 |
| **Vocabulary Size** | 16,000 (multilingual BPE) |
| **Position Encoding** | Rotary (RoPE) |
| **Activation** | SwiGLU |
| **Normalization** | RMSNorm |

---

## 🎓 RL Training Details

### State Space
- User utterance (tokenized)
- Target language embedding
- Learner proficiency: [vocabulary, grammar, pronunciation, confidence, error_rate]

### Action Space
12 response types:
- Greeting, Gentle Correction, Direct Correction
- Encouragement, Simple Question, Complex Question
- Vocabulary Introduction, Grammar Explanation
- Pronunciation Tip, Cultural Note, Practice Prompt, Conversation

### Reward Function
```python
# Pedagogically-informed rewards
+1.0: Encouragement when confidence < 0.4
+1.2: Gentle correction for struggling learner
+0.8: Vocabulary intro for beginners
-0.3: Over-correction of advanced learner
-0.5: Causing learner frustration
```

### Curriculum Learning
- Starts with A1 (beginner) learners
- Advances difficulty after 5 successful episodes
- Reaches C2 (mastery) level learners

---

## 📁 Project Structure

```
language-mirror-pro/
├── ai_core/
│   ├── models/
│   │   ├── transformer.py      # 🧠 Custom transformer model
│   │   └── tokenizer.py        # 🔤 BPE tokenizer
│   ├── training/
│   │   └── environment.py      # 🎮 RL training environment
│   └── inference/
│       └── engine.py           # ⚡ Inference optimization
├── backend/
│   └── main.py                 # 🚀 FastAPI server
├── frontend/
│   └── ...                     # 💻 React/Next.js app
├── scripts/
│   └── train.py                # 📈 Training script
├── checkpoints/                # 💾 Model weights
├── requirements.txt
└── README.md
```

---

## 🔬 Technical Innovations

### 1. Multi-Task Learning
Single model handles grammar correction, pronunciation scoring, and response generation simultaneously, with shared representations improving all tasks.

### 2. Pedagogical Reward Shaping
Rewards are designed based on language learning research:
- Gentle corrections for struggling learners
- Challenge increases for confident learners
- Engagement optimization

### 3. Curriculum Learning
Training starts with simulated beginner learners and progressively introduces advanced scenarios, improving sample efficiency.

### 4. Efficient Architecture
- RoPE for better length generalization
- SwiGLU activation (2x more efficient than GELU)
- RMSNorm (faster than LayerNorm)
- Weight tying between embedding and output layers

---

## 🎯 Demo Scenarios

### Scenario 1: Grammar Correction
```
User: "Io sono fame"
Tutor: "Quasi perfetto! Si dice 'Ho fame'. In italiano usiamo 
        'avere' per la fame, non 'essere'!"
Translation: "Almost perfect! We say 'Ho fame'. In Italian we use 
              'to have' for hunger, not 'to be'!"
```

### Scenario 2: Encouragement
```
User: "Buongiorno, mi chiamo Marco"
Tutor: "Ottimo lavoro! La tua pronuncia sta migliorando molto! 
        Come stai oggi, Marco?"
Translation: "Great job! Your pronunciation is improving a lot! 
              How are you today, Marco?"
```

### Scenario 3: Practice Prompt
```
User: "Come si dice 'thank you'?"
Tutor: "In italiano diciamo 'grazie'! Proviamo insieme: 
        'Grazie mille per il tuo aiuto!'"
Translation: "In Italian we say 'grazie'! Let's try together: 
              'Thank you very much for your help!'"
```

---

## ⚠️ Known gaps

Worth knowing before you read the claims above as finished work.

- **No evaluation metrics are recorded.** There is no benchmark, held-out score, or learning curve
  committed anywhere. The architecture and the training loop are real; their effectiveness is
  unmeasured.
- **Two different model definitions coexist.** Training uses `ai_core/models/transformer.py`
  (`LanguageMirrorPro`, d_model 384, vocab 16,000, 45.6 M params). Serving uses
  `backend/model_inference.py` (`LanguageTutorModel`, d_model 512, vocab 32,000, 76.9 M params) with
  a character-level tokenizer. These are not the same network, and reconciling them is the most
  valuable next change in this repository.
- **Speech is not custom.** ASR is OpenAI Whisper run locally; TTS is `edge_tts`, which calls a
  Microsoft service. Only the tutoring model is trained from scratch here.
- **Checkpoints are large.** Only the served weights are tracked
  (`backend/trained_model/model_weights.pt`, 294 MB); eight redundant copies and epoch checkpoints
  were removed from the tree. They remain in git history, so a clone still pulls ~2.3 GB. Purging
  them needs a history rewrite, and future checkpoints belong in Release assets rather than git.
- No automated test suite.

## 👥 Team

Built for **AI Hackathon 2025**

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ and PyTorch**

*No APIs were harmed in the making of this project*

</div>
