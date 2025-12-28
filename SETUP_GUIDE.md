# 🚀 Language Mirror Pro - Complete Setup Guide

This guide will help you set up and run Language Mirror Pro on your Mac M4 Pro.

**NEW: Now includes React Native mobile app with TanStack Query!**

---

## 📋 Prerequisites

- **Python 3.10+** 
- **Node.js 18+**
- **8GB RAM minimum**
- **Expo Go app** (for mobile testing - iOS/Android)

---

## 🔧 Step-by-Step Setup

### Step 1: Extract and Navigate

```bash
# Extract the zip
cd ~/Downloads
unzip language-mirror-pro.zip

# Move to Projects
mv language-mirror-pro ~/Projects/
cd ~/Projects/language-mirror-pro
```

### Step 2: Backend Setup

```bash
# Create Python virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch numpy fastapi uvicorn pydantic tqdm edge-tts websockets python-multipart

# Optional: Install Whisper for speech-to-text
# pip install openai-whisper
```

### Step 3: Test the AI Model

```bash
# Test the custom transformer model
python -m ai_core.models.transformer
```

**Expected output:**
```
🧠 Testing Language Mirror Pro Model...
📊 Total Parameters: 12,XXX,XXX (12.XM)
✅ Encoder output: torch.Size([2, 32, 384])
✅ Grammar error logits: ...
🎉 All tests passed! Model ready for training.
```

### Step 4: Test the Training Environment

```bash
python -m ai_core.training.environment
```

**Expected output:**
```
🎓 Testing Language Learning Environment...
📚 Episode 1
   Language: italian
   Level: A1
   Total Reward: X.XX
🎉 Environment tests passed!
```

### Step 5: Start the Backend Server

```bash
cd backend
python main.py
```

**Expected output:**
```
🚀 Starting Language Mirror Pro API...
✅ Model loaded on cpu
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Leave this terminal running!**

### Step 6: Setup Mobile App (New Terminal)

```bash
# Open a new terminal (Cmd + T)
cd ~/Projects/language-mirror-pro/mobile

# Install Node dependencies
npm install

# Start the mobile app
npm start
```

**Expected output:**
```
▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
█ QR CODE HERE                █
▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

› Press i │ open iOS simulator
› Press a │ open Android emulator
› Press w │ open web
```

### Step 7: Open on Your Phone! 🎉

**Option A: Physical Device (Recommended)**
1. Install "Expo Go" app on your phone
2. Scan the QR code with your camera (iOS) or Expo Go (Android)

**Option B: Simulator**
- Press `i` for iOS Simulator
- Press `a` for Android Emulator

**⚠️ Important for Physical Device:**
The mobile app needs to connect to your backend. Edit `mobile/lib/api.ts`:

```typescript
// Change this:
export const API_URL = __DEV__ 
  ? 'http://localhost:8000'
  : 'https://your-api.com';

// To your computer's IP:
export const API_URL = 'http://192.168.1.XXX:8000';
```

Find your IP: `ifconfig | grep "inet " | grep -v 127.0.0.1`

---

## 🧪 Testing the App

### Test 1: Basic Conversation

1. Click **"Start Learning"**
2. Select **Italian** → **Standard**
3. Choose **Free Chat** scenario
4. Click **"Start Conversation"**
5. Type: `Ciao, come stai?`
6. Press Enter or click Send

### Test 2: Grammar Correction

Type an intentional error:
```
Io sono fame
```

The tutor should correct you:
```
"Quasi perfetto! Si dice 'Ho fame'. In italiano usiamo 'avere' per la fame!"
```

### Test 3: Try Different Languages

1. Click the Settings icon (⚙️)
2. Select **Japanese**
3. Type: `こんにちは`
4. The tutor responds in Japanese with translation!

---

## 🏋️ Training the Model (Optional)

To train your custom AI model:

```bash
cd ~/Projects/language-mirror-pro
source venv/bin/activate

# Quick training (10-15 minutes)
python scripts/train.py --num_updates 100 --log_interval 10

# Full training (1-2 hours)
python scripts/train.py --num_updates 1000 --save_interval 100
```

Trained models are saved to `checkpoints/`.

---

## 🔊 Speech Features

### Text-to-Speech (TTS)

Already installed with `edge-tts`. Test it:

```bash
cd ~/Projects/language-mirror-pro
source venv/bin/activate
python -m speech.tts
```

### Speech-to-Text (STT) - Optional

Install Whisper for voice input:

```bash
pip install openai-whisper
python -m speech.stt
```

---

## 🐛 Troubleshooting

### "Module not found" error
```bash
cd ~/Projects/language-mirror-pro
source venv/bin/activate
pip install -r requirements.txt
```

### "Port already in use"
```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9
lsof -ti:3000 | xargs kill -9
```

### Backend not responding
1. Check if backend is running on port 8000
2. Check terminal for errors
3. Restart with: `python main.py`

### Frontend not loading
1. Make sure `npm install` completed
2. Check for errors in terminal
3. Try: `rm -rf .next node_modules && npm install && npm run dev`

---

## 📱 Demo Video Tips

When recording your demo:

1. **Show the Welcome Screen** - Beautiful gradient animation
2. **Select Language** - Show the flag icons
3. **Have a Conversation** - 3-4 turns in Italian
4. **Make an Error** - Show grammar correction
5. **Show Translation Toggle** - Click the globe icon
6. **Briefly Show Code** - Scroll through transformer.py

---

## 🏆 Key Points for Judges

When presenting, emphasize:

1. **"100% Custom AI"** - No OpenAI/Google APIs
2. **"12M Parameter Transformer"** - Real deep learning
3. **"PPO Reinforcement Learning"** - State-of-the-art training
4. **"Multi-Task Learning"** - Grammar + Pronunciation + Response
5. **"Runs Locally"** - Privacy, no API costs

---

## 📞 Quick Commands Reference

```bash
# Start backend
cd ~/Projects/language-mirror-pro
source venv/bin/activate
cd backend && python main.py

# Start mobile app (new terminal)
cd ~/Projects/language-mirror-pro/mobile
npm start

# Train model
cd ~/Projects/language-mirror-pro
source venv/bin/activate
python scripts/train.py --num_updates 100

# Test model
python -m ai_core.models.transformer

# Test TTS
python -m speech.tts
```

---

## 📱 Mobile App Architecture

The mobile app uses:

| Technology | Purpose |
|------------|---------|
| **Expo SDK 50** | React Native framework |
| **TanStack Query v5** | Data fetching & caching |
| **Expo Router** | File-based navigation |
| **Reanimated 3** | 60fps animations |

### Key Files

```
mobile/
├── lib/api.ts          # TanStack Query client + API
├── hooks/useApi.ts     # Custom query hooks
├── app/_layout.tsx     # QueryClientProvider setup
├── app/index.tsx       # Welcome screen
├── app/setup.tsx       # Language selection
└── app/chat.tsx        # Main chat UI
```

---

**Good luck with the hackathon! 🏆**
