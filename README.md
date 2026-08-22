# 🌾 AI Crop Doctor

Multimodal, agentic AI assistant for farmers. Diagnose crop diseases through voice, text, and images using a single intelligent agent that reasons through evidence before diagnosing.

## Architecture

```
Farmer (Voice/Text/Image)
    │
    ▼
FastAPI Backend
    ├── Groq Whisper (STT) ←→ Groq Orpheus (TTS)
    ├── Groq Llama-4 Scout (Vision)
    └── CrewAI Diagnostic Agent
            ├── 🔧 analyze_crop_image (Groq Vision)
            ├── 🔧 search_knowledge_base (FAISS RAG)
            ├── 🔧 check_confidence (LLM evaluation)
            └── 🔧 ask_followup_question (conversation)
```

**Key design:** One agent with four tools, not four agents. The agent autonomously decides what evidence to gather through a ReAct reasoning loop.

## Quick Start

### 1. Prerequisites
- Python 3.11+
- [Groq API key](https://console.groq.com/) (free tier works)

### 2. Setup
```bash
# Clone and enter the project
cd AI-Crop-Doctor

# Create virtual environment (from repo root or inside backend)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Ingest Knowledge Base
```bash
cd backend
python -m app.rag.ingestion
```

### 4. Run the Server
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
Then visit **http://localhost:8000** in your browser to interact with the multimodal UI.

### 5. Test
```bash
# Health check
curl http://localhost:8000/api/health

# Text diagnosis
curl -X POST http://localhost:8000/api/diagnosis \
  -F "text=My tomato leaves have brown spots with rings"

# Image + text diagnosis
curl -X POST http://localhost:8000/api/diagnosis \
  -F "image=@leaf_photo.jpg" \
  -F "text=What disease is this?"
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System health check |
| POST | `/api/diagnosis` | Unified diagnosis (image + text + audio) |
| POST | `/api/voice/transcribe` | Speech-to-text (Groq Whisper) |
| POST | `/api/voice/synthesize` | Text-to-speech (Groq Orpheus / gTTS) |

## Project Structure

```
AI-Crop-Doctor/
│
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + lifespan + static mount
│   │   ├── config.py             # Pydantic Settings
│   │   ├── api/                  # Route handlers (diagnosis, voice, health)
│   │   ├── agent/                # CrewAI diagnostic agent + tools + flow
│   │   ├── services/             # Vision, speech (Groq + gTTS), RAG services
│   │   ├── rag/                  # Embeddings, vector store, ingestion
│   │   ├── session/              # Case/conversation management
│   │   ├── schemas/              # Request/response models
│   │   └── evaluation/           # Logging and metrics
│   ├── knowledge_base/raw/       # Agricultural reference data
│   ├── evaluation/               # Test cases and runner
│   ├── uploads/                  # Uploaded crop leaves/stems
│   ├── logs/                     # Application logs
│   ├── requirements.txt          # Python dependencies (including gTTS)
│   ├── Dockerfile                # Container definition
│   ├── .env                      # Environment variables
│   └── .env.example              # Example environment configuration
│
├── frontend/
│   ├── index.html                # Main UI interface
│   ├── app.js                    # UI logic, WebRTC, mic recorder & TTS playback
│   └── style.css                 # Styling & layout
│
└── README.md
```


## How the Agent Works

1. **Farmer provides input** (image, text, or voice)
2. **Agent assesses** available evidence
3. **Agent decides** if evidence is sufficient for diagnosis
4. If **insufficient**, the agent autonomously chooses a tool:
   - `analyze_crop_image` → Groq Vision analysis
   - `search_knowledge_base` → FAISS RAG retrieval
   - `ask_followup_question` → Ask farmer for more details
5. After receiving new evidence → **loop back to step 2**
6. When **confident (≥70%)** → provide diagnosis with treatments
7. If still **uncertain** after max iterations → escalate with honest confidence

## Evaluation

```bash
# Start the server, then:
python evaluation/evaluate.py
```

## Tech Stack

- **Backend:** FastAPI + Uvicorn
- **AI Orchestration:** CrewAI (single agent with tools)
- **LLM Integration:** LangChain + LangChain-Groq
- **Models:** Groq (Llama-4 Scout vision, Llama 3.3 70B text, Whisper STT, Orpheus TTS)
- **RAG:** FAISS + HuggingFace embeddings (local, free)
- **Frontend:** Next.js (coming in Phase 12)

## License

MIT
