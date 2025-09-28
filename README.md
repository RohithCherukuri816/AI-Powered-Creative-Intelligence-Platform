# ✨ AI-Powered Creative Intelligence Platform

Transform your doodles into stunning designs with AI! Sketch, prompt, and generate beautiful artwork in seconds.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.11 (recommended) or 3.8-3.12
- npm or pnpm
- NVIDIA GPU with CUDA 11.8+ (optional, for faster AI processing)

### Automated Setup (Recommended)

**Windows:**
```bash
scripts/setup.bat
```

**macOS/Linux:**
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Manual Setup

1. **Install frontend dependencies:**
```bash
cd frontend && npm install && cd ..
```

2. **Install backend dependencies:**

**Option A - Auto-detect (recommended):**
```bash
cd backend && pip install -r requirements.txt && cd ..
```

**Option B - CPU-only (no GPU):**
```bash
cd backend && pip install -r requirements-cpu.txt && cd ..
```

**Option C - GPU with CUDA:**
```bash
cd backend && pip install -r requirements-gpu.txt && cd ..
```

3. **Set up environment files:**
```bash
cp .env.example .env
cp frontend/.env.example frontend/.env
cp backend/.env.example backend/.env
```

4. **Setup AI models (optional but recommended):**
```bash
cd backend
python scripts/setup-ai.py
```

5. **Configure Hugging Face token (for AI models):**
   - Get a free token at https://huggingface.co/settings/tokens
   - Add it to your `.env` file: `HUGGINGFACE_TOKEN=your_token_here`

6. **Start development servers:**
```bash
npm run dev
```

This will start:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 🤖 AI Models

The platform uses **ControlNet + Stable Diffusion** for AI-powered image generation:

- **ControlNet**: Preserves the structure of your sketch
- **Stable Diffusion**: Applies artistic styles based on your prompts
- **Fallback**: Traditional image processing if AI models aren't available

**System Requirements for AI:**
- 8GB+ RAM (16GB recommended)
- NVIDIA GPU with 6GB+ VRAM (optional, but much faster)
- ~10GB disk space for model downloads

## 🐍 Python Version Compatibility

| Python Version | Status | Notes |
|----------------|--------|-------|
| 3.8 | ✅ Supported | Minimum version |
| 3.9 | ✅ Supported | Stable |
| 3.10 | ✅ Supported | Stable |
| 3.11 | ✅ **Recommended** | Best performance |
| 3.12 | ⚠️ Limited | Some packages may have issues |

**Installation Tips:**
- Use `requirements.txt` for automatic detection
- Use `requirements-cpu.txt` for CPU-only systems
- Use `requirements-gpu.txt` for NVIDIA GPU systems
- Virtual environment recommended: `python -m venv venv`

## 🏗️ Project Structure

```
ai-creative-platform/
├── frontend/          # React + Vite + Tailwind
├── backend/           # FastAPI + Python
├── .env              # Shared environment variables
├── package.json      # Root package.json with workspaces
└── README.md
```

## 🎨 Features

- **Interactive Drawing Canvas** - Sketch your ideas with customizable brushes
- **ControlNet + Stable Diffusion** - AI-powered transformation that preserves your sketch structure
- **AI Style Prompts** - Describe your desired style and aesthetic
- **Real-time Generation** - Transform doodles into professional designs with AI
- **Beautiful Pastel UI** - Modern, playful interface with smooth animations
- **Fallback Processing** - Works even without AI models using traditional image processing

## 🛠️ Tech Stack

**Frontend:**
- React 18 + Vite
- Tailwind CSS + shadcn/ui
- Framer Motion
- HTML5 Canvas

**Backend:**
- FastAPI
- Python 3.8+
- ControlNet + Stable Diffusion (AI models)
- PIL/Pillow for image processing
- PyTorch for deep learning
- CORS enabled

---

**Made by Rohith Cherukuri** 💜