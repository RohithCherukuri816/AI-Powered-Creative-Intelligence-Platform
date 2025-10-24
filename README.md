# 🎨 SketchCraft - AI-Powered Creative Intelligence Platform

<div align="center">

![Version](https://img.shields.io/badge/version-5.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![Node](https://img.shields.io/badge/node-18%2B-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

**Transform your doodles into stunning realistic designs with AI-powered recognition and style transformation**

[Features](#-features) • [Quick Start](#-quick-start) • [Tech Stack](#-tech-stack) • [Training](#-train-your-own-model) • [Documentation](#-documentation)

</div>

---

## 🌟 What is SketchCraft?

SketchCraft is an intelligent creative platform that combines **doodle recognition** with **AI-powered image transformation**. Draw anything, and watch as the system:

1. **🔍 Automatically recognizes** what you drew (car, cat, house, etc.)
2. **🎨 Transforms it** into a realistic, styled image based on your prompt
3. **✨ Applies artistic effects** using ControlNet + Stable Diffusion

### 🎯 Key Capabilities

- **100+ Doodle Categories** - Recognizes vehicles, animals, objects, and more
- **Smart Recognition** - CNN-based model with 87%+ accuracy
- **Realistic Transformation** - Converts sketches to photorealistic images
- **Style Flexibility** - Watercolor, digital art, minimalist, vintage, and more
- **Dual Backend System** - Advanced AI backend + fallback processing
- **Production Ready** - Complete training pipeline included

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|------------|---------|---------|
| **Node.js** | 18+ | Frontend development |
| **Python** | 3.8-3.12 | Backend & AI models |
| **npm/pnpm** | Latest | Package management |
| **NVIDIA GPU** | Optional | Faster AI processing (CUDA 11.8+) |

### 🎬 Installation (3 Minutes)

#### Option 1: Automated Setup (Recommended)

**Windows:**
```bash
scripts\setup.bat
```

**macOS/Linux:**
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

#### Option 2: Manual Setup

```bash
# 1. Install frontend dependencies
cd frontend && npm install && cd ..

# 2. Install backend dependencies
cd backend && pip install -r requirements.txt && cd ..

# 3. Setup environment files
cp .env.example .env
cp frontend/.env.example frontend/.env
cp backend/.env.example backend/.env

# 4. Start development servers
npm run dev
```

### 🌐 Access Your Application

Once started, access:
- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

---

## ✨ Features

### 🎨 Interactive Drawing Canvas

- **Multi-tool Support** - Brush, eraser with customizable sizes
- **Color Palette** - 10+ colors for creative expression
- **Undo/Redo** - Full history management
- **Keyboard Shortcuts** - Professional workflow (Ctrl+Z, Ctrl+S, etc.)
- **Touch Support** - Works on tablets and touch devices
- **Download Sketches** - Save your drawings locally

### 🤖 AI-Powered Recognition

- **Automatic Detection** - Recognizes 100+ doodle categories
- **High Accuracy** - 87% test accuracy, 96% top-3 accuracy
- **Confidence Scoring** - Shows recognition confidence
- **Smart Fallback** - Heuristic recognition when model unavailable
- **Real-time Processing** - <50ms inference time

### 🎭 Style Transformation

- **ControlNet Integration** - Preserves sketch structure
- **Stable Diffusion** - Applies artistic styles
- **Multiple Styles:**
  - 🌊 Watercolor painting
  - 💻 Digital art
  - ✏️ Minimalist line art
  - 📺 Vintage poster
  - 🔷 Geometric patterns
  - 📸 Photorealistic

### 🎯 Smart Prompt Integration

- **Auto-Enhancement** - Combines recognized label with user prompt
- **Style Detection** - Analyzes prompt for style keywords
- **Quality Boosters** - Adds quality enhancers automatically
- **Negative Prompts** - Filters out unwanted elements

---

## 🏗️ Architecture

### Project Structure

```
ai-creative-platform/
├── 📁 frontend/                    # React + Vite Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── DrawingCanvas.jsx       # Interactive canvas
│   │   │   ├── PromptInput.jsx         # Style prompt input
│   │   │   ├── GeneratedImageCard.jsx  # Result display
│   │   │   ├── LoadingSpinner.jsx      # Loading animation
│   │   │   ├── KeyboardShortcuts.jsx   # Shortcuts helper
│   │   │   └── TouchGestures.jsx       # Touch support
│   │   ├── App.jsx                     # Main application
│   │   ├── api.js                      # Backend API client
│   │   └── index.css                   # Tailwind styles
│   ├── package.json
│   ├── tailwind.config.js              # Custom theme
│   └── vite.config.js
│
├── 📁 backend/                     # FastAPI Backend (Fallback)
│   ├── routes/
│   │   └── design.py                   # API endpoints
│   ├── utils/
│   │   └── image_processor.py          # Image transformations
│   ├── config/
│   │   └── ai_config.py                # AI configuration
│   ├── main.py                         # FastAPI app
│   ├── requirements.txt                # Python dependencies
│   ├── requirements-cpu.txt            # CPU-only version
│   └── requirements-gpu.txt            # GPU version
│
├── 📁 colab_backend/               # Advanced AI Backend
│   ├── app.py                          # Smart recognition API
│   ├── recognizer.py                   # CNN doodle recognizer
│   ├── processor.py                    # ControlNet processor
│   └── models/                         # Trained models directory
│       ├── doodle_classifier.h5
│       ├── doodle_classifier_metadata.json
│       └── class_names.json
│
├── 📁 training/                    # Model Training Pipeline
│   ├── kaggle_training.ipynb           # ⭐ Kaggle notebook
│   ├── config.py                       # Training configuration
│   ├── download_dataset.py             # Dataset downloader
│   ├── data_preprocessing.py           # Data preprocessing
│   ├── model_architecture.py           # CNN architectures
│   ├── train_model.py                  # Training script
│   ├── evaluate_model.py               # Model evaluation
│   ├── export_model.py                 # Model export
│   ├── quick_train.py                  # Quick training
│   ├── run_training.bat                # Windows automation
│   ├── run_training.sh                 # Unix automation
│   └── utils/                          # Training utilities
│
├── 📁 scripts/                     # Setup Scripts
│   ├── setup.bat                       # Windows setup
│   └── setup.sh                        # Unix setup
│
├── 📄 Documentation
│   ├── README.md                       # This file
│   ├── PROJECT_OVERVIEW.md             # Project details
│   ├── HOW_TO_TRAIN.md                 # Training guide
│   ├── INTEGRATION_GUIDE.md            # Integration guide
│   ├── TRAINING_INDEX.md               # Documentation index
│   ├── TRAINING_SUMMARY.md             # Quick reference
│   ├── START_HERE.md                   # Quick start
│   └── WHAT_I_CREATED.md               # File inventory
│
├── dev.js                          # Development server
├── package.json                    # Root workspace config
└── .env                           # Environment variables
```

### Dual Backend System

**Backend (FastAPI)** - Fallback Processing
- Traditional image processing
- Style-based transformations
- Works without AI models
- Fast and lightweight

**Colab Backend** - Advanced AI
- CNN-based doodle recognition
- ControlNet + Stable Diffusion
- 100+ category recognition
- Realistic transformations

---

## 🛠️ Tech Stack

### Frontend Technologies

| Technology | Version | Purpose |
|-----------|---------|---------|
| **React** | 18.2.0 | UI framework |
| **Vite** | 4.5.0 | Build tool & dev server |
| **Tailwind CSS** | 3.3.5 | Utility-first styling |
| **Framer Motion** | 10.16.4 | Smooth animations |
| **Lucide React** | 0.292.0 | Icon library |
| **HTML5 Canvas** | Native | Drawing interface |

### Backend Technologies

| Technology | Version | Purpose |
|-----------|---------|---------|
| **FastAPI** | 0.109.2 | Web framework |
| **Uvicorn** | 0.27.1 | ASGI server |
| **PyTorch** | 2.2.0 | Deep learning |
| **TensorFlow** | 2.13.0 | Model training |
| **Diffusers** | 0.26.3 | Stable Diffusion |
| **ControlNet** | Latest | Structure preservation |
| **Pillow** | 10.2.0 | Image processing |
| **OpenCV** | 4.9.0 | Computer vision |

### AI Models

| Model | Purpose | Size |
|-------|---------|------|
| **Custom CNN** | Doodle recognition | ~25 MB |
| **ControlNet Canny** | Edge-based control | ~1.5 GB |
| **Stable Diffusion 1.5** | Image generation | ~4 GB |

---

## 🎓 Train Your Own Model

### Why Train?

**Before Training:**
- ❌ Random weights
- ❌ ~10% accuracy
- ❌ Poor recognition

**After Training:**
- ✅ 100 categories
- ✅ ~87% accuracy
- ✅ ~96% top-3 accuracy
- ✅ Production ready

### 🚀 Quick Training (3 Steps)

#### Step 1: Upload to Kaggle
```
1. Go to https://www.kaggle.com
2. Create account (free)
3. Upload: training/kaggle_training.ipynb
4. Enable GPU: Settings → Accelerator → GPU T4 x2
```

#### Step 2: Train Model
```
1. Click "Run All"
2. Wait 2-4 hours (free GPU!)
3. Check accuracy (~87%)
```

#### Step 3: Integrate
```
1. Download from Output tab
2. Copy to: colab_backend/models/
3. Update: colab_backend/recognizer.py
4. Run: npm run dev
```

### 📊 Expected Results

```
Training Metrics:
├── Test Accuracy: 87.56%
├── Top-3 Accuracy: 96.23%
├── Inference Time: <50ms
├── Model Size: ~25 MB
└── Categories: 100

Supported Categories:
├── Vehicles: car, van, bus, bicycle, train, airplane, helicopter
├── Animals: cat, dog, bird, fish, horse, elephant, lion, tiger
├── Objects: house, tree, flower, book, chair, table, cup
└── 80+ more categories...
```

### 📚 Training Documentation

| Document | Purpose | Time |
|----------|---------|------|
| `training/QUICK_START.md` | 3-step guide | 5 min |
| `HOW_TO_TRAIN.md` | Complete guide | 15 min |
| `training/KAGGLE_GUIDE.md` | Kaggle details | 10 min |
| `INTEGRATION_GUIDE.md` | Integration help | 10 min |
| `TRAINING_INDEX.md` | Navigation | 2 min |

---

## 🎨 Usage Guide

### Basic Workflow

1. **Draw Your Sketch**
   - Use the interactive canvas
   - Choose brush size and color
   - Draw your idea freely

2. **Add Style Prompt**
   - Describe desired style
   - Example: "realistic car, vibrant colors"
   - System auto-recognizes doodle

3. **Generate Design**
   - Click "Create Magic"
   - AI recognizes + transforms
   - View realistic result

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `B` | Switch to brush |
| `E` | Switch to eraser |
| `[` / `]` | Decrease/increase brush size |
| `Ctrl+Z` | Undo |
| `Ctrl+S` | Download sketch |
| `Ctrl+C` | Clear canvas |

### Style Prompts Examples

```
Realistic:
- "realistic car, photorealistic, detailed"
- "realistic cat, lifelike, high quality"

Artistic:
- "watercolor painting, soft colors, artistic"
- "digital art, vibrant, modern style"

Minimalist:
- "minimalist line art, clean, simple"
- "geometric design, modern, abstract"

Vintage:
- "vintage poster, retro style, classic"
- "old photograph, sepia tone, nostalgic"
```

---

## 🐍 Python Version Compatibility

| Version | Status | Notes |
|---------|--------|-------|
| 3.8 | ✅ Supported | Minimum version |
| 3.9 | ✅ Supported | Stable |
| 3.10 | ✅ Supported | Stable |
| 3.11 | ✅ **Recommended** | Best performance |
| 3.12 | ⚠️ Limited | Some packages may have issues |

### Installation Options

```bash
# Auto-detect (recommended)
pip install -r backend/requirements.txt

# CPU-only (no GPU)
pip install -r backend/requirements-cpu.txt

# GPU with CUDA
pip install -r backend/requirements-gpu.txt
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` files in root, frontend, and backend directories:

```bash
# Root .env
USE_AI_MODELS=true
DEVICE=auto
HUGGINGFACE_TOKEN=your_token_here

# Frontend .env
VITE_API_URL=http://localhost:8000

# Backend .env
CORS_ALLOWED_ORIGINS=http://localhost:5173
CONTROLNET_MODEL=lllyasviel/sd-controlnet-canny
STABLE_DIFFUSION_MODEL=runwayml/stable-diffusion-v1-5
```

### AI Model Configuration

Edit `backend/config/ai_config.py`:

```python
DEFAULT_GENERATION_PARAMS = {
    "num_inference_steps": 20,      # Quality vs speed
    "guidance_scale": 7.5,           # Prompt adherence
    "controlnet_conditioning_scale": 1.0,  # Structure preservation
    "width": 512,
    "height": 512
}
```

---

## 📊 Performance

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **GPU VRAM** | N/A (CPU) | 6 GB+ |
| **Storage** | 5 GB | 15 GB |
| **CPU** | 4 cores | 8 cores |

### Benchmarks

```
Recognition Performance:
├── Inference Time: <50ms
├── Accuracy: 87.56%
├── Top-3 Accuracy: 96.23%
└── Model Load Time: ~2s

Generation Performance (GPU):
├── ControlNet: ~3-5s
├── Stable Diffusion: ~5-10s
└── Total: ~8-15s per image

Generation Performance (CPU):
├── Fallback Processing: ~1-2s
└── No AI models required
```

---

## 🧪 Testing

### Run Tests

```bash
# Test backend
cd backend
python -m pytest

# Test model loading
cd colab_backend
python load_trained_model.py

# Test API
curl http://localhost:8000/health
```

### Manual Testing

1. Draw a simple car
2. Enter prompt: "realistic car"
3. Check console for recognition
4. Verify generated image

---

## 📚 Documentation

### Quick Links

- **[Quick Start](training/QUICK_START.md)** - Get started in 5 minutes
- **[Training Guide](HOW_TO_TRAIN.md)** - Complete training instructions
- **[Integration Guide](INTEGRATION_GUIDE.md)** - Model integration
- **[Project Overview](PROJECT_OVERVIEW.md)** - Detailed project info
- **[Training Index](TRAINING_INDEX.md)** - Documentation navigation

### API Documentation

Access interactive API docs at: http://localhost:8000/docs

Key endpoints:
- `POST /generate-design` - Generate design from sketch
- `GET /health` - Health check
- `GET /supported-labels` - List recognized categories
- `GET /styles` - Available style transformations

---

## 🚀 Deployment

### Production Build

```bash
# Build frontend
cd frontend
npm run build

# Serve with production server
npm run preview
```

### Docker Deployment (Coming Soon)

```bash
docker-compose up -d
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add more doodle categories
- [ ] Improve recognition accuracy
- [ ] Add more artistic styles
- [ ] Optimize inference speed
- [ ] Mobile app version
- [ ] Cloud deployment guides

---

## 📝 License

MIT License - feel free to use for personal or commercial projects.

---

## 🙏 Acknowledgments

- **Quick Draw Dataset** - Google's doodle dataset
- **Hugging Face** - Model hosting and diffusers library
- **Kaggle** - Free GPU for training
- **ControlNet** - Structure-preserving AI
- **Stable Diffusion** - Image generation

---

## 👨‍💻 Author

**Rohith Cherukuri**

- GitHub: [@RohithCherukuri816](https://github.com/RohithCherukuri816)
- Project: [AI-Powered Creative Intelligence Platform](https://github.com/RohithCherukuri816/AI-Powered-Creative-Intelligence-Platform)

---

## 📞 Support

Having issues? Check these resources:

1. **Documentation** - Read the guides in this repo
2. **Console Logs** - Check browser and terminal output
3. **Troubleshooting** - See HOW_TO_TRAIN.md
4. **GitHub Issues** - Report bugs or request features

---

## 🎉 What's Next?

1. **Train your model** - Follow `training/QUICK_START.md`
2. **Experiment with styles** - Try different prompts
3. **Customize UI** - Edit Tailwind config
4. **Add features** - Extend the codebase
5. **Deploy** - Share with the world!

---

<div align="center">

**Made with ❤️ by Rohith Cherukuri**

⭐ Star this repo if you find it useful!

[⬆ Back to Top](#-sketchcraft---ai-powered-creative-intelligence-platform)

</div>