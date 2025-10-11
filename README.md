<div align="center">

# 🎨 AI-Powered Creative Intelligence Platform

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=6366F1&center=true&vCenter=true&width=600&lines=Transform+Doodles+into+Art;AI-Powered+Creativity;Sketch+%E2%86%92+Prompt+%E2%86%92+Generate" alt="Typing SVG" />

<p align="center">
  <img src="https://img.shields.io/badge/AI-Powered-6366F1?style=for-the-badge&logo=brain&logoColor=white" alt="AI Powered" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=white" alt="React" />
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/yourusername/ai-creative-platform?style=social" alt="GitHub stars" />
  <img src="https://img.shields.io/github/forks/yourusername/ai-creative-platform?style=social" alt="GitHub forks" />
  <img src="https://img.shields.io/github/watchers/yourusername/ai-creative-platform?style=social" alt="GitHub watchers" />
</p>

---

### 🌟 Transform your sketches into stunning artwork with the power of AI

</div>

<div align="center">

## 🚀 Quick Start Guide

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700">

</div>

> 💡 **Pro Tip**: Use our automated setup scripts for the fastest installation experience!

### 📋 Prerequisites

<table>
<tr>
<td align="center"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/nodejs/nodejs-original.svg" width="40" height="40"/><br><b>Node.js 18+</b></td>
<td align="center"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="40" height="40"/><br><b>Python 3.11</b></td>
<td align="center"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/npm/npm-original-wordmark.svg" width="40" height="40"/><br><b>npm/pnpm</b></td>
<td align="center"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/nvidia/nvidia-original.svg" width="40" height="40"/><br><b>CUDA 11.8+</b><br><i>(optional)</i></td>
</tr>
</table>

### ⚡ Automated Setup (Recommended)

<div align="center">

| Platform | Command | Status |
|----------|---------|--------|
| 🪟 **Windows** | `scripts/setup.bat` | ![Windows](https://img.shields.io/badge/Windows-0078D6?style=flat-square&logo=windows&logoColor=white) |
| 🐧 **Linux** | `chmod +x scripts/setup.sh && ./scripts/setup.sh` | ![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black) |
| 🍎 **macOS** | `chmod +x scripts/setup.sh && ./scripts/setup.sh` | ![macOS](https://img.shields.io/badge/macOS-000000?style=flat-square&logo=apple&logoColor=white) |

</div>

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

<div align="center">

## 🤖 AI Models & Architecture

<img src="https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif" width="100">

</div>

<table align="center">
<tr>
<th>🎯 ControlNet</th>
<th>🎨 Stable Diffusion</th>
<th>🔄 Fallback Mode</th>
</tr>
<tr>
<td align="center">
<img src="https://img.shields.io/badge/Structure-Preservation-FF6B6B?style=for-the-badge" /><br>
Maintains your sketch's structure and composition
</td>
<td align="center">
<img src="https://img.shields.io/badge/Style-Generation-4ECDC4?style=for-the-badge" /><br>
Applies artistic styles from text prompts
</td>
<td align="center">
<img src="https://img.shields.io/badge/Traditional-Processing-45B7D1?style=for-the-badge" /><br>
Works without AI models installed
</td>
</tr>
</table>

### 💻 System Requirements

<div align="center">

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| 🧠 **RAM** | 8GB | 16GB | 32GB+ |
| 🎮 **GPU** | CPU Only | GTX 1060 6GB | RTX 3080+ |
| 💾 **Storage** | 5GB | 10GB | 20GB+ |
| ⚡ **VRAM** | N/A | 6GB | 12GB+ |

</div>

<div align="center">

## 🐍 Python Version Compatibility

<img src="https://user-images.githubusercontent.com/74038190/212257467-871d32b7-e401-42e8-a166-fcfd7baa4c6b.gif" width="100">

</div>

<div align="center">

| Python Version | Status | Performance | Notes |
|----------------|--------|-------------|-------|
| ![Python 3.8](https://img.shields.io/badge/3.8-✅_Supported-green?style=flat-square&logo=python) | ✅ | ⭐⭐⭐ | Minimum version |
| ![Python 3.9](https://img.shields.io/badge/3.9-✅_Supported-green?style=flat-square&logo=python) | ✅ | ⭐⭐⭐⭐ | Stable |
| ![Python 3.10](https://img.shields.io/badge/3.10-✅_Supported-green?style=flat-square&logo=python) | ✅ | ⭐⭐⭐⭐ | Stable |
| ![Python 3.11](https://img.shields.io/badge/3.11-🌟_Recommended-gold?style=flat-square&logo=python) | 🌟 | ⭐⭐⭐⭐⭐ | **Best performance** |
| ![Python 3.12](https://img.shields.io/badge/3.12-⚠️_Limited-orange?style=flat-square&logo=python) | ⚠️ | ⭐⭐⭐ | Some packages may have issues |

</div>

### 📦 Installation Options

<div align="center">

| Setup Type | Command | Use Case |
|------------|---------|----------|
| 🔄 **Auto-detect** | `pip install -r requirements.txt` | Let the system choose optimal packages |
| 💻 **CPU Only** | `pip install -r requirements-cpu.txt` | No GPU available |
| 🚀 **GPU Accelerated** | `pip install -r requirements-gpu.txt` | NVIDIA GPU with CUDA |
| 🏠 **Virtual Environment** | `python -m venv venv` | Isolated environment (recommended) |

</div>

<div align="center">

## 🏗️ Project Architecture

<img src="https://user-images.githubusercontent.com/74038190/212284087-bbe7e430-757e-4901-90bf-4cd2ce3e1852.gif" width="100">

</div>

```
🎨 ai-creative-platform/
├── 🖥️  frontend/              # React + Vite + Tailwind CSS
│   ├── 📱 src/components/     # Reusable UI components
│   ├── 🎨 src/styles/         # Global styles & themes
│   └── ⚙️  vite.config.js     # Build configuration
├── 🔧 backend/               # FastAPI + Python AI Engine
│   ├── 🤖 models/            # AI model configurations
│   ├── 🛠️  api/               # REST API endpoints
│   └── 🧠 services/          # Core business logic
├── 🌍 .env                   # Environment variables
├── 📦 package.json           # Workspace configuration
└── 📖 README.md              # You are here!
```

<div align="center">

### 🔗 Service Communication

```mermaid
graph LR
    A[🎨 Frontend<br/>React + Vite] --> B[🔧 Backend<br/>FastAPI]
    B --> C[🤖 AI Models<br/>ControlNet + SD]
    B --> D[🖼️ Image Processing<br/>PIL + OpenCV]
    C --> E[🎯 Generated Art]
    D --> E
```

</div>

<div align="center">

## ✨ Features & Capabilities

<img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="400">

</div>

<table>
<tr>
<td align="center" width="33%">
<img src="https://user-images.githubusercontent.com/74038190/212257472-08e52665-c503-4bd9-aa20-f5a4dae769b5.gif" width="100"><br>
<h3>🎨 Interactive Canvas</h3>
<p>Sketch with customizable brushes, colors, and tools. Intuitive drawing experience with undo/redo support.</p>
</td>
<td align="center" width="33%">
<img src="https://user-images.githubusercontent.com/74038190/212257468-1e9a91f1-b626-4baa-b15d-5c385dfa7763.gif" width="100"><br>
<h3>🤖 AI Transformation</h3>
<p>ControlNet + Stable Diffusion preserve your sketch structure while applying stunning artistic styles.</p>
</td>
<td align="center" width="33%">
<img src="https://user-images.githubusercontent.com/74038190/212257465-7ce8d493-cac5-494e-982a-5a9deb852c4b.gif" width="100"><br>
<h3>⚡ Real-time Generation</h3>
<p>Transform doodles into professional artwork in seconds with AI-powered processing.</p>
</td>
</tr>
<tr>
<td align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257460-738ff738-247f-4445-a718-cdd0ca76e2db.gif" width="100"><br>
<h3>💬 Style Prompts</h3>
<p>Describe your vision with natural language. "Make it cyberpunk", "Add watercolor effects", etc.</p>
</td>
<td align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257463-4d082cb4-7483-4eaf-bc25-6dde2628aabd.gif" width="100"><br>
<h3>🎭 Beautiful UI</h3>
<p>Modern pastel interface with smooth animations and responsive design across all devices.</p>
</td>
<td align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257469-7e8c204f-c544-41f8-a292-85c262fcf4bd.gif" width="100"><br>
<h3>🔄 Fallback Mode</h3>
<p>Traditional image processing ensures functionality even without AI models installed.</p>
</td>
</tr>
</table>

<div align="center">

## 🛠️ Technology Stack

<img src="https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif" width="100">

</div>

<table align="center">
<tr>
<th>🖥️ Frontend</th>
<th>🔧 Backend</th>
<th>🤖 AI/ML</th>
</tr>
<tr>
<td align="center">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/react/react-original.svg" width="40" height="40"/><br>
<b>React 18</b><br>
<img src="https://img.shields.io/badge/Vite-646CFF?style=flat-square&logo=vite&logoColor=white" /><br>
<img src="https://img.shields.io/badge/Tailwind-38B2AC?style=flat-square&logo=tailwind-css&logoColor=white" /><br>
<img src="https://img.shields.io/badge/Framer_Motion-0055FF?style=flat-square&logo=framer&logoColor=white" /><br>
<img src="https://img.shields.io/badge/HTML5_Canvas-E34F26?style=flat-square&logo=html5&logoColor=white" />
</td>
<td align="center">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/fastapi/fastapi-original.svg" width="40" height="40"/><br>
<b>FastAPI</b><br>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" /><br>
<img src="https://img.shields.io/badge/Pillow-3776AB?style=flat-square&logo=python&logoColor=white" /><br>
<img src="https://img.shields.io/badge/CORS-Enabled-00D4AA?style=flat-square" /><br>
<img src="https://img.shields.io/badge/Async-Support-FF6B6B?style=flat-square" />
</td>
<td align="center">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pytorch/pytorch-original.svg" width="40" height="40"/><br>
<b>PyTorch</b><br>
<img src="https://img.shields.io/badge/ControlNet-FF9500?style=flat-square&logo=ai&logoColor=white" /><br>
<img src="https://img.shields.io/badge/Stable_Diffusion-7C3AED?style=flat-square&logo=ai&logoColor=white" /><br>
<img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black" /><br>
<img src="https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white" />
</td>
</tr>
</table>

<div align="center">

---

## 🤝 Contributing

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif" width="100">

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

<table>
<tr>
<td align="center">
<img src="https://img.shields.io/badge/Issues-Welcome-brightgreen?style=for-the-badge&logo=github" /><br>
<b>Report Bugs</b>
</td>
<td align="center">
<img src="https://img.shields.io/badge/PRs-Welcome-blue?style=for-the-badge&logo=git" /><br>
<b>Submit PRs</b>
</td>
<td align="center">
<img src="https://img.shields.io/badge/Ideas-Welcome-purple?style=for-the-badge&logo=lightbulb" /><br>
<b>Share Ideas</b>
</td>
</tr>
</table>

## 📊 Project Stats

<div align="center">
<img src="https://github-readme-stats.vercel.app/api?username=yourusername&repo=ai-creative-platform&show_icons=true&theme=radical" alt="GitHub Stats" />
</div>

## 📄 License

<img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="MIT License" />

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### 💜 Made with Love by **Rohith Cherukuri**

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="500">



**⭐ Star this repo if you found it helpful!**

</div>

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer" />
</div>
