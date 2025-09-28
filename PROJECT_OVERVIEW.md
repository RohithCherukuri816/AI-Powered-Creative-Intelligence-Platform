# 🎨 AI-Powered Creative Intelligence Platform

## Project Structure

```
ai-creative-platform/
├── 📁 frontend/                 # React + Vite + Tailwind CSS
│   ├── src/
│   │   ├── components/
│   │   │   ├── DrawingCanvas.jsx      # Interactive HTML5 canvas
│   │   │   ├── PromptInput.jsx        # Style prompt input with suggestions
│   │   │   ├── GeneratedImageCard.jsx # Display results with actions
│   │   │   └── LoadingSpinner.jsx     # Beautiful loading animation
│   │   ├── App.jsx                    # Main app with hero section
│   │   ├── api.js                     # Backend API calls
│   │   ├── index.css                  # Tailwind + custom styles
│   │   └── main.jsx                   # React entry point
│   ├── package.json
│   ├── tailwind.config.js             # Pastel theme configuration
│   └── vite.config.js
├── 📁 backend/                  # FastAPI + Python
│   ├── routes/
│   │   └── design.py                  # Main API endpoints
│   ├── utils/
│   │   └── image_processor.py         # AI-style image transformations
│   ├── main.py                        # FastAPI app entry point
│   └── requirements.txt
├── 📁 scripts/                  # Setup automation
│   ├── setup.sh                       # Unix setup script
│   └── setup.bat                      # Windows setup script
├── dev.js                             # Development server launcher
├── package.json                       # Root workspace configuration
├── pnpm-workspace.yaml               # PNPM workspace config
├── .env                              # Shared environment variables
└── README.md
```

## 🎨 Design Features

### Frontend (React + Tailwind + Framer Motion)
- **Pastel Theme**: Lavender, soft pinks, light blues
- **Interactive Canvas**: HTML5 canvas with brush tools, colors, undo/redo
- **Smooth Animations**: Framer Motion for hover effects, page transitions
- **Responsive Design**: Works on desktop and mobile
- **Modern UI**: Glass morphism cards, gradient buttons, floating elements

### Backend (FastAPI + Python)
- **RESTful API**: Clean endpoints for design generation
- **Image Processing**: PIL-based artistic transformations
- **Style Detection**: Prompt analysis for different art styles
- **File Management**: Automatic image saving and serving
- **CORS Enabled**: Ready for frontend integration

## 🚀 Key Features

1. **Drawing Canvas**
   - Multiple brush sizes and colors
   - Eraser tool
   - Undo/redo functionality
   - Clear canvas option
   - Real-time sketch capture

2. **Style Prompts**
   - Text input for style descriptions
   - Predefined style suggestions
   - Keyword-based style detection

3. **AI Transformations** (Mock Implementation)
   - Watercolor effect
   - Digital art enhancement
   - Minimalist line art
   - Vintage poster style
   - Geometric patterns

4. **Result Display**
   - High-quality image preview
   - Download functionality
   - Share options
   - Try again feature

## 🛠️ Tech Stack

**Frontend:**
- React 18 + Vite
- Tailwind CSS (custom pastel theme)
- Framer Motion (animations)
- Lucide React (icons)
- HTML5 Canvas API

**Backend:**
- FastAPI (Python web framework)
- PIL/Pillow (image processing)
- Uvicorn (ASGI server)
- Pydantic (data validation)

**Development:**
- PNPM workspaces
- Concurrent development servers
- Hot reload for both frontend and backend
- Environment variable management

## 🎯 Ready for AI Integration

The backend is structured to easily integrate with real AI models:

- **OpenAI DALL-E**: Replace `image_processor.py` with DALL-E API calls
- **Stability AI**: Integrate Stable Diffusion for style transfer
- **Custom Models**: Add your own AI model endpoints
- **Multiple Providers**: Support different AI services

## 🎨 Customization

**Colors**: Edit `tailwind.config.js` to change the pastel theme
**Styles**: Modify `image_processor.py` to add new artistic effects
**UI**: Update components in `frontend/src/components/`
**API**: Extend `backend/routes/design.py` for new features

## 📱 Mobile Ready

The interface is fully responsive and works great on:
- Desktop computers
- Tablets
- Mobile phones
- Touch devices (canvas drawing works with touch)

---

**Made by Rohith Cherukuri** 💜

This platform provides a solid foundation for building AI-powered creative tools with a beautiful, modern interface and scalable backend architecture.