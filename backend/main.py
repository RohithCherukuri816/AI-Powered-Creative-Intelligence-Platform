from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import logging
from dotenv import load_dotenv

from routes.design import router as design_router

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check AI model configuration
USE_AI_MODELS = os.getenv("USE_AI_MODELS", "false").lower() == "true"
if USE_AI_MODELS:
    logger.info("AI models enabled - ControlNet + Stable Diffusion will be used")
else:
    logger.info("AI models disabled - using fallback image processing")

# Create FastAPI app
app = FastAPI(
    title="AI Creative Platform API",
    description="Transform doodles into stunning designs with AI",
    version="1.0.0"
)

# Configure CORS (allow extra origins via env CORS_ALLOWED_ORIGINS, comma-separated)
default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]
extra_origins = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if os.getenv("CORS_ALLOWED_ORIGINS") else []
allowed_origins = [o.strip() for o in (default_origins + extra_origins) if o and o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include routers
app.include_router(design_router)

@app.get("/")
async def root():
    return {
        "message": "🎨 AI Creative Platform API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    from routes.design import get_image_processor
    
    # Get processor status without triggering model load
    processor = get_image_processor()
    models_loaded = processor.are_models_loaded()
    
    return {
        "status": "ok",
        "message": "AI Creative Platform is running smoothly! ✨",
        "ai_models_loaded": models_loaded,
        "device": processor.device,
        "ai_enabled": processor.config["use_ai_models"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )