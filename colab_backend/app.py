import os
import base64
import io
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from PIL import Image

from .processor import ImageProcessor
from .mobilenet_recognizer import MobileNetDoodleRecognizer


class DesignRequest(BaseModel):
    prompt: str
    sketch: str  # base64 data URL or raw base64


class DesignResponse(BaseModel):
    image_url: str
    generation_id: str
    prompt_used: str
    processing_time: float
    recognized_label: str | None = None
    confidence: float | None = None
    recognition_success: bool = False


def create_app() -> FastAPI:
    app = FastAPI(title="Smart Doodle to Realistic Generator", version="5.0.0")

    # CORS configuration
    default_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    extra_origins = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if os.getenv("CORS_ALLOWED_ORIGINS") else []
    allowed_origins = [o.strip() for o in (default_origins + extra_origins) if o and o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create directories
    os.makedirs("uploads", exist_ok=True)
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

    # Initialize components
    processor = ImageProcessor()
    recognizer = MobileNetDoodleRecognizer()

    @app.get("/")
    async def root():
        return {
            "message": "Smart Doodle to Realistic Generator", 
            "version": "5.0.0",
            "features": ["auto_doodle_recognition", "smart_prompt_integration", "doodle_to_realistic"],
            "mode": "AUTO_RECOGNITION_AND_TRANSFORMATION"
        }

    @app.get("/health")
    async def health():
        import torch
        return {
            "status": "ok", 
            "cuda_available": torch.cuda.is_available(),
            "mode": "auto_recognition_transformation"
        }

    @app.post("/generate-design", response_model=DesignResponse)
    async def generate_design(request: DesignRequest):
        start = datetime.now()

        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        if not request.sketch:
            raise HTTPException(status_code=400, detail="Sketch data is required")

        print(f"🎨 Received request - Prompt: '{request.prompt}'")

        # Decode base64 image
        try:
            sketch_b64 = request.sketch.split(',')[1] if request.sketch.startswith('data:image') else request.sketch
            image_data = base64.b64decode(sketch_b64)
            sketch_image = Image.open(io.BytesIO(image_data))
            print("✅ Sketch image decoded successfully")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

        generation_id = str(uuid.uuid4())
        recognized_label = None
        confidence = 0.0
        recognition_success = False

        # STEP 1: AUTOMATICALLY Recognize the doodle
        try:
            print("🔍 Auto-recognizing doodle...")
            recognized_label, confidence = recognizer.recognize_doodle(sketch_image)
            recognition_success = recognized_label is not None and confidence > 0.1
            
            if recognition_success:
                print(f"✅ Auto-recognized as: {recognized_label} (confidence: {confidence:.2f})")
            else:
                print(f"⚠️ Recognition failed or low confidence")
            
        except Exception as e:
            print(f"❌ Recognition failed: {e}")

        # STEP 2: Generate REALISTIC version using AUTO-RECOGNITION + USER PROMPT
        print("🚀 Transforming with auto-recognition + user prompt...")
        result_img = processor.apply_style_transformation(
            sketch_image, 
            request.prompt, 
            generation_id,
            recognized_label
        )

        # Save result
        filename = f"generated_{generation_id}.png"
        filepath = os.path.join("uploads", filename)
        result_img.save(filepath, "PNG", quality=95)

        processing_time = (datetime.now() - start).total_seconds()

        # Cache-bust URL
        image_url = f"/uploads/{filename}?t={int(datetime.now().timestamp())}"

        print(f"✅ Transformation completed in {processing_time:.2f}s")

        return DesignResponse(
            image_url=image_url,
            generation_id=generation_id,
            prompt_used=request.prompt,
            processing_time=processing_time,
            recognized_label=recognized_label,
            confidence=confidence,
            recognition_success=recognition_success,
        )
    
    @app.get("/supported-labels")
    async def get_supported_labels():
        """Return list of supported doodle labels"""
        return {
            "supported_labels": recognizer.classes,
            "total_categories": len(recognizer.classes),
            "message": "MobileNet-powered doodle recognition with Stable Diffusion generation"
        }
    
    @app.get("/model-info")
    async def get_model_info():
        """Return information about the current model"""
        return recognizer.get_model_info()

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("colab_backend.app:app", host="0.0.0.0", port=8000, reload=False)