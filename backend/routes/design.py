from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import base64
import io
import os
import uuid
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import random
from datetime import datetime
import time

from utils.image_processor import ImageProcessor

# Create a global instance to avoid reloading models
_image_processor = None

def get_image_processor():
    """Get the singleton ImageProcessor instance"""
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor()
    return _image_processor

router = APIRouter()

class DesignRequest(BaseModel):
    prompt: str
    sketch: str  # base64 encoded image

class DesignResponse(BaseModel):
    image_url: str
    generation_id: str
    prompt_used: str
    processing_time: float
    recognized_label: str | None = None

@router.post("/generate-design", response_model=DesignResponse)
async def generate_design(request: DesignRequest):
    """
    Generate a design from a sketch and style prompt.
    This is a stub implementation that applies artistic filters to the sketch.
    In production, this would integrate with AI models like DALL-E, Midjourney, or Stable Diffusion.
    """
    try:
        start_time = datetime.now()
        
        # Validate input
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if not request.sketch:
            raise HTTPException(status_code=400, detail="Sketch data is required")
        
        # Process the sketch (use singleton to avoid reloading models)
        processor_start = time.time()
        processor = get_image_processor()
        processor_time = time.time() - processor_start
        print(f"⏱️ Processor instance obtained in {processor_time:.3f}s (models loaded: {processor.are_models_loaded()})")
        
        # Decode base64 image
        try:
            # Remove data URL prefix if present (do not mutate request model)
            sketch_b64 = request.sketch.split(',')[1] if request.sketch.startswith('data:image') else request.sketch
            
            image_data = base64.b64decode(sketch_b64)
            sketch_image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        # Validate that the sketch contains actual drawing (not blank canvas)
        try:
            img_rgb = sketch_image.convert('RGB')
            img_arr = np.array(img_rgb)
            # Count pixels that are not almost white
            non_white = np.sum((img_arr[:, :, 0] < 240) | (img_arr[:, :, 1] < 240) | (img_arr[:, :, 2] < 240))
            total = img_arr.shape[0] * img_arr.shape[1]
            drawing_ratio = non_white / max(1, total)
            if drawing_ratio < 0.005:  # <0.5% non-white pixels → treat as blank
                raise HTTPException(status_code=400, detail="No drawing detected in sketch. Please draw something first.")
        except HTTPException:
            raise
        except Exception:
            # If detection fails, proceed without blocking
            pass

        # Generate unique ID for this generation
        generation_id = str(uuid.uuid4())
        
        # Apply AI-style transformations based on prompt
        processed_image = processor.apply_style_transformation(
            sketch_image, 
            request.prompt,
            generation_id
        )
        
        # Save the processed image
        filename = f"generated_{generation_id}.png"
        filepath = os.path.join("uploads", filename)
        processed_image.save(filepath, "PNG", quality=95)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Recognize doodle label (simple keyword-based heuristic)
        recognized_label = None
        try:
            keywords = [
                "van", "car", "bus", "bicycle", "bike", "motorcycle", "truck",
                "cat", "dog", "bird", "fish", "horse",
                "house", "tree", "flower", "boat", "airplane", "face", "person"
            ]
            prompt_lower = request.prompt.lower()
            for k in keywords:
                if k in prompt_lower:
                    recognized_label = k
                    break
        except Exception:
            recognized_label = None

        # Return the result
        image_url = f"/uploads/{filename}"
        
        return DesignResponse(
            image_url=image_url,
            generation_id=generation_id,
            prompt_used=request.prompt,
            processing_time=processing_time,
            recognized_label=recognized_label
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.get("/generations/{generation_id}")
async def get_generation(generation_id: str):
    """Get details about a specific generation"""
    filepath = os.path.join("uploads", f"generated_{generation_id}.png")
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Generation not found")
    
    return {
        "generation_id": generation_id,
        "image_url": f"/uploads/generated_{generation_id}.png",
        "created_at": datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
    }

@router.delete("/generations/{generation_id}")
async def delete_generation(generation_id: str):
    """Delete a generated image"""
    filepath = os.path.join("uploads", f"generated_{generation_id}.png")
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Generation not found")
    
    try:
        os.remove(filepath)
        return {"message": "Generation deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete generation: {str(e)}")

@router.get("/styles")
async def get_available_styles():
    """Get list of available style transformations"""
    return {
        "styles": [
            {
                "name": "watercolor",
                "description": "Soft watercolor painting effect",
                "keywords": ["watercolor", "painting", "soft", "artistic"]
            },
            {
                "name": "digital_art",
                "description": "Vibrant digital art style",
                "keywords": ["digital", "vibrant", "modern", "colorful"]
            },
            {
                "name": "minimalist",
                "description": "Clean minimalist line art",
                "keywords": ["minimalist", "clean", "simple", "line art"]
            },
            {
                "name": "vintage",
                "description": "Retro vintage poster style",
                "keywords": ["vintage", "retro", "poster", "classic"]
            },
            {
                "name": "geometric",
                "description": "Modern geometric patterns",
                "keywords": ["geometric", "modern", "patterns", "abstract"]
            }
        ]
    }