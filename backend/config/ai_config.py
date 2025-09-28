"""
AI Model Configuration for ControlNet + Stable Diffusion
"""

import os
from typing import Dict, Any

# Model configurations
CONTROLNET_MODELS = {
    "canny": {
        "model_id": "lllyasviel/sd-controlnet-canny",
        "description": "Edge-based control using Canny edge detection"
    },
    "lineart": {
        "model_id": "lllyasviel/sd-controlnet-mlsd", 
        "description": "Line art control for sketches and drawings"
    },
    "scribble": {
        "model_id": "lllyasviel/sd-controlnet-scribble",
        "description": "Scribble and rough sketch control"
    }
}

STABLE_DIFFUSION_MODELS = {
    "v1.5": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "description": "Standard Stable Diffusion v1.5"
    },
    "dreamlike": {
        "model_id": "dreamlike-art/dreamlike-diffusion-1.0",
        "description": "Dreamlike artistic style"
    }
}

# Generation parameters
DEFAULT_GENERATION_PARAMS = {
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "controlnet_conditioning_scale": 1.0,
    "width": 512,
    "height": 512
}

# Quality enhancement prompts
QUALITY_ENHANCERS = [
    "high quality",
    "detailed", 
    "beautiful",
    "masterpiece",
    "best quality",
    "sharp focus"
]

NEGATIVE_PROMPTS = [
    "low quality",
    "blurry",
    "ugly", 
    "bad anatomy",
    "worst quality",
    "low resolution"
]

# Style-specific enhancements
STYLE_ENHANCEMENTS = {
    "watercolor": [
        "watercolor painting",
        "soft brushstrokes", 
        "artistic",
        "flowing colors",
        "paper texture"
    ],
    "digital_art": [
        "digital art",
        "vibrant colors",
        "modern",
        "crisp details",
        "digital painting"
    ],
    "anime": [
        "anime style",
        "manga",
        "japanese art",
        "cel shading",
        "vibrant"
    ],
    "realistic": [
        "photorealistic",
        "realistic",
        "detailed",
        "lifelike",
        "high resolution"
    ],
    "minimalist": [
        "minimalist",
        "clean lines",
        "simple",
        "geometric",
        "modern design"
    ]
}

def get_ai_config() -> Dict[str, Any]:
    """Get AI configuration from environment variables"""
    return {
        "use_ai_models": os.getenv("USE_AI_MODELS", "false").lower() == "true",
        "device": os.getenv("DEVICE", "auto"),
        "controlnet_model": os.getenv("CONTROLNET_MODEL", "lllyasviel/sd-controlnet-canny"),
        "stable_diffusion_model": os.getenv("STABLE_DIFFUSION_MODEL", "runwayml/stable-diffusion-v1-5"),
        "huggingface_token": os.getenv("HUGGINGFACE_TOKEN"),
        "generation_params": DEFAULT_GENERATION_PARAMS
    }