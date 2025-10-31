#!/usr/bin/env python3
"""
Pre-load AI models to cache them for faster subsequent use
"""

import os
import sys
from utils.image_processor import ImageProcessor

def preload_models():
    """Pre-load and cache the AI models"""
    print("🚀 Pre-loading AI models...")
    print("This will download and cache models for faster subsequent use.")
    print("=" * 60)
    
    try:
        # Create processor instance
        processor = ImageProcessor()
        
        # Force load the models
        print("📥 Loading ControlNet and Stable Diffusion models...")
        success = processor._load_controlnet_models()
        
        if success:
            print("✅ Models loaded and cached successfully!")
            print("🎉 Future generations will be much faster!")
            print("\nModel cache location:")
            print("  ~/.cache/huggingface/hub/")
            print("\nYou can now start the server with:")
            print("  python main.py")
        else:
            print("❌ Failed to load models")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = preload_models()
    sys.exit(0 if success else 1)