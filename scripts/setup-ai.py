#!/usr/bin/env python3
"""
Setup script for AI models (ControlNet + Stable Diffusion)
This script will download and cache the required models.
"""

import os
import sys
import torch
from huggingface_hub import login
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector, LineartDetector

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'torch', 'diffusers', 'transformers', 'controlnet_aux', 
        'opencv-python', 'numpy', 'pillow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def setup_huggingface_token():
    """Setup Hugging Face token for model downloads"""
    token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not token or token == "your_huggingface_token_here":
        print("⚠️  Hugging Face token not found in environment variables")
        print("You can get a free token at: https://huggingface.co/settings/tokens")
        
        token = input("Enter your Hugging Face token (or press Enter to skip): ").strip()
        
        if token:
            try:
                login(token=token)
                print("✅ Hugging Face token configured successfully")
                return True
            except Exception as e:
                print(f"❌ Failed to login with token: {e}")
                return False
        else:
            print("⚠️  Skipping Hugging Face login - some models may not be accessible")
            return False
    else:
        try:
            login(token=token)
            print("✅ Hugging Face token loaded from environment")
            return True
        except Exception as e:
            print(f"❌ Failed to login with environment token: {e}")
            return False

def download_models():
    """Download and cache the AI models"""
    print("\n🤖 Downloading AI models...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Download ControlNet model
        print("📥 Downloading ControlNet Canny model...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        print("✅ ControlNet model downloaded")
        
        # Download Stable Diffusion pipeline
        print("📥 Downloading Stable Diffusion v1.5...")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        print("✅ Stable Diffusion model downloaded")
        
        # Download preprocessors
        print("📥 Downloading preprocessors...")
        canny_detector = CannyDetector()
        lineart_detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
        print("✅ Preprocessors downloaded")
        
        print("\n🎉 All models downloaded successfully!")
        print("💡 Models are cached and ready for use")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download models: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (~10GB)")
        print("3. Try running with a Hugging Face token")
        print("4. Check if you have CUDA installed for GPU support")
        return False

def test_models():
    """Test if models can be loaded successfully"""
    print("\n🧪 Testing model loading...")
    
    try:
        from utils.image_processor import ImageProcessor
        
        processor = ImageProcessor()
        print("✅ ImageProcessor initialized successfully")
        
        # Test model loading
        success = processor._load_controlnet_models()
        if success:
            print("✅ ControlNet models loaded successfully")
            print("🎉 Setup complete! Your AI models are ready to use.")
        else:
            print("⚠️  Models downloaded but failed to load - check logs for details")
            
    except Exception as e:
        print(f"❌ Failed to test models: {e}")

def main():
    print("🎨 AI Creative Platform - Model Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Setup Hugging Face token
    setup_huggingface_token()
    
    # Download models
    if download_models():
        test_models()
    else:
        print("\n⚠️  Model download failed, but the app will still work with fallback processing")
    
    print("\n🚀 You can now start the application with: npm run dev")

if __name__ == "__main__":
    main()