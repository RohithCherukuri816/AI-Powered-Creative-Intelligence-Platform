"""
Colab Backend for AI-Powered Doodle Recognition Platform

This package provides a complete backend implementation optimized for Google Colab,
featuring MobileNet-based doodle recognition and Stable Diffusion image generation.

Components:
- MobileNetDoodleRecognizer: Efficient doodle recognition using MobileNetV2/V3
- MobileNetTrainer: Training pipeline for custom models
- ImageProcessor: Stable Diffusion integration with ControlNet
- HuggingFaceIntegration: Upload and manage models on Hugging Face Hub
- FastAPI App: Complete REST API server

Usage:
    from colab_backend.setup_colab import ColabSetup
    
    setup = ColabSetup()
    results = setup.complete_setup()
"""

__version__ = "1.0.0"
__author__ = "Rohith Cherukuri"

# Import main components
try:
    from .mobilenet_recognizer import MobileNetDoodleRecognizer
    from .mobilenet_trainer import MobileNetTrainer
    from .processor import ImageProcessor
    from .huggingface_integration import HuggingFaceIntegration
    from .app import create_app
    from .setup_colab import ColabSetup
    
    __all__ = [
        "MobileNetDoodleRecognizer",
        "MobileNetTrainer", 
        "ImageProcessor",
        "HuggingFaceIntegration",
        "create_app",
        "ColabSetup"
    ]
    
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Some components could not be imported: {e}")
    __all__ = []