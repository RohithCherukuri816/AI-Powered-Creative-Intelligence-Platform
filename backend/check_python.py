#!/usr/bin/env python3
"""
Python Version and Compatibility Checker for AI Creative Platform
Run this script to check if your Python environment is compatible.
"""

import sys
import subprocess
import importlib.util

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro} detected")
    
    if version < (3, 8):
        print("❌ ERROR: Python 3.8+ is required")
        print("Please upgrade your Python version")
        return False
    elif version >= (3, 12):
        print("⚠️  WARNING: Python 3.12+ detected")
        print("Some AI packages may not be fully compatible")
        print("Consider using Python 3.11 for best compatibility")
    elif version >= (3, 11):
        print("✅ EXCELLENT: Python 3.11+ is recommended")
    else:
        print("✅ GOOD: Python version is compatible")
    
    return True

def check_pip():
    """Check if pip is available and up to date"""
    try:
        import pip
        print("✅ pip is available")
        
        # Check pip version
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"📦 {result.stdout.strip()}")
        
        return True
    except ImportError:
        print("❌ pip is not available")
        print("Please install pip: https://pip.pypa.io/en/stable/installation/")
        return False

def check_gpu_support():
    """Check for NVIDIA GPU and CUDA support"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("🎮 NVIDIA GPU detected")
            print("You can use the GPU version for faster AI processing")
            return True
    except FileNotFoundError:
        pass
    
    print("💻 No NVIDIA GPU detected (CPU-only mode)")
    print("AI processing will be slower but still functional")
    return False

def check_virtual_environment():
    """Check if running in virtual environment"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
        return True
    else:
        print("⚠️  Not in virtual environment")
        print("Recommendation: Create a virtual environment")
        print("  python -m venv venv")
        print("  source venv/bin/activate  # Linux/Mac")
        print("  venv\\Scripts\\activate     # Windows")
        return False

def recommend_requirements():
    """Recommend which requirements file to use"""
    has_gpu = check_gpu_support()
    
    print("\n📋 Recommended installation:")
    
    if has_gpu:
        print("🚀 For GPU acceleration:")
        print("  pip install -r requirements-gpu.txt")
        print("\n💻 For CPU-only (slower):")
        print("  pip install -r requirements-cpu.txt")
    else:
        print("💻 For your system:")
        print("  pip install -r requirements-cpu.txt")
        print("\n🚀 If you get a GPU later:")
        print("  pip install -r requirements-gpu.txt")
    
    print("\n🔄 Auto-detect (recommended):")
    print("  pip install -r requirements.txt")

def main():
    print("🎨 AI Creative Platform - Python Compatibility Check")
    print("=" * 55)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    print()
    
    # Check pip
    check_pip()
    
    print()
    
    # Check virtual environment
    check_virtual_environment()
    
    print()
    
    # GPU check and recommendations
    recommend_requirements()
    
    print("\n" + "=" * 55)
    print("✅ Compatibility check complete!")
    print("Run the recommended pip install command above to proceed.")

if __name__ == "__main__":
    main()