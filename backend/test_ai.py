#!/usr/bin/env python3
"""
Test script to verify AI models are working
"""

def test_pytorch():
    """Test PyTorch installation"""
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
        return True
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False

def test_diffusers():
    """Test Diffusers library"""
    try:
        import diffusers
        print(f"✅ Diffusers version: {diffusers.__version__}")
        return True
    except Exception as e:
        print(f"❌ Diffusers error: {e}")
        return False

def test_controlnet_aux():
    """Test ControlNet auxiliary library"""
    try:
        import controlnet_aux
        print(f"✅ ControlNet-aux available")
        return True
    except Exception as e:
        print(f"❌ ControlNet-aux error: {e}")
        return False

def test_model_loading():
    """Test loading a small model"""
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        
        print("🔄 Testing model loading (this may take a moment)...")
        
        # Try to load a small model
        model_id = "hf-internal-testing/tiny-stable-diffusion-torch"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe.to("cpu")
        
        print("✅ Model loading successful!")
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing AI Model Setup")
    print("=" * 40)
    
    tests = [
        ("PyTorch", test_pytorch),
        ("Diffusers", test_diffusers),
        ("ControlNet-aux", test_controlnet_aux),
        ("Model Loading", test_model_loading),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n🔍 Testing {name}...")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed! AI models are ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please install Visual C++ Redistributable:")
        print("   https://aka.ms/vs/16/release/vc_redist.x64.exe")