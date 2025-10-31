#!/usr/bin/env python3
"""
Test script to verify model loading optimization
"""

import time
from routes.design import get_image_processor

def test_model_loading():
    print("🧪 Testing ImageProcessor model loading optimization...")
    
    # First call - should load models
    print("\n1️⃣ First processor instance:")
    start_time = time.time()
    processor1 = get_image_processor()
    print(f"   Instance created in {time.time() - start_time:.2f}s")
    print(f"   Models loaded: {processor1.are_models_loaded()}")
    
    # Second call - should reuse existing instance
    print("\n2️⃣ Second processor instance:")
    start_time = time.time()
    processor2 = get_image_processor()
    print(f"   Instance created in {time.time() - start_time:.2f}s")
    print(f"   Models loaded: {processor2.are_models_loaded()}")
    print(f"   Same instance: {processor1 is processor2}")
    
    # Third call - should still reuse
    print("\n3️⃣ Third processor instance:")
    start_time = time.time()
    processor3 = get_image_processor()
    print(f"   Instance created in {time.time() - start_time:.2f}s")
    print(f"   Models loaded: {processor3.are_models_loaded()}")
    print(f"   Same instance: {processor1 is processor3}")
    
    print("\n✅ Test completed!")
    print(f"💡 All instances are the same object: {processor1 is processor2 is processor3}")

if __name__ == "__main__":
    test_model_loading()