#!/usr/bin/env python3
"""
Quick training script for testing on a small dataset
"""

import os
import sys

# Override config to use quick settings
os.environ['USE_QUICK_CONFIG'] = 'True'

from config import USE_QUICK_CONFIG, QUICK_CONFIG
from download_dataset import main as download_main
from data_preprocessing import main as preprocess_main
from train_model import train_model
from evaluate_model import evaluate_model
from export_model import export_for_production

def quick_train_pipeline():
    """Complete quick training pipeline for testing"""
    
    print("⚡ Quick Training Pipeline")
    print("=" * 50)
    print("This will train a small model on 10 classes with 1000 samples each")
    print("Perfect for testing the training pipeline on a GPU machine")
    print()
    
    # Confirm
    response = input("Continue with quick training? (y/n): ").lower().strip()
    if response != 'y':
        print("Quick training cancelled.")
        return
    
    try:
        # Step 1: Download dataset
        print("\n🔄 Step 1: Downloading dataset...")
        download_main()
        
        # Step 2: Preprocess data
        print("\n🔄 Step 2: Preprocessing data...")
        preprocess_main()
        
        # Step 3: Train model
        print("\n🔄 Step 3: Training model...")
        model, history, info = train_model(architecture='simple')
        
        # Step 4: Evaluate model
        print("\n🔄 Step 4: Evaluating model...")
        model_path = os.path.join("models", "doodle_classifier_simple_final.h5")
        if os.path.exists(model_path):
            evaluate_model(model_path, create_visualizations=True)
        
        # Step 5: Export model
        print("\n🔄 Step 5: Exporting model...")
        if os.path.exists(model_path):
            export_for_production(model_path, "quick_export")
        
        print("\n🎉 Quick Training Complete!")
        print("📁 Check the following directories:")
        print("   - models/ - Trained model files")
        print("   - plots/ - Training visualizations")
        print("   - quick_export/ - Exported model for production")
        
        print("\n📊 Training Summary:")
        print(f"   Architecture: {info.get('architecture', 'simple')}")
        print(f"   Classes: {info.get('num_classes', 'unknown')}")
        print(f"   Test Accuracy: {info.get('final_test_accuracy', 'unknown'):.4f}")
        print(f"   Training Time: {info.get('training_time', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Quick training failed: {e}")
        print("Check the error messages above for details.")

if __name__ == "__main__":
    # Force quick config
    import config
    config.USE_QUICK_CONFIG = True
    
    quick_train_pipeline()