#!/usr/bin/env python3
"""
Main Training Script for Doodle Recognition Model
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from config import *
from data_preprocessing import QuickDrawPreprocessor
from model_architecture import create_model
from utils.visualization import plot_training_history, plot_sample_predictions
from utils.metrics import calculate_metrics

def setup_gpu():
    """Setup GPU configuration"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set memory limit if specified
            if GPU_CONFIG.get("memory_limit"):
                tf.config.experimental.set_memory_limit(
                    gpus[0], GPU_CONFIG["memory_limit"]
                )
            
            print(f"✅ GPU setup complete: {len(gpus)} GPU(s) available")
            return True
            
        except RuntimeError as e:
            print(f"❌ GPU setup failed: {e}")
            return False
    else:
        print("⚠️  No GPU detected, using CPU")
        return False

def create_data_generators(X_train, y_train, X_val, y_val):
    """Create data generators with augmentation"""
    if not AUGMENTATION_CONFIG["enabled"]:
        return X_train, y_train, X_val, y_val
    
    print("🔄 Creating data generators with augmentation...")
    
    # Create data generator for training
    train_datagen = ImageDataGenerator(
        rotation_range=AUGMENTATION_CONFIG["rotation_range"],
        width_shift_range=AUGMENTATION_CONFIG["width_shift_range"],
        height_shift_range=AUGMENTATION_CONFIG["height_shift_range"],
        zoom_range=AUGMENTATION_CONFIG["zoom_range"],
        horizontal_flip=AUGMENTATION_CONFIG["horizontal_flip"],
        fill_mode=AUGMENTATION_CONFIG["fill_mode"],
        cval=AUGMENTATION_CONFIG["cval"]
    )
    
    # Validation generator (no augmentation)
    val_datagen = ImageDataGenerator()
    
    # Fit generators
    train_datagen.fit(X_train)
    val_datagen.fit(X_val)
    
    print("✅ Data generators created")
    return train_datagen, val_datagen

def create_callbacks(model_name):
    """Create training callbacks"""
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = os.path.join(PATHS["models_dir"], f"{model_name}_best.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=TRAINING_CONFIG["save_best_only"],
        save_weights_only=TRAINING_CONFIG["save_weights_only"],
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping
    if TRAINING_CONFIG["use_early_stopping"]:
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=TRAINING_CONFIG["early_stopping"]["patience"],
            restore_best_weights=TRAINING_CONFIG["early_stopping"]["restore_best_weights"],
            verbose=1
        )
        callbacks.append(early_stop)
    
    # Learning rate scheduler
    if TRAINING_CONFIG["use_lr_scheduler"]:
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=TRAINING_CONFIG["lr_schedule"]["factor"],
            patience=TRAINING_CONFIG["lr_schedule"]["patience"],
            min_lr=TRAINING_CONFIG["lr_schedule"]["min_lr"],
            verbose=1
        )
        callbacks.append(lr_scheduler)
    
    # TensorBoard
    if LOGGING_CONFIG["tensorboard"]:
        log_dir = os.path.join(PATHS["logs_dir"], f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard)
        print(f"📊 TensorBoard logs: {log_dir}")
    
    return callbacks

def train_model(architecture='simple', resume_from=None):
    """Main training function"""
    print("🎨 Starting Doodle Recognition Training")
    print("=" * 50)
    
    # Setup GPU
    setup_gpu()
    
    # Load preprocessed data
    print("📂 Loading preprocessed data...")
    preprocessor = QuickDrawPreprocessor()
    
    try:
        (X_train, X_val, X_test), (y_train, y_val, y_test), info = preprocessor.load_preprocessed_data()
    except FileNotFoundError:
        print("❌ Preprocessed data not found. Running preprocessing...")
        from data_preprocessing import main as preprocess_main
        preprocess_main()
        (X_train, X_val, X_test), (y_train, y_val, y_test), info = preprocessor.load_preprocessed_data()
    
    num_classes = info["num_classes"]
    class_names = info["class_names"]
    
    print(f"✅ Data loaded:")
    print(f"   Classes: {num_classes}")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,}")
    print(f"   Test samples: {len(X_test):,}")
    
    # Create model
    print(f"\n🏗️  Creating {architecture} model...")
    
    # Use quick config if enabled
    if USE_QUICK_CONFIG:
        config = QUICK_CONFIG
    else:
        config = TRAINING_CONFIG
    
    model, classifier = create_model(
        num_classes=num_classes,
        architecture=architecture,
        input_shape=info["input_shape"]
    )
    
    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        print(f"📂 Loading weights from {resume_from}")
        model.load_weights(resume_from)
    
    # Create data generators
    if AUGMENTATION_CONFIG["enabled"]:
        train_gen, val_gen = create_data_generators(X_train, y_train, X_val, y_val)
        
        train_generator = train_gen.flow(
            X_train, y_train,
            batch_size=config["batch_size"],
            shuffle=True
        )
        
        val_generator = val_gen.flow(
            X_val, y_val,
            batch_size=config["batch_size"],
            shuffle=False
        )
        
        steps_per_epoch = len(X_train) // config["batch_size"]
        validation_steps = len(X_val) // config["batch_size"]
        
    else:
        train_generator = None
        val_generator = None
        steps_per_epoch = None
        validation_steps = None
    
    # Create callbacks
    model_name = f"doodle_classifier_{architecture}"
    callbacks = create_callbacks(model_name)
    
    # Training configuration
    print(f"\n⚙️  Training Configuration:")
    print(f"   Architecture: {architecture}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Augmentation: {AUGMENTATION_CONFIG['enabled']}")
    
    # Start training
    print(f"\n🚀 Starting training...")
    start_time = datetime.now()
    
    if AUGMENTATION_CONFIG["enabled"]:
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=config["epochs"],
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    
    training_time = datetime.now() - start_time
    print(f"✅ Training completed in {training_time}")
    
    # Evaluate on test set
    print(f"\n📊 Evaluating on test set...")
    test_loss, test_accuracy, test_top3 = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"📈 Final Results:")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Test Top-3 Accuracy: {test_top3:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(PATHS["models_dir"], f"{model_name}_final.h5")
    model.save(final_model_path)
    print(f"💾 Final model saved: {final_model_path}")
    
    # Save training info
    training_info = {
        "architecture": architecture,
        "num_classes": num_classes,
        "class_names": class_names,
        "input_shape": info["input_shape"],
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "final_test_accuracy": float(test_accuracy),
        "final_test_top3_accuracy": float(test_top3),
        "final_test_loss": float(test_loss),
        "training_time": str(training_time),
        "model_parameters": int(model.count_params()),
        "config_used": "quick" if USE_QUICK_CONFIG else "full"
    }
    
    info_path = os.path.join(PATHS["models_dir"], f"{model_name}_info.json")
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    
    # Plot training history
    plot_path = os.path.join(PATHS["plots_dir"], f"{model_name}_history.png")
    plot_training_history(history.history, plot_path)
    
    # Plot sample predictions
    predictions = model.predict(X_test[:100], verbose=0)
    pred_plot_path = os.path.join(PATHS["plots_dir"], f"{model_name}_predictions.png")
    plot_sample_predictions(X_test[:100], y_test[:100], predictions, class_names, pred_plot_path)
    
    print(f"✅ Visualizations saved to {PATHS['plots_dir']}")
    
    print(f"\n🎉 Training Complete!")
    print(f"📁 Model files saved in: {PATHS['models_dir']}")
    print(f"📊 Logs and plots in: {PATHS['logs_dir']} and {PATHS['plots_dir']}")
    
    return model, history, training_info

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Train Doodle Recognition Model')
    parser.add_argument('--architecture', type=str, default='simple',
                       choices=['simple', 'advanced', 'lightweight', 'mobilenet'],
                       help='Model architecture to use')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--quick', action='store_true',
                       help='Use quick configuration for testing')
    
    args = parser.parse_args()
    
    # Override quick config if specified
    if args.quick:
        global USE_QUICK_CONFIG
        USE_QUICK_CONFIG = True
        print("⚡ Using quick configuration")
    
    # Start training
    model, history, info = train_model(args.architecture, args.resume)
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Run: python evaluate_model.py")
    print(f"   2. Run: python export_model.py")
    print(f"   3. Copy trained model to main project")

if __name__ == "__main__":
    main()