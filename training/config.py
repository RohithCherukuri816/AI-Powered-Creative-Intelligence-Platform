"""
Training Configuration for Doodle Recognition Model
"""

import os

# Dataset Configuration
DATASET_CONFIG = {
    # Quick Draw dataset categories to train on
    # You can reduce this list for faster training or add more for better coverage
    "categories": [
        'airplane', 'ambulance', 'angel', 'ant', 'apple', 'axe', 'banana', 
        'baseball', 'basketball', 'bat', 'bathtub', 'bear', 'bed', 'bee', 
        'bicycle', 'bird', 'book', 'bread', 'bus', 'butterfly', 'cake', 
        'car', 'cat', 'chair', 'cloud', 'computer', 'cookie', 'cow', 
        'crab', 'cup', 'deer', 'dog', 'dolphin', 'donut', 'dragon', 
        'duck', 'elephant', 'eye', 'face', 'fish', 'flower', 'frog', 
        'giraffe', 'guitar', 'hamburger', 'hammer', 'hat', 'helicopter', 
        'horse', 'house', 'ice cream', 'key', 'knight', 'ladder', 
        'lighthouse', 'lion', 'monkey', 'moon', 'mosquito', 'mouse', 
        'mushroom', 'octopus', 'owl', 'panda', 'parrot', 'pear', 
        'penguin', 'piano', 'pig', 'pineapple', 'pizza', 'rabbit', 
        'raccoon', 'rhinoceros', 'saw', 'scissors', 'sea turtle', 
        'shark', 'sheep', 'snail', 'snake', 'snowman', 'spider', 
        'squirrel', 'star', 'strawberry', 'swan', 'sword', 'table', 
        'teapot', 'teddy-bear', 'telephone', 'tiger', 'train', 'tree', 
        'truck', 'umbrella', 'van', 'violin', 'watermelon', 'whale', 
        'wheel', 'windmill', 'zebra'
    ],
    
    # Number of samples per category (max available varies by category)
    "samples_per_category": 10000,  # Reduce for faster training, increase for better accuracy
    
    # Quick Draw dataset base URL
    "base_url": "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap",
    
    # Data directories
    "raw_data_dir": "data/raw",
    "processed_data_dir": "data/processed",
}

# Model Architecture Configuration
MODEL_CONFIG = {
    # Input image size (Quick Draw is 28x28)
    "input_shape": (28, 28, 1),
    
    # CNN Architecture
    "conv_layers": [
        {"filters": 32, "kernel_size": (3, 3), "activation": "relu"},
        {"filters": 64, "kernel_size": (3, 3), "activation": "relu"},
        {"filters": 128, "kernel_size": (3, 3), "activation": "relu"},
    ],
    
    # Dense layers
    "dense_layers": [
        {"units": 256, "activation": "relu", "dropout": 0.5},
        {"units": 128, "activation": "relu", "dropout": 0.3},
    ],
    
    # Output layer (will be set automatically based on number of categories)
    "output_activation": "softmax",
}

# Training Configuration
TRAINING_CONFIG = {
    # Training parameters
    "batch_size": 128,  # Adjust based on GPU memory
    "epochs": 50,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    
    # Optimizer
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy", "top_3_accuracy"],
    
    # Learning rate scheduling
    "use_lr_scheduler": True,
    "lr_schedule": {
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-7
    },
    
    # Early stopping
    "use_early_stopping": True,
    "early_stopping": {
        "patience": 10,
        "restore_best_weights": True
    },
    
    # Model checkpointing
    "save_best_only": True,
    "save_weights_only": False,
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    "enabled": True,
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.1,
    "horizontal_flip": False,  # Usually not good for doodles
    "fill_mode": "constant",
    "cval": 0,  # Fill with black
}

# Paths Configuration
PATHS = {
    "data_dir": "data",
    "models_dir": "models",
    "logs_dir": "logs",
    "plots_dir": "plots",
}

# Ensure directories exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# GPU Configuration
GPU_CONFIG = {
    "allow_growth": True,  # Allow GPU memory to grow as needed
    "memory_limit": None,  # Set to specific MB if you want to limit GPU memory
}

# Logging Configuration
LOGGING_CONFIG = {
    "log_level": "INFO",
    "save_logs": True,
    "tensorboard": True,
}

# Export Configuration
EXPORT_CONFIG = {
    "model_name": "doodle_classifier",
    "export_formats": ["h5", "savedmodel"],  # TensorFlow formats
    "include_preprocessing": True,
    "optimize_for_inference": True,
}

# Quick training configuration for testing
QUICK_CONFIG = {
    "categories": ['car', 'cat', 'dog', 'house', 'tree', 'face', 'bird', 'fish', 'flower', 'star'],
    "samples_per_category": 1000,
    "epochs": 10,
    "batch_size": 64,
}

# Use quick config for testing
USE_QUICK_CONFIG = True  # Set to True for quick testing