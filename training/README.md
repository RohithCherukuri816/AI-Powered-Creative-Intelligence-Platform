# 🎨 Doodle Recognition Model Training

This directory contains everything you need to train a doodle recognition model on Kaggle's free GPU and use it in your application.

## 🚀 Quick Start (Recommended)

**Want to train on Kaggle?** → Read `QUICK_START.md`

1. Upload `kaggle_training.ipynb` to Kaggle
2. Enable GPU and run all cells (2-4 hours)
3. Download trained model
4. Copy to your project
5. Done! 🎉

## 📋 Two Ways to Train

### Option 1: Kaggle (Recommended - Free GPU) ⭐
- **File:** `kaggle_training.ipynb`
- **Guide:** `KAGGLE_GUIDE.md`
- **Time:** 2-4 hours
- **Cost:** FREE
- **Best for:** Most users

### Option 2: Local/GPU Machine (Advanced)
- **Files:** All Python scripts in this folder
- **Guide:** This README
- **Time:** 2-4 hours
- **Cost:** Requires GPU
- **Best for:** Advanced users with GPU access

## 🚀 Quick Start

### On GPU Machine:

1. **Setup Environment:**
```bash
cd training
pip install -r requirements.txt
```

2. **Download Dataset:**
```bash
python download_dataset.py
```

3. **Train Model:**
```bash
python train_model.py
```

4. **Evaluate Model:**
```bash
python evaluate_model.py
```

5. **Export for Production:**
```bash
python export_model.py
```

### Transfer Back to Main Machine:

Copy the `models/` directory back to your main project and update the recognizer to use the trained model.

## 📁 Directory Structure

```
training/
├── README.md                 # This file
├── requirements.txt          # Training dependencies
├── config.py                # Training configuration
├── download_dataset.py      # Dataset downloader
├── data_preprocessing.py    # Data preprocessing utilities
├── model_architecture.py   # CNN model definition
├── train_model.py          # Main training script
├── evaluate_model.py       # Model evaluation
├── export_model.py         # Export trained model
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── data_utils.py
│   ├── visualization.py
│   └── metrics.py
├── data/                   # Dataset storage (created automatically)
└── models/                 # Trained models (created automatically)
```

## 🎯 Features

- **Quick Draw Dataset**: 50M+ drawings across 345 categories
- **Efficient Training**: Optimized for GPU training
- **Data Augmentation**: Rotation, scaling, noise for better generalization
- **Model Checkpointing**: Save best models during training
- **Visualization**: Training progress and sample predictions
- **Export Ready**: Easy integration with main application

## ⚙️ Configuration

Edit `config.py` to customize:
- Number of classes to train on
- Batch size and learning rate
- Model architecture parameters
- Training epochs and validation split

## 🔧 Hardware Requirements

- **GPU**: NVIDIA GPU with 4GB+ VRAM (recommended)
- **RAM**: 8GB+ system RAM
- **Storage**: 10GB+ free space for dataset
- **CUDA**: Compatible CUDA installation

## 📊 Expected Results

- **Training Time**: 2-4 hours on modern GPU
- **Accuracy**: 85-92% on validation set
- **Model Size**: ~10-50MB depending on architecture
- **Inference Speed**: <50ms per prediction

## 🔄 Integration

After training, copy the trained model files to your main project:

```bash
# Copy trained model to main project
cp models/doodle_classifier.h5 ../colab_backend/models/
cp models/class_names.json ../colab_backend/models/
```

Then update your recognizer to load the trained model instead of using random weights.