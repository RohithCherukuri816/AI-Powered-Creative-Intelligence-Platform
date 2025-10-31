#!/bin/bash

# Complete training pipeline script for GPU machine
# This script will run the entire training process from start to finish

echo "🎨 Doodle Recognition Model Training Pipeline"
echo "=============================================="

# Check if we're in the training directory
if [ ! -f "config.py" ]; then
    echo "❌ Please run this script from the training directory"
    exit 1
fi

# Check Python and dependencies
echo "🔍 Checking dependencies..."
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" || {
    echo "❌ TensorFlow not found. Please install requirements first:"
    echo "   pip install -r requirements.txt"
    exit 1
}

# Check GPU availability
echo "🔍 Checking GPU availability..."
python3 -c "import tensorflow as tf; print(f'GPU Available: {tf.config.list_physical_devices(\"GPU\")}')"

# Ask user for training type
echo ""
echo "Choose training mode:"
echo "1. Quick training (10 classes, 1000 samples each, ~30 minutes)"
echo "2. Full training (100 classes, 10000 samples each, ~4 hours)"
echo "3. Custom training (specify parameters)"

read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "⚡ Starting quick training..."
        python3 quick_train.py
        ;;
    2)
        echo "🚀 Starting full training..."
        
        # Download dataset
        echo "📥 Step 1: Downloading dataset..."
        python3 download_dataset.py || {
            echo "❌ Dataset download failed"
            exit 1
        }
        
        # Preprocess data
        echo "🔄 Step 2: Preprocessing data..."
        python3 data_preprocessing.py || {
            echo "❌ Data preprocessing failed"
            exit 1
        }
        
        # Train model
        echo "🏋️ Step 3: Training model..."
        python3 train_model.py --architecture simple || {
            echo "❌ Model training failed"
            exit 1
        }
        
        # Evaluate model
        echo "📊 Step 4: Evaluating model..."
        python3 evaluate_model.py --model models/doodle_classifier_simple_final.h5 || {
            echo "❌ Model evaluation failed"
            exit 1
        }
        
        # Export model
        echo "📦 Step 5: Exporting model..."
        python3 export_model.py --model models/doodle_classifier_simple_final.h5 || {
            echo "❌ Model export failed"
            exit 1
        }
        ;;
    3)
        echo "⚙️ Custom training..."
        
        # Ask for parameters
        read -p "Architecture (simple/advanced/lightweight): " arch
        read -p "Number of epochs (default 50): " epochs
        read -p "Batch size (default 128): " batch_size
        
        # Set defaults
        arch=${arch:-simple}
        epochs=${epochs:-50}
        batch_size=${batch_size:-128}
        
        echo "🔄 Starting custom training with:"
        echo "   Architecture: $arch"
        echo "   Epochs: $epochs"
        echo "   Batch size: $batch_size"
        
        # Download and preprocess if needed
        if [ ! -d "data/processed" ]; then
            echo "📥 Downloading and preprocessing data..."
            python3 download_dataset.py
            python3 data_preprocessing.py
        fi
        
        # Train with custom parameters
        python3 train_model.py --architecture $arch
        
        # Evaluate
        python3 evaluate_model.py --model models/doodle_classifier_${arch}_final.h5
        
        # Export
        python3 export_model.py --model models/doodle_classifier_${arch}_final.h5
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Training pipeline completed!"
echo ""
echo "📁 Generated files:"
echo "   - models/          : Trained model files"
echo "   - plots/           : Training visualizations"
echo "   - exported_models/ : Production-ready model"
echo "   - logs/            : Training logs"
echo ""
echo "🚀 Next steps:"
echo "   1. Copy the exported model to your main project"
echo "   2. Update your recognizer to use the trained model"
echo "   3. Test the model with real doodles"
echo ""
echo "📋 To copy to main project:"
echo "   cp exported_models/* ../colab_backend/models/"