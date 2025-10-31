#!/usr/bin/env python3
"""
Export trained model for production use
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import argparse

from config import PATHS, EXPORT_CONFIG
from data_preprocessing import QuickDrawPreprocessor

def export_for_production(model_path, output_dir="exported_models"):
    """Export model for production deployment"""
    
    print("📦 Exporting Model for Production")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"📂 Loading model from {model_path}")
    model = load_model(model_path)
    
    # Load preprocessing info
    preprocessor = QuickDrawPreprocessor()
    try:
        _, _, info = preprocessor.load_preprocessed_data()
        class_names = info["class_names"]
        input_shape = info["input_shape"]
    except:
        print("⚠️  Could not load preprocessing info, using defaults")
        class_names = [f"class_{i}" for i in range(model.output_shape[1])]
        input_shape = (28, 28, 1)
    
    model_name = EXPORT_CONFIG["model_name"]
    
    # 1. Export as H5 format (Keras native)
    if "h5" in EXPORT_CONFIG["export_formats"]:
        h5_path = os.path.join(output_dir, f"{model_name}.h5")
        model.save(h5_path)
        print(f"✅ H5 model saved: {h5_path}")
    
    # 2. Export as SavedModel format (TensorFlow native)
    if "savedmodel" in EXPORT_CONFIG["export_formats"]:
        savedmodel_path = os.path.join(output_dir, f"{model_name}_savedmodel")
        model.save(savedmodel_path, save_format='tf')
        print(f"✅ SavedModel saved: {savedmodel_path}")
    
    # 3. Export as TensorFlow Lite (for mobile/edge deployment)
    if "tflite" in EXPORT_CONFIG["export_formats"]:
        tflite_path = os.path.join(output_dir, f"{model_name}.tflite")
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if EXPORT_CONFIG.get("optimize_for_inference", True):
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TensorFlow Lite model saved: {tflite_path}")
        
        # Test TFLite model
        test_tflite_model(tflite_path, input_shape)
    
    # 4. Export class names and metadata
    metadata = {
        "model_name": model_name,
        "class_names": class_names,
        "num_classes": len(class_names),
        "input_shape": input_shape,
        "model_parameters": int(model.count_params()),
        "preprocessing": {
            "normalize": True,
            "invert_colors": True,
            "resize_to": input_shape[:2]
        },
        "export_info": {
            "tensorflow_version": tf.__version__,
            "formats": EXPORT_CONFIG["export_formats"]
        }
    }
    
    metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved: {metadata_path}")
    
    # 5. Export label encoder if available
    try:
        processed_dir = "data/processed"
        label_encoder_path = os.path.join(processed_dir, "label_encoder.pkl")
        
        if os.path.exists(label_encoder_path):
            import shutil
            output_encoder_path = os.path.join(output_dir, "label_encoder.pkl")
            shutil.copy2(label_encoder_path, output_encoder_path)
            print(f"✅ Label encoder copied: {output_encoder_path}")
    except Exception as e:
        print(f"⚠️  Could not copy label encoder: {e}")
    
    # 6. Create inference script template
    create_inference_script(output_dir, model_name, metadata)
    
    # 7. Create integration guide
    create_integration_guide(output_dir, model_name, metadata)
    
    print(f"\n🎉 Export Complete!")
    print(f"📁 All files saved to: {output_dir}")
    
    return output_dir

def test_tflite_model(tflite_path, input_shape):
    """Test TensorFlow Lite model"""
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Test with dummy data
        dummy_input = np.random.random((1,) + input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"   TFLite test successful - Output shape: {output.shape}")
        
    except Exception as e:
        print(f"   ⚠️  TFLite test failed: {e}")

def create_inference_script(output_dir, model_name, metadata):
    """Create inference script template"""
    
    script_content = f'''#!/usr/bin/env python3
"""
Inference script for {model_name}
Auto-generated by export_model.py
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import json

class DoodleClassifier:
    def __init__(self, model_path, metadata_path):
        """Initialize the classifier"""
        self.model = tf.keras.models.load_model(model_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.class_names = self.metadata['class_names']
        self.input_shape = tuple(self.metadata['input_shape'])
        
        print(f"Model loaded: {{len(self.class_names)}} classes")
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to model input size
        target_size = self.input_shape[:2]
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Invert colors (white strokes on black background)
        img_array = 1.0 - img_array
        
        # Add batch and channel dimensions
        img_array = img_array.reshape(1, *self.input_shape)
        
        return img_array
    
    def predict(self, image, top_k=3):
        """Make prediction on image"""
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Predict
        predictions = self.model.predict(processed_image, verbose=0)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({{
                'class': self.class_names[idx],
                'confidence': float(predictions[idx])
            }})
        
        return results
    
    def predict_from_file(self, image_path, top_k=3):
        """Predict from image file"""
        image = Image.open(image_path)
        return self.predict(image, top_k)

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    
    # Initialize classifier
    classifier = DoodleClassifier(
        model_path="{model_name}.h5",
        metadata_path="{model_name}_metadata.json"
    )
    
    # Make prediction
    image_path = sys.argv[1]
    results = classifier.predict_from_file(image_path)
    
    print(f"Predictions for {{image_path}}:")
    for i, result in enumerate(results, 1):
        print(f"  {{i}}. {{result['class']}}: {{result['confidence']:.4f}}")
'''
    
    script_path = os.path.join(output_dir, "inference.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Inference script created: {script_path}")

def create_integration_guide(output_dir, model_name, metadata):
    """Create integration guide"""
    
    guide_content = f'''# {model_name} Integration Guide

## Overview
This directory contains the exported {model_name} model and all necessary files for production deployment.

## Files Included
- `{model_name}.h5` - Keras model file
- `{model_name}_metadata.json` - Model metadata and configuration
- `inference.py` - Ready-to-use inference script
- `label_encoder.pkl` - Label encoder (if available)
- `integration_guide.md` - This file

## Model Information
- **Classes**: {metadata['num_classes']}
- **Input Shape**: {metadata['input_shape']}
- **Parameters**: {metadata['model_parameters']:,}
- **TensorFlow Version**: {metadata['export_info']['tensorflow_version']}

## Quick Start

### 1. Install Dependencies
```bash
pip install tensorflow pillow numpy
```

### 2. Basic Usage
```python
from inference import DoodleClassifier

# Initialize classifier
classifier = DoodleClassifier(
    model_path="{model_name}.h5",
    metadata_path="{model_name}_metadata.json"
)

# Predict from image file
results = classifier.predict_from_file("my_doodle.png")
print(results)
```

### 3. Command Line Usage
```bash
python inference.py my_doodle.png
```

## Integration with Main Project

### Step 1: Copy Model Files
Copy the following files to your main project:
```bash
cp {model_name}.h5 ../colab_backend/models/
cp {model_name}_metadata.json ../colab_backend/models/
cp label_encoder.pkl ../colab_backend/models/  # if available
```

### Step 2: Update Recognizer
Update your `colab_backend/recognizer.py` to use the trained model:

```python
class DoodleRecognizer:
    def __init__(self):
        # Load trained model instead of creating random one
        model_path = "models/{model_name}.h5"
        metadata_path = "models/{model_name}_metadata.json"
        
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.classes = metadata['class_names']
            
            print(f"✅ Loaded trained model with {{len(self.classes)}} classes")
        else:
            # Fallback to existing implementation
            print("⚠️  Trained model not found, using fallback")
```

## Image Preprocessing
The model expects images to be preprocessed as follows:
1. Convert to grayscale
2. Resize to {metadata['input_shape'][:2]}
3. Normalize pixel values to [0, 1]
4. Invert colors (white strokes → black strokes)
5. Add batch dimension

## Performance Expectations
- **Accuracy**: Check evaluation report for detailed metrics
- **Inference Time**: ~10-50ms per image (CPU)
- **Memory Usage**: ~{metadata['model_parameters'] * 4 // (1024*1024)}MB

## Troubleshooting

### Common Issues
1. **Import Error**: Make sure TensorFlow is installed
2. **Shape Mismatch**: Ensure input images are preprocessed correctly
3. **Low Accuracy**: Check if image preprocessing matches training

### Performance Optimization
1. Use GPU for faster inference if available
2. Batch multiple predictions together
3. Consider TensorFlow Lite for mobile deployment

## Support
For issues or questions, refer to the main project documentation.
'''
    
    guide_path = os.path.join(output_dir, "integration_guide.md")
    with open(guide_path, 'w') as f:
        f.write(guide_content)
    
    print(f"✅ Integration guide created: {guide_path}")

def main():
    """Main export function"""
    parser = argparse.ArgumentParser(description='Export trained model for production')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.h5 file)')
    parser.add_argument('--output', type=str, default="exported_models",
                       help='Output directory for exported files')
    parser.add_argument('--formats', nargs='+', 
                       choices=['h5', 'savedmodel', 'tflite'],
                       default=['h5', 'savedmodel'],
                       help='Export formats')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    # Update export config
    EXPORT_CONFIG["export_formats"] = args.formats
    
    # Export model
    output_dir = export_for_production(args.model, args.output)
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Test the exported model: python {output_dir}/inference.py <image>")
    print(f"   2. Copy model files to your main project")
    print(f"   3. Update your recognizer to use the trained model")
    print(f"   4. Deploy to production!")

if __name__ == "__main__":
    main()