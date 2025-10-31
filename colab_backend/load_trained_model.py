"""
Helper script to integrate trained model from Kaggle
Place this in colab_backend/ directory
"""

import os
import json
import tensorflow as tf
import pickle
from PIL import Image
import numpy as np

class TrainedDoodleRecognizer:
    """
    Recognizer using the trained model from Kaggle
    Drop-in replacement for the existing DoodleRecognizer
    """
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.model = None
        self.classes = []
        self._model_loaded = False
        
        # Try to load trained model
        self._load_trained_model()
    
    def _load_trained_model(self):
        """Load the trained model from Kaggle"""
        model_path = os.path.join(self.model_dir, "doodle_classifier.h5")
        metadata_path = os.path.join(self.model_dir, "doodle_classifier_metadata.json")
        
        if not os.path.exists(model_path):
            print("⚠️  Trained model not found at:", model_path)
            print("Please download the model from Kaggle and place it in:", self.model_dir)
            return False
        
        try:
            print("🔄 Loading trained model from Kaggle...")
            
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            
            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.classes = metadata['class_names']
                
                print(f"✅ Loaded trained model successfully!")
                print(f"   Classes: {len(self.classes)}")
                print(f"   Test Accuracy: {metadata.get('test_accuracy', 'N/A'):.4f}")
                print(f"   Top-3 Accuracy: {metadata.get('test_top3_accuracy', 'N/A'):.4f}")
                print(f"   Parameters: {metadata.get('model_parameters', 'N/A'):,}")
            else:
                # Fallback: load class names from separate file
                class_names_path = os.path.join(self.model_dir, "class_names.json")
                if os.path.exists(class_names_path):
                    with open(class_names_path, 'r') as f:
                        self.classes = json.load(f)
                else:
                    print("⚠️  Class names not found, using default")
                    self.classes = [f"class_{i}" for i in range(self.model.output_shape[1])]
            
            self._model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to load trained model: {e}")
            return False
    
    def recognize_doodle(self, image: Image.Image):
        """
        Recognize doodle using trained model
        
        Args:
            image: PIL Image of the doodle
            
        Returns:
            tuple: (predicted_label, confidence)
        """
        if not self._model_loaded:
            print("⚠️  Model not loaded, cannot recognize")
            return None, 0.0
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)[0]
            
            # Get top prediction
            predicted_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_idx])
            predicted_label = self.classes[predicted_idx]
            
            return predicted_label, confidence
            
        except Exception as e:
            print(f"❌ Recognition failed: {e}")
            return None, 0.0
    
    def _preprocess_image(self, image: Image.Image):
        """Preprocess image for model input"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Invert colors (white strokes on black background)
        img_array = 1.0 - img_array
        
        # Add batch and channel dimensions
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    
    def get_top_k_predictions(self, image: Image.Image, k=3):
        """
        Get top-k predictions
        
        Args:
            image: PIL Image of the doodle
            k: Number of top predictions to return
            
        Returns:
            list: List of (label, confidence) tuples
        """
        if not self._model_loaded:
            return []
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)[0]
            
            # Get top-k predictions
            top_k_indices = np.argsort(predictions)[-k:][::-1]
            
            results = []
            for idx in top_k_indices:
                results.append((
                    self.classes[idx],
                    float(predictions[idx])
                ))
            
            return results
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return []


# Test function
def test_recognizer():
    """Test the trained model recognizer"""
    print("🧪 Testing Trained Model Recognizer")
    print("=" * 50)
    
    recognizer = TrainedDoodleRecognizer()
    
    if not recognizer._model_loaded:
        print("❌ Model not loaded. Please download from Kaggle first.")
        print("\nInstructions:")
        print("1. Train model on Kaggle using kaggle_training.ipynb")
        print("2. Download files from export_for_local folder")
        print("3. Copy files to colab_backend/models/")
        print("4. Run this script again")
        return
    
    # Create a test image (simple square)
    test_image = Image.new('L', (100, 100), color=255)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_image)
    draw.rectangle([20, 20, 80, 80], outline=0, width=3)
    
    # Test recognition
    label, confidence = recognizer.recognize_doodle(test_image)
    
    print(f"\n🎯 Test Recognition:")
    print(f"   Predicted: {label}")
    print(f"   Confidence: {confidence:.4f}")
    
    # Test top-k predictions
    top_k = recognizer.get_top_k_predictions(test_image, k=5)
    
    print(f"\n📊 Top-5 Predictions:")
    for i, (label, conf) in enumerate(top_k, 1):
        print(f"   {i}. {label}: {conf:.4f}")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    test_recognizer()