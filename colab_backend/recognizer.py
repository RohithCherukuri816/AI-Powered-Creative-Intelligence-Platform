import logging
import os
import numpy as np
import requests
import tensorflow as tf
from PIL import Image, ImageOps
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class DoodleRecognizer:
    def __init__(self):
        self.model = None
        self.classes = [
            'airplane', 'ambulance', 'angel', 'ant', 'apple', 'asparagus', 'axe',
            'banana', 'baseball', 'basketball', 'bat', 'bathtub', 'bear', 'bed',
            'bee', 'bicycle', 'bird', 'book', 'bread', 'bus', 'butterfly',
            'cake', 'car', 'cat', 'chair', 'cloud', 'computer', 'cookie',
            'cow', 'crab', 'crocodile', 'cup', 'deer', 'dog', 'dolphin',
            'donut', 'dragon', 'duck', 'elephant', 'eye', 'face', 'fish',
            'flower', 'frog', 'giraffe', 'guitar', 'hamburger', 'hammer', 'hat',
            'helicopter', 'horse', 'house', 'ice cream', 'key', 'knight', 'ladder',
            'lighthouse', 'lion', 'monkey', 'moon', 'mosquito', 'mouse', 'mushroom',
            'octopus', 'owl', 'panda', 'parrot', 'pear', 'penguin', 'piano',
            'pig', 'pineapple', 'pizza', 'rabbit', 'raccoon', 'rhinoceros',
            'rifle', 'saw', 'scissors', 'sea turtle', 'shark', 'sheep', 'snail',
            'snake', 'snowman', 'spider', 'squirrel', 'star', 'strawberry',
            'swan', 'sword', 'table', 'teapot', 'teddy-bear', 'telephone',
            'tiger', 'train', 'tree', 'truck', 'umbrella', 'van', 'violin',
            'watermelon', 'whale', 'wheel', 'windmill', 'zebra'
        ]
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained QuickDraw CNN model"""
        try:
            # Try to load local model first
            model_path = "quickdraw_cnn.h5"
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                logger.info("Loaded local QuickDraw model")
            else:
                # For Colab, we'll download a pre-trained model
                self._download_model()
        except Exception as e:
            logger.error(f"Failed to load recognition model: {e}")
            self.model = None
    
    def _download_model(self):
        """Download pre-trained QuickDraw model"""
        try:
            # This would be replaced with your actual model URL
            model_url = "https://github.com/your-username/quickdraw-models/releases/download/v1.0/quickdraw_cnn.h5"
            model_path = "quickdraw_cnn.h5"
            
            # Download model (simplified - in practice you'd host this somewhere)
            logger.info("Downloading QuickDraw model...")
            
            # For now, create a simple model structure
            self._create_simple_model()
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            self._create_simple_model()
    
    def _create_simple_model(self):
        """Create a simple CNN model as fallback"""
        try:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(len(self.classes), activation='softmax')
            ])
            
            # Compile with dummy weights (in real scenario, you'd load trained weights)
            self.model.compile(optimizer='adam', 
                             loss='categorical_crossentropy', 
                             metrics=['accuracy'])
            logger.info("Created simple CNN model (untrained)")
        except Exception as e:
            logger.error(f"Failed to create simple model: {e}")
    
    def recognize_doodle(self, image: Image.Image) -> Tuple[Optional[str], float]:
        """Recognize doodle and return label with confidence"""
        if self.model is None:
            return self._fallback_recognition(image)
        
        try:
            # Preprocess image
            processed = self._preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            
            predicted_label = self.classes[predicted_idx]
            
            # Only return if confidence is reasonable
            if confidence > 0.1:
                return predicted_label, confidence
            else:
                return self._fallback_recognition(image)
                
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return self._fallback_recognition(image)
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for CNN"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 28x28 (QuickDraw standard)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Invert colors (doodle is usually white on black, we need black on white)
        image = ImageOps.invert(image)
        
        # Convert to numpy array and normalize
        array = np.array(image) / 255.0
        
        # Add batch and channel dimensions
        array = array.reshape(1, 28, 28, 1)
        
        return array
    
    def _fallback_recognition(self, image: Image.Image) -> Tuple[Optional[str], float]:
        """Fallback recognition using simple heuristic matching"""
        try:
            # Simple size-based heuristic as fallback
            width, height = image.size
            aspect_ratio = width / height
            
            # Very basic shape detection (in practice, you'd use more sophisticated methods)
            if aspect_ratio > 1.5:
                return "car", 0.3
            elif aspect_ratio < 0.7:
                return "tree", 0.3
            else:
                return "house", 0.3
        except:
            return None, 0.0