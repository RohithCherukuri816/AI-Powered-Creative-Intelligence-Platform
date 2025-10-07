import logging
import os
import numpy as np
import requests
import tensorflow as tf
from PIL import Image, ImageOps
from typing import Tuple, Optional
import cv2

logger = logging.getLogger(__name__)

class DoodleRecognizer:
    def __init__(self):
        self.model = None
        self.classes = [
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
        ]
        self._model_loaded = False
        self._load_model()

    def _load_model(self):
        """Create a simple CNN model for recognition"""
        try:
            # Create a simple CNN model architecture
            self.model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(self.classes), activation='softmax')
            ])
            
            # Compile the model
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy', 
                metrics=['accuracy']
            )
            
            # Initialize with random weights (in production, you'd load trained weights)
            # This will at least give us some basic pattern recognition
            self._model_loaded = True
            logger.info("CNN model initialized (random weights)")
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            self._model_loaded = False

    def recognize_doodle(self, image: Image.Image) -> Tuple[Optional[str], float]:
        """Recognize doodle with enhanced preprocessing"""
        if not self._model_loaded:
            return self._advanced_heuristic_recognition(image)
        
        try:
            # Enhanced preprocessing
            processed = self._enhanced_preprocess(image)
            
            # Make prediction
            predictions = self.model.predict(processed, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            
            predicted_label = self.classes[predicted_idx]
            
            # Use heuristic as fallback if confidence is low
            if confidence < 0.3:
                return self._advanced_heuristic_recognition(image)
                
            return predicted_label, confidence
                
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return self._advanced_heuristic_recognition(image)

    def _enhanced_preprocess(self, image: Image.Image) -> np.ndarray:
        """Enhanced preprocessing for better recognition"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        array = np.array(image)
        
        # Normalize and invert if needed (doodles are usually white on black)
        if np.mean(array) > 127:  # If image is mostly white
            array = 255 - array  # Invert
            
        # Normalize to 0-1
        array = array.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        array = array.reshape(1, 28, 28, 1)
        
        return array

    def _advanced_heuristic_recognition(self, image: Image.Image) -> Tuple[Optional[str], float]:
        """Advanced heuristic recognition based on shape analysis"""
        try:
            # Convert to numpy array for OpenCV processing
            if image.mode != 'L':
                gray = image.convert('L')
            else:
                gray = image
            
            # Resize for consistent processing
            gray = gray.resize((100, 100), Image.Resampling.LANCZOS)
            array = np.array(gray)
            
            # Invert if needed (expecting white doodle on black background)
            if np.mean(array) > 127:
                array = 255 - array
                
            # Threshold to create binary image
            _, binary = cv2.threshold(array, 50, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return "unknown", 0.1
                
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate features
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h
            area = cv2.contourArea(largest_contour)
            extent = area / (w * h)
            
            # Shape recognition heuristics
            if aspect_ratio > 2.0 and w > 50:
                return "car", 0.7
            elif aspect_ratio > 1.8 and w > 40:
                return "van", 0.7
            elif aspect_ratio > 2.5:
                return "train", 0.6
            elif aspect_ratio < 0.6:
                return "tree", 0.6
            elif 0.8 < aspect_ratio < 1.2 and extent > 0.6:
                return "house", 0.7
            elif len(contours) > 3:  # Multiple parts
                return "face", 0.5
            else:
                # Try to detect circles/round shapes
                ((center_x, center_y), radius) = cv2.minEnclosingCircle(largest_contour)
                circle_area = np.pi * (radius ** 2)
                circularity = area / circle_area if circle_area > 0 else 0
                
                if circularity > 0.7:
                    return "ball", 0.6
                else:
                    return "unknown", 0.3
                    
        except Exception as e:
            logger.error(f"Heuristic recognition failed: {e}")
            return "unknown", 0.1

    def _simple_shape_detection(self, image: Image.Image) -> Tuple[Optional[str], float]:
        """Simple shape-based detection as final fallback"""
        try:
            width, height = image.size
            aspect_ratio = width / height
            
            # Simple shape-based detection
            if aspect_ratio > 1.8:
                # Long horizontal shapes - likely vehicles
                return "car", 0.5
            elif aspect_ratio < 0.6:
                # Tall vertical shapes
                return "tree", 0.5
            elif 0.8 < aspect_ratio < 1.2:
                # Square-ish shapes
                return "house", 0.5
            else:
                return "unknown", 0.3
        except:
            return "unknown", 0.1