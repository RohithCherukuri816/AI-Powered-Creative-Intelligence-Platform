import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple, Optional, List
import timm
from huggingface_hub import hf_hub_download
import cv2

logger = logging.getLogger(__name__)

class MobileNetDoodleRecognizer:
    """
    MobileNetV2/V3 based doodle recognition system
    Optimized for efficiency and mobile deployment
    """
    
    def __init__(self, model_name: str = "mobilenetv3_large_100", num_classes: int = 100):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self.classes = self._get_quickdraw_classes()
        self._model_loaded = False
        
        # Initialize transforms
        self._setup_transforms()
        
        # Try to load model
        self._load_model()
        
        logger.info(f"MobileNet recognizer initialized on {self.device}")

    def _get_quickdraw_classes(self) -> List[str]:
        """Get the 100 Quick Draw categories"""
        return [
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

    def _setup_transforms(self):
        """Setup image preprocessing transforms for MobileNet"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # MobileNet input size
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

    def _load_model(self):
        """Load or create MobileNet model"""
        try:
            # Try to load from Hugging Face Hub first
            if self._try_load_from_hub():
                return
            
            # Create new model with pre-trained weights
            self.model = timm.create_model(
                self.model_name, 
                pretrained=True, 
                num_classes=self.num_classes
            )
            
            self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True
            
            logger.info(f"Created new {self.model_name} model with random classification head")
            
        except Exception as e:
            logger.error(f"Failed to load MobileNet model: {e}")
            self._model_loaded = False

    def _try_load_from_hub(self) -> bool:
        """Try to load trained model from Hugging Face Hub"""
        try:
            # This would be the path once we upload the trained model
            model_path = hf_hub_download(
                repo_id="vinayabc1824/doodle-recognition-mobilenet",
                filename="pytorch_model.bin"
            )
            
            # Create model architecture
            self.model = timm.create_model(
                self.model_name, 
                pretrained=False, 
                num_classes=self.num_classes
            )
            
            # Load trained weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True
            
            logger.info("Loaded trained MobileNet model from Hugging Face Hub")
            return True
            
        except Exception as e:
            logger.debug(f"Could not load from Hub (expected if not uploaded yet): {e}")
            return False

    def recognize_doodle(self, image: Image.Image) -> Tuple[Optional[str], float]:
        """Recognize doodle using MobileNet"""
        if not self._model_loaded:
            return self._fallback_recognition(image)
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(processed_image)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_label = self.classes[predicted_idx.item()]
                confidence_score = confidence.item()
                
                # Use fallback if confidence is too low
                if confidence_score < 0.3:
                    return self._fallback_recognition(image)
                
                return predicted_label, confidence_score
                
        except Exception as e:
            logger.error(f"MobileNet recognition error: {e}")
            return self._fallback_recognition(image)

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for MobileNet input"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        processed = self.transform(image)
        
        # Add batch dimension
        processed = processed.unsqueeze(0).to(self.device)
        
        return processed

    def _fallback_recognition(self, image: Image.Image) -> Tuple[Optional[str], float]:
        """Enhanced fallback recognition using shape analysis"""
        try:
            # Convert to numpy array for analysis
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
            
            # Calculate shape features
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h
            area = cv2.contourArea(largest_contour)
            extent = area / (w * h) if w * h > 0 else 0
            
            # Enhanced shape recognition
            if aspect_ratio > 2.5:
                return "train", 0.7
            elif aspect_ratio > 2.0:
                return "car", 0.7
            elif aspect_ratio > 1.5:
                return "van", 0.6
            elif aspect_ratio < 0.5:
                return "tree", 0.7
            elif 0.8 < aspect_ratio < 1.2:
                if extent > 0.7:
                    return "house", 0.6
                else:
                    return "face", 0.5
            elif len(contours) > 3:
                return "flower", 0.5
            else:
                # Check for circular shapes
                ((center_x, center_y), radius) = cv2.minEnclosingCircle(largest_contour)
                circle_area = np.pi * (radius ** 2)
                circularity = area / circle_area if circle_area > 0 else 0
                
                if circularity > 0.7:
                    return "apple", 0.6
                else:
                    return "unknown", 0.3
                    
        except Exception as e:
            logger.error(f"Fallback recognition failed: {e}")
            return "unknown", 0.1

    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "device": str(self.device),
            "model_loaded": self._model_loaded,
            "total_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else 0
        }

    def train_mode(self):
        """Switch model to training mode"""
        if self.model:
            self.model.train()

    def eval_mode(self):
        """Switch model to evaluation mode"""
        if self.model:
            self.model.eval()

    def save_model(self, path: str):
        """Save model state dict"""
        if self.model:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved to {path}")

    def load_model_weights(self, path: str):
        """Load model weights from file"""
        if self.model and os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self._model_loaded = True
            logger.info(f"Model weights loaded from {path}")