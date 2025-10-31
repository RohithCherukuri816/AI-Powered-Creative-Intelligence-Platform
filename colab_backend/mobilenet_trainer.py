import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import requests
import zipfile
from typing import List, Tuple, Dict
import json

logger = logging.getLogger(__name__)

class QuickDrawDataset(Dataset):
    """Dataset class for Quick Draw doodles"""
    
    def __init__(self, images: List[np.ndarray], labels: List[int], transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Convert numpy array to PIL Image
        image = Image.fromarray(self.images[idx].astype(np.uint8))
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MobileNetTrainer:
    """
    Trainer class for MobileNet on Quick Draw dataset
    """
    
    def __init__(self, model_name: str = "mobilenetv3_large_100", num_classes: int = 100):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.classes = self._get_quickdraw_classes()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        logger.info(f"MobileNet trainer initialized on {self.device}")

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

    def create_model(self, pretrained: bool = True):
        """Create MobileNet model with transfer learning"""
        try:
            # Create model with pre-trained ImageNet weights
            self.model = timm.create_model(
                self.model_name,
                pretrained=pretrained,
                num_classes=self.num_classes
            )
            
            # Freeze early layers for transfer learning
            if pretrained:
                # Freeze all layers except the classifier
                for param in self.model.parameters():
                    param.requires_grad = False
                
                # Unfreeze the classifier
                if hasattr(self.model, 'classifier'):
                    for param in self.model.classifier.parameters():
                        param.requires_grad = True
                elif hasattr(self.model, 'head'):
                    for param in self.model.head.parameters():
                        param.requires_grad = True
            
            self.model.to(self.device)
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Model created: {self.model_name}")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

    def download_quickdraw_data(self, categories: List[str] = None, samples_per_class: int = 1000):
        """Download Quick Draw dataset"""
        if categories is None:
            categories = self.classes[:10]  # Use first 10 classes for demo
        
        base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
        
        images = []
        labels = []
        
        print(f"Downloading Quick Draw data for {len(categories)} categories...")
        
        for idx, category in enumerate(tqdm(categories, desc="Downloading categories")):
            try:
                # Download category data
                url = f"{base_url}/{category}.npy"
                response = requests.get(url)
                
                if response.status_code == 200:
                    # Save temporarily
                    temp_file = f"temp_{category}.npy"
                    with open(temp_file, 'wb') as f:
                        f.write(response.content)
                    
                    # Load data
                    data = np.load(temp_file)
                    
                    # Take subset of samples
                    if len(data) > samples_per_class:
                        indices = np.random.choice(len(data), samples_per_class, replace=False)
                        data = data[indices]
                    
                    # Reshape from (N, 784) to (N, 28, 28)
                    data = data.reshape(-1, 28, 28)
                    
                    # Add to dataset
                    images.extend(data)
                    labels.extend([idx] * len(data))
                    
                    # Clean up
                    os.remove(temp_file)
                    
                    print(f"✅ {category}: {len(data)} samples")
                    
                else:
                    print(f"❌ Failed to download {category}")
                    
            except Exception as e:
                print(f"❌ Error downloading {category}: {e}")
        
        return np.array(images), np.array(labels), categories

    def prepare_data(self, images: np.ndarray, labels: np.ndarray, 
                    test_size: float = 0.2, val_size: float = 0.1, batch_size: int = 32):
        """Prepare data loaders"""
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        print(f"Dataset splits:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = QuickDrawDataset(X_train, y_train, transform=train_transform)
        val_dataset = QuickDrawDataset(X_val, y_val, transform=val_transform)
        test_dataset = QuickDrawDataset(X_test, y_test, transform=val_transform)
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    def train_epoch(self, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc

    def validate_epoch(self, criterion):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc

    def train(self, epochs: int = 10, learning_rate: float = 0.001, 
              weight_decay: float = 1e-4, save_path: str = "mobilenet_doodle_model.pth"):
        """Train the model"""
        
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        if self.train_loader is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        # Setup optimizer and criterion
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        best_val_acc = 0.0
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")

    def evaluate(self):
        """Evaluate model on test set"""
        if self.test_loader is None:
            raise ValueError("Test data not available. Call prepare_data() first.")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        
        print(f"\nTest Results:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return accuracy, all_predictions, all_targets

    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def save_model_for_huggingface(self, save_dir: str = "mobilenet_doodle_model"):
        """Save model in Hugging Face format"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)
        
        # Save config
        config = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "classes": self.classes,
            "input_size": [224, 224],
            "architecture": "MobileNet",
            "framework": "PyTorch"
        }
        
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save training history
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies
        }
        
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Model saved to {save_dir} for Hugging Face upload")
        
        return save_dir