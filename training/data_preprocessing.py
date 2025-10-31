#!/usr/bin/env python3
"""
Data Preprocessing for Quick Draw Dataset
"""

import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import pickle

from config import DATASET_CONFIG, PATHS, TRAINING_CONFIG, USE_QUICK_CONFIG, QUICK_CONFIG

class QuickDrawPreprocessor:
    """Preprocessor for Quick Draw dataset"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.class_mapping = None
        self.num_classes = 0
        
    def load_class_mapping(self):
        """Load class mapping from file"""
        mapping_file = os.path.join(PATHS["data_dir"], "class_mapping.json")
        
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                self.class_mapping = json.load(f)
            self.num_classes = self.class_mapping["num_classes"]
            print(f"✅ Loaded class mapping: {self.num_classes} classes")
        else:
            raise FileNotFoundError("Class mapping not found. Run download_dataset.py first.")
    
    def load_raw_data(self, categories, samples_per_category=None):
        """Load raw data from .npy files"""
        print("📂 Loading raw data...")
        
        data_dir = DATASET_CONFIG["raw_data_dir"]
        all_data = []
        all_labels = []
        
        for category in tqdm(categories, desc="Loading categories"):
            filename = f"{category.replace(' ', '_')}.npy"
            filepath = os.path.join(data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"⚠️  Warning: {category} data not found, skipping...")
                continue
            
            try:
                # Load data
                category_data = np.load(filepath)
                
                # Limit samples if specified
                if samples_per_category and len(category_data) > samples_per_category:
                    category_data = category_data[:samples_per_category]
                
                # Add to collections
                all_data.append(category_data)
                all_labels.extend([category] * len(category_data))
                
                print(f"   {category}: {len(category_data):,} samples")
                
            except Exception as e:
                print(f"❌ Error loading {category}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data loaded. Check your data directory.")
        
        # Combine all data
        X = np.vstack(all_data)
        y = np.array(all_labels)
        
        print(f"✅ Loaded {len(X):,} total samples across {len(set(y))} categories")
        return X, y
    
    def preprocess_images(self, X):
        """Preprocess image data"""
        print("🖼️  Preprocessing images...")
        
        # Reshape to 28x28x1 (add channel dimension)
        X = X.reshape(-1, 28, 28, 1)
        
        # Normalize pixel values to [0, 1]
        X = X.astype('float32') / 255.0
        
        # Invert colors (Quick Draw has white strokes on black background)
        # We want black strokes on white background for better recognition
        X = 1.0 - X
        
        print(f"✅ Preprocessed images: {X.shape}")
        return X
    
    def encode_labels(self, y):
        """Encode string labels to integers and one-hot"""
        print("🏷️  Encoding labels...")
        
        # Fit label encoder
        self.label_encoder.fit(y)
        
        # Transform to integers
        y_int = self.label_encoder.transform(y)
        
        # Convert to one-hot encoding
        y_categorical = to_categorical(y_int, num_classes=len(self.label_encoder.classes_))
        
        print(f"✅ Encoded labels: {len(self.label_encoder.classes_)} classes")
        return y_categorical, y_int
    
    def split_data(self, X, y, validation_split=0.2, test_split=0.1, random_state=42):
        """Split data into train/validation/test sets"""
        print("✂️  Splitting data...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation
        val_size = validation_split / (1 - test_split)  # Adjust validation size
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        print(f"✅ Data split:")
        print(f"   Training: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Test: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)
    
    def save_preprocessed_data(self, data_splits, label_splits):
        """Save preprocessed data to disk"""
        print("💾 Saving preprocessed data...")
        
        processed_dir = DATASET_CONFIG["processed_data_dir"]
        os.makedirs(processed_dir, exist_ok=True)
        
        (X_train, X_val, X_test) = data_splits
        (y_train, y_val, y_test) = label_splits
        
        # Save data splits
        np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
        np.save(os.path.join(processed_dir, "X_val.npy"), X_val)
        np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
        np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
        np.save(os.path.join(processed_dir, "y_val.npy"), y_val)
        np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
        
        # Save label encoder
        with open(os.path.join(processed_dir, "label_encoder.pkl"), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save preprocessing info
        info = {
            "num_classes": len(self.label_encoder.classes_),
            "class_names": self.label_encoder.classes_.tolist(),
            "input_shape": X_train.shape[1:],
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        }
        
        with open(os.path.join(processed_dir, "preprocessing_info.json"), 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✅ Preprocessed data saved to {processed_dir}")
        return info
    
    def load_preprocessed_data(self):
        """Load preprocessed data from disk"""
        processed_dir = DATASET_CONFIG["processed_data_dir"]
        
        if not os.path.exists(processed_dir):
            raise FileNotFoundError("Preprocessed data not found. Run preprocessing first.")
        
        print("📂 Loading preprocessed data...")
        
        # Load data splits
        X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
        X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
        X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
        y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
        y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
        y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
        
        # Load label encoder
        with open(os.path.join(processed_dir, "label_encoder.pkl"), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load preprocessing info
        with open(os.path.join(processed_dir, "preprocessing_info.json"), 'r') as f:
            info = json.load(f)
        
        print(f"✅ Loaded preprocessed data: {info['num_classes']} classes")
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test), info

def main():
    """Main preprocessing function"""
    print("🎨 Quick Draw Data Preprocessing")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = QuickDrawPreprocessor()
    
    try:
        # Load class mapping
        preprocessor.load_class_mapping()
        
        # Use quick config if enabled
        if USE_QUICK_CONFIG:
            print("⚡ Using quick configuration")
            categories = QUICK_CONFIG["categories"]
            samples_per_category = QUICK_CONFIG["samples_per_category"]
        else:
            categories = list(preprocessor.class_mapping["class_to_idx"].keys())
            samples_per_category = DATASET_CONFIG["samples_per_category"]
        
        # Load raw data
        X, y = preprocessor.load_raw_data(categories, samples_per_category)
        
        # Preprocess images
        X = preprocessor.preprocess_images(X)
        
        # Encode labels
        y_categorical, y_int = preprocessor.encode_labels(y)
        
        # Split data
        validation_split = TRAINING_CONFIG["validation_split"]
        data_splits, label_splits = preprocessor.split_data(
            X, y_categorical, validation_split=validation_split
        )
        
        # Save preprocessed data
        info = preprocessor.save_preprocessed_data(data_splits, label_splits)
        
        print(f"\n🎉 Preprocessing Complete!")
        print(f"📊 Dataset Summary:")
        print(f"   Classes: {info['num_classes']}")
        print(f"   Input shape: {info['input_shape']}")
        print(f"   Training samples: {info['train_samples']:,}")
        print(f"   Validation samples: {info['val_samples']:,}")
        print(f"   Test samples: {info['test_samples']:,}")
        
        print(f"\n🚀 Next step: python train_model.py")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        print("Make sure you've run download_dataset.py first.")

if __name__ == "__main__":
    main()