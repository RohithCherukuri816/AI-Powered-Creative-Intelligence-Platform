#!/usr/bin/env python3
"""
Data utility functions for training
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def analyze_dataset(X, y, class_names):
    """Analyze dataset distribution and characteristics"""
    print("📊 Dataset Analysis")
    print("=" * 30)
    
    # Basic statistics
    print(f"Total samples: {len(X):,}")
    print(f"Input shape: {X.shape[1:]}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Data type: {X.dtype}")
    print(f"Value range: [{X.min():.3f}, {X.max():.3f}]")
    
    # Class distribution
    unique, counts = np.unique(y.argmax(axis=1), return_counts=True)
    print(f"\nClass distribution:")
    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
        percentage = count / len(y) * 100
        print(f"  {class_name}: {count:,} samples ({percentage:.1f}%)")
    
    # Check for class imbalance
    min_samples = counts.min()
    max_samples = counts.max()
    imbalance_ratio = max_samples / min_samples
    
    if imbalance_ratio > 2.0:
        print(f"\n⚠️  Class imbalance detected (ratio: {imbalance_ratio:.1f})")
        print("Consider using class weights or resampling techniques")
    else:
        print(f"\n✅ Classes are reasonably balanced (ratio: {imbalance_ratio:.1f})")
    
    return {
        "total_samples": len(X),
        "input_shape": X.shape[1:],
        "num_classes": len(class_names),
        "class_distribution": dict(zip([class_names[i] for i in unique], counts)),
        "imbalance_ratio": imbalance_ratio
    }

def visualize_samples(X, y, class_names, num_samples=20, save_path=None):
    """Visualize random samples from each class"""
    num_classes = len(class_names)
    
    # Create subplot grid
    cols = min(10, num_classes)
    rows = (num_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Get class indices
    y_labels = y.argmax(axis=1)
    
    for i, class_name in enumerate(class_names):
        row = i // cols
        col = i % cols
        
        # Find samples for this class
        class_indices = np.where(y_labels == i)[0]
        
        if len(class_indices) > 0:
            # Select random sample
            sample_idx = np.random.choice(class_indices)
            sample_image = X[sample_idx].squeeze()
            
            # Plot
            axes[row, col].imshow(sample_image, cmap='gray')
            axes[row, col].set_title(class_name, fontsize=8)
            axes[row, col].axis('off')
        else:
            axes[row, col].text(0.5, 0.5, 'No samples', ha='center', va='center')
            axes[row, col].set_title(class_name, fontsize=8)
            axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(num_classes, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sample visualization saved: {save_path}")
    
    plt.show()

def create_class_weights(y):
    """Create class weights for imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    
    y_labels = y.argmax(axis=1)
    classes = np.unique(y_labels)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=y_labels
    )
    
    class_weight_dict = dict(zip(classes, class_weights))
    
    print("⚖️  Class weights computed:")
    for class_idx, weight in class_weight_dict.items():
        print(f"  Class {class_idx}: {weight:.3f}")
    
    return class_weight_dict

def augment_minority_classes(X, y, target_samples_per_class=None):
    """Augment minority classes to balance dataset"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    y_labels = y.argmax(axis=1)
    unique_classes, class_counts = np.unique(y_labels, return_counts=True)
    
    if target_samples_per_class is None:
        target_samples_per_class = class_counts.max()
    
    print(f"🔄 Augmenting minority classes to {target_samples_per_class} samples each...")
    
    # Data generator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode='constant',
        cval=0
    )
    
    augmented_X = []
    augmented_y = []
    
    for class_idx in unique_classes:
        class_mask = y_labels == class_idx
        class_X = X[class_mask]
        class_y = y[class_mask]
        
        current_count = len(class_X)
        
        if current_count < target_samples_per_class:
            # Add original samples
            augmented_X.append(class_X)
            augmented_y.append(class_y)
            
            # Generate additional samples
            needed_samples = target_samples_per_class - current_count
            
            # Create generator
            gen = datagen.flow(class_X, class_y, batch_size=needed_samples, shuffle=True)
            aug_X, aug_y = next(gen)
            
            augmented_X.append(aug_X)
            augmented_y.append(aug_y)
            
            print(f"  Class {class_idx}: {current_count} -> {target_samples_per_class} samples")
        else:
            # Keep original samples (possibly subsample if too many)
            if current_count > target_samples_per_class:
                indices = np.random.choice(current_count, target_samples_per_class, replace=False)
                class_X = class_X[indices]
                class_y = class_y[indices]
            
            augmented_X.append(class_X)
            augmented_y.append(class_y)
    
    # Combine all classes
    X_balanced = np.vstack(augmented_X)
    y_balanced = np.vstack(augmented_y)
    
    # Shuffle
    indices = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]
    
    print(f"✅ Dataset balanced: {len(X)} -> {len(X_balanced)} samples")
    
    return X_balanced, y_balanced

def calculate_dataset_statistics(X):
    """Calculate dataset statistics for normalization"""
    stats = {
        "mean": float(X.mean()),
        "std": float(X.std()),
        "min": float(X.min()),
        "max": float(X.max()),
        "median": float(np.median(X)),
        "percentile_25": float(np.percentile(X, 25)),
        "percentile_75": float(np.percentile(X, 75))
    }
    
    print("📊 Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    return stats

def split_by_drawing_complexity(X, y, complexity_threshold=0.1):
    """Split dataset by drawing complexity (amount of ink used)"""
    # Calculate complexity as the proportion of non-zero pixels
    complexity_scores = []
    
    for i in range(len(X)):
        image = X[i].squeeze()
        non_zero_pixels = np.count_nonzero(image)
        total_pixels = image.size
        complexity = non_zero_pixels / total_pixels
        complexity_scores.append(complexity)
    
    complexity_scores = np.array(complexity_scores)
    
    # Split into simple and complex drawings
    simple_mask = complexity_scores < complexity_threshold
    complex_mask = ~simple_mask
    
    simple_X, simple_y = X[simple_mask], y[simple_mask]
    complex_X, complex_y = X[complex_mask], y[complex_mask]
    
    print(f"📊 Complexity-based split:")
    print(f"  Simple drawings: {len(simple_X):,} ({len(simple_X)/len(X)*100:.1f}%)")
    print(f"  Complex drawings: {len(complex_X):,} ({len(complex_X)/len(X)*100:.1f}%)")
    
    return (simple_X, simple_y), (complex_X, complex_y), complexity_scores