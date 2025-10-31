#!/usr/bin/env python3
"""
Visualization utilities for training
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plot_training_history(history, save_path=None):
    """Plot training history with loss and accuracy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation accuracy
    ax1.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training & validation loss
    ax2.plot(history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved: {save_path}")
    
    plt.show()

def plot_interactive_training_history(history, save_path=None):
    """Create interactive training history plot using Plotly"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy', 'Loss', 'Top-3 Accuracy', 'Learning Rate'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    epochs = list(range(1, len(history['accuracy']) + 1))
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(x=epochs, y=history['accuracy'], name='Train Accuracy', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_accuracy'], name='Val Accuracy', line=dict(color='red')),
        row=1, col=1
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(x=epochs, y=history['loss'], name='Train Loss', line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss', line=dict(color='red')),
        row=1, col=2
    )
    
    # Top-3 accuracy if available
    if 'top_3_accuracy' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['top_3_accuracy'], name='Train Top-3', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_top_3_accuracy'], name='Val Top-3', line=dict(color='red')),
            row=2, col=1
        )
    
    # Learning rate if available
    if 'lr' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['lr'], name='Learning Rate', line=dict(color='green')),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Training History",
        height=800,
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path.replace('.png', '.html'))
        print(f"Interactive training history saved: {save_path.replace('.png', '.html')}")
    
    fig.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, normalize=True):
    """Plot confusion matrix"""
    # Calculate confusion matrix
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Create plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved: {save_path}")
    
    plt.show()

def plot_sample_predictions(X, y_true, y_pred, class_names, save_path=None, num_samples=20):
    """Plot sample predictions with confidence scores"""
    # Select random samples
    indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
    
    # Calculate grid size
    cols = 5
    rows = (len(indices) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        row = i // cols
        col = i % cols
        
        # Get image and predictions
        image = X[idx].squeeze()
        true_label = class_names[y_true[idx].argmax()]
        pred_probs = y_pred[idx]
        pred_label = class_names[pred_probs.argmax()]
        confidence = pred_probs.max()
        
        # Determine color based on correctness
        color = 'green' if true_label == pred_label else 'red'
        
        # Plot image
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(
            f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}',
            fontsize=8,
            color=color
        )
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(len(indices), rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions plot saved: {save_path}")
    
    plt.show()

def plot_class_accuracy(y_true, y_pred, class_names, save_path=None):
    """Plot per-class accuracy"""
    # Calculate per-class accuracy
    y_true_labels = y_true.argmax(axis=1)
    y_pred_labels = y_pred.argmax(axis=1)
    
    class_accuracies = []
    for i, class_name in enumerate(class_names):
        class_mask = y_true_labels == i
        if class_mask.sum() > 0:
            class_acc = (y_pred_labels[class_mask] == i).mean()
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    # Create plot
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(class_names)), class_accuracies, color='skyblue', alpha=0.7)
    
    # Color bars based on accuracy
    for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
        if acc >= 0.8:
            bar.set_color('green')
        elif acc >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add accuracy values on bars
    for i, acc in enumerate(class_accuracies):
        plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class accuracy plot saved: {save_path}")
    
    plt.show()

def plot_top_k_accuracy(y_true, y_pred, class_names, k_values=[1, 3, 5], save_path=None):
    """Plot top-k accuracy for different k values"""
    accuracies = []
    
    for k in k_values:
        # Get top-k predictions
        top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
        true_labels = y_true.argmax(axis=1)
        
        # Check if true label is in top-k
        correct = 0
        for i, true_label in enumerate(true_labels):
            if true_label in top_k_preds[i]:
                correct += 1
        
        accuracy = correct / len(true_labels)
        accuracies.append(accuracy)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar([f'Top-{k}' for k in k_values], accuracies, color='lightblue', alpha=0.7)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Top-K Accuracy', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Top-k accuracy plot saved: {save_path}")
    
    plt.show()

def plot_model_architecture(model, save_path=None):
    """Visualize model architecture"""
    try:
        from tensorflow.keras.utils import plot_model
        
        plot_model(
            model,
            to_file=save_path or 'model_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=150
        )
        
        print(f"Model architecture plot saved: {save_path or 'model_architecture.png'}")
        
    except ImportError:
        print("⚠️  graphviz not available for model visualization")
        print("Install with: pip install graphviz")

def create_training_dashboard(history, y_true, y_pred, class_names, save_path=None):
    """Create comprehensive training dashboard"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Training History - Accuracy',
            'Training History - Loss',
            'Confusion Matrix',
            'Per-Class Accuracy',
            'Top-K Accuracy',
            'Prediction Confidence Distribution'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "histogram"}]
        ]
    )
    
    epochs = list(range(1, len(history['accuracy']) + 1))
    
    # Training history - Accuracy
    fig.add_trace(
        go.Scatter(x=epochs, y=history['accuracy'], name='Train Acc', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_accuracy'], name='Val Acc', line=dict(color='red')),
        row=1, col=1
    )
    
    # Training history - Loss
    fig.add_trace(
        go.Scatter(x=epochs, y=history['loss'], name='Train Loss', line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss', line=dict(color='red')),
        row=1, col=2
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig.add_trace(
        go.Heatmap(
            z=cm_normalized,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            showscale=False
        ),
        row=2, col=1
    )
    
    # Per-class accuracy
    y_true_labels = y_true.argmax(axis=1)
    y_pred_labels = y_pred.argmax(axis=1)
    
    class_accuracies = []
    for i in range(len(class_names)):
        class_mask = y_true_labels == i
        if class_mask.sum() > 0:
            class_acc = (y_pred_labels[class_mask] == i).mean()
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    fig.add_trace(
        go.Bar(x=class_names, y=class_accuracies, name='Class Accuracy'),
        row=2, col=2
    )
    
    # Top-k accuracy
    k_values = [1, 3, 5]
    top_k_accs = []
    
    for k in k_values:
        top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
        true_labels = y_true.argmax(axis=1)
        
        correct = sum(1 for i, true_label in enumerate(true_labels) 
                     if true_label in top_k_preds[i])
        accuracy = correct / len(true_labels)
        top_k_accs.append(accuracy)
    
    fig.add_trace(
        go.Bar(x=[f'Top-{k}' for k in k_values], y=top_k_accs, name='Top-K Accuracy'),
        row=3, col=1
    )
    
    # Prediction confidence distribution
    confidences = y_pred.max(axis=1)
    fig.add_trace(
        go.Histogram(x=confidences, nbinsx=50, name='Confidence Distribution'),
        row=3, col=2
    )
    
    fig.update_layout(
        title="Training Dashboard",
        height=1200,
        showlegend=False
    )
    
    if save_path:
        fig.write_html(save_path.replace('.png', '_dashboard.html'))
        print(f"Training dashboard saved: {save_path.replace('.png', '_dashboard.html')}")
    
    fig.show()