#!/usr/bin/env python3
"""
Metrics and evaluation utilities
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, top_k_accuracy_score
)
import json

def calculate_metrics(y_true, y_pred, class_names, average='weighted'):
    """Calculate comprehensive classification metrics"""
    
    # Convert one-hot to labels if needed
    if len(y_true.shape) > 1:
        y_true_labels = y_true.argmax(axis=1)
    else:
        y_true_labels = y_true
    
    if len(y_pred.shape) > 1:
        y_pred_labels = y_pred.argmax(axis=1)
    else:
        y_pred_labels = y_pred
    
    # Basic metrics
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision = precision_score(y_true_labels, y_pred_labels, average=average, zero_division=0)
    recall = recall_score(y_true_labels, y_pred_labels, average=average, zero_division=0)
    f1 = f1_score(y_true_labels, y_pred_labels, average=average, zero_division=0)
    
    # Top-k accuracy (if predictions are probabilities)
    top_3_acc = None
    top_5_acc = None
    if len(y_pred.shape) > 1:
        try:
            top_3_acc = top_k_accuracy_score(y_true_labels, y_pred, k=3)
            if len(class_names) >= 5:
                top_5_acc = top_k_accuracy_score(y_true_labels, y_pred, k=5)
        except:
            pass
    
    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_mask = y_true_labels == i
        if class_mask.sum() > 0:
            class_precision = precision_score(
                y_true_labels == i, y_pred_labels == i, zero_division=0
            )
            class_recall = recall_score(
                y_true_labels == i, y_pred_labels == i, zero_division=0
            )
            class_f1 = f1_score(
                y_true_labels == i, y_pred_labels == i, zero_division=0
            )
            class_accuracy = (y_pred_labels[class_mask] == i).mean()
            
            per_class_metrics[class_name] = {
                'precision': float(class_precision),
                'recall': float(class_recall),
                'f1_score': float(class_f1),
                'accuracy': float(class_accuracy),
                'support': int(class_mask.sum())
            }
    
    # Overall metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'top_3_accuracy': float(top_3_acc) if top_3_acc is not None else None,
        'top_5_accuracy': float(top_5_acc) if top_5_acc is not None else None,
        'per_class_metrics': per_class_metrics,
        'num_samples': len(y_true_labels),
        'num_classes': len(class_names)
    }
    
    return metrics

def print_classification_report(y_true, y_pred, class_names):
    """Print detailed classification report"""
    
    # Convert one-hot to labels if needed
    if len(y_true.shape) > 1:
        y_true_labels = y_true.argmax(axis=1)
    else:
        y_true_labels = y_true
    
    if len(y_pred.shape) > 1:
        y_pred_labels = y_pred.argmax(axis=1)
    else:
        y_pred_labels = y_pred
    
    print("📊 Classification Report")
    print("=" * 60)
    
    report = classification_report(
        y_true_labels, 
        y_pred_labels, 
        target_names=class_names,
        digits=4
    )
    print(report)

def calculate_confidence_metrics(y_pred):
    """Calculate confidence-related metrics"""
    if len(y_pred.shape) == 1:
        return None
    
    # Prediction confidences (max probability)
    confidences = y_pred.max(axis=1)
    
    # Entropy (uncertainty measure)
    epsilon = 1e-8  # Small value to avoid log(0)
    entropies = -np.sum(y_pred * np.log(y_pred + epsilon), axis=1)
    
    # Confidence statistics
    confidence_stats = {
        'mean_confidence': float(confidences.mean()),
        'std_confidence': float(confidences.std()),
        'min_confidence': float(confidences.min()),
        'max_confidence': float(confidences.max()),
        'median_confidence': float(np.median(confidences)),
        'mean_entropy': float(entropies.mean()),
        'std_entropy': float(entropies.std()),
        'high_confidence_ratio': float((confidences > 0.8).mean()),
        'low_confidence_ratio': float((confidences < 0.5).mean())
    }
    
    return confidence_stats

def analyze_misclassifications(y_true, y_pred, class_names, top_n=10):
    """Analyze most common misclassifications"""
    
    # Convert one-hot to labels if needed
    if len(y_true.shape) > 1:
        y_true_labels = y_true.argmax(axis=1)
    else:
        y_true_labels = y_true
    
    if len(y_pred.shape) > 1:
        y_pred_labels = y_pred.argmax(axis=1)
    else:
        y_pred_labels = y_pred
    
    # Find misclassifications
    misclassified = y_true_labels != y_pred_labels
    
    if not misclassified.any():
        return {"message": "No misclassifications found!"}
    
    # Count misclassification pairs
    misclass_pairs = {}
    for i in range(len(y_true_labels)):
        if misclassified[i]:
            true_class = class_names[y_true_labels[i]]
            pred_class = class_names[y_pred_labels[i]]
            pair = f"{true_class} -> {pred_class}"
            
            if pair not in misclass_pairs:
                misclass_pairs[pair] = 0
            misclass_pairs[pair] += 1
    
    # Sort by frequency
    sorted_pairs = sorted(misclass_pairs.items(), key=lambda x: x[1], reverse=True)
    
    print(f"🔍 Top {top_n} Misclassifications:")
    print("-" * 40)
    
    analysis = {
        'total_misclassifications': int(misclassified.sum()),
        'misclassification_rate': float(misclassified.mean()),
        'top_misclassifications': []
    }
    
    for i, (pair, count) in enumerate(sorted_pairs[:top_n]):
        percentage = count / misclassified.sum() * 100
        print(f"{i+1:2d}. {pair}: {count} ({percentage:.1f}%)")
        
        analysis['top_misclassifications'].append({
            'pair': pair,
            'count': int(count),
            'percentage': float(percentage)
        })
    
    return analysis

def calculate_class_confusion(y_true, y_pred, class_names):
    """Calculate detailed class confusion analysis"""
    
    # Convert one-hot to labels if needed
    if len(y_true.shape) > 1:
        y_true_labels = y_true.argmax(axis=1)
    else:
        y_true_labels = y_true
    
    if len(y_pred.shape) > 1:
        y_pred_labels = y_pred.argmax(axis=1)
    else:
        y_pred_labels = y_pred
    
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Find most confused classes
    confusion_analysis = {}
    
    for i, true_class in enumerate(class_names):
        # Find most common misclassifications for this class
        row = cm_normalized[i]
        
        # Get indices sorted by confusion (excluding correct predictions)
        confused_indices = np.argsort(row)[::-1]
        confused_indices = confused_indices[confused_indices != i]  # Remove correct class
        
        most_confused_with = []
        for j in confused_indices[:3]:  # Top 3 confusions
            if row[j] > 0.01:  # Only if > 1% confusion
                most_confused_with.append({
                    'class': class_names[j],
                    'confusion_rate': float(row[j])
                })
        
        confusion_analysis[true_class] = {
            'accuracy': float(row[i]),
            'most_confused_with': most_confused_with,
            'total_samples': int(cm[i].sum())
        }
    
    return confusion_analysis

def evaluate_model_robustness(model, X_test, y_test, noise_levels=[0.1, 0.2, 0.3]):
    """Evaluate model robustness to noise"""
    print("🧪 Testing Model Robustness")
    print("-" * 30)
    
    robustness_results = {
        'clean_accuracy': None,
        'noisy_accuracies': {}
    }
    
    # Clean accuracy
    clean_pred = model.predict(X_test, verbose=0)
    clean_acc = accuracy_score(y_test.argmax(axis=1), clean_pred.argmax(axis=1))
    robustness_results['clean_accuracy'] = float(clean_acc)
    
    print(f"Clean accuracy: {clean_acc:.4f}")
    
    # Test with different noise levels
    for noise_level in noise_levels:
        # Add Gaussian noise
        noisy_X = X_test + np.random.normal(0, noise_level, X_test.shape)
        noisy_X = np.clip(noisy_X, 0, 1)  # Keep in valid range
        
        # Predict on noisy data
        noisy_pred = model.predict(noisy_X, verbose=0)
        noisy_acc = accuracy_score(y_test.argmax(axis=1), noisy_pred.argmax(axis=1))
        
        robustness_results['noisy_accuracies'][noise_level] = float(noisy_acc)
        
        accuracy_drop = clean_acc - noisy_acc
        print(f"Noise σ={noise_level}: {noisy_acc:.4f} (drop: {accuracy_drop:.4f})")
    
    return robustness_results

def save_evaluation_report(metrics, save_path):
    """Save comprehensive evaluation report"""
    
    report = {
        'timestamp': str(np.datetime64('now')),
        'metrics': metrics,
        'summary': {
            'overall_accuracy': metrics['accuracy'],
            'best_performing_classes': [],
            'worst_performing_classes': [],
            'recommendations': []
        }
    }
    
    # Find best and worst performing classes
    class_accuracies = [(name, data['accuracy']) 
                       for name, data in metrics['per_class_metrics'].items()]
    class_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    report['summary']['best_performing_classes'] = class_accuracies[:5]
    report['summary']['worst_performing_classes'] = class_accuracies[-5:]
    
    # Generate recommendations
    recommendations = []
    
    if metrics['accuracy'] < 0.8:
        recommendations.append("Consider increasing model complexity or training time")
    
    if metrics['top_3_accuracy'] and metrics['top_3_accuracy'] - metrics['accuracy'] > 0.2:
        recommendations.append("Model shows good top-3 performance, consider ensemble methods")
    
    worst_classes = [name for name, acc in class_accuracies[-3:] if acc < 0.6]
    if worst_classes:
        recommendations.append(f"Focus on improving these classes: {', '.join(worst_classes)}")
    
    report['summary']['recommendations'] = recommendations
    
    # Save report
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Evaluation report saved: {save_path}")
    
    return report