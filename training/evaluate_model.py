#!/usr/bin/env python3
"""
Model Evaluation Script
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse

from config import PATHS
from data_preprocessing import QuickDrawPreprocessor
from utils.metrics import (
    calculate_metrics, print_classification_report, 
    calculate_confidence_metrics, analyze_misclassifications,
    calculate_class_confusion, evaluate_model_robustness,
    save_evaluation_report
)
from utils.visualization import (
    plot_confusion_matrix, plot_sample_predictions,
    plot_class_accuracy, plot_top_k_accuracy,
    create_training_dashboard
)

def load_trained_model(model_path):
    """Load trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"📂 Loading model from {model_path}")
    
    # Load model with custom objects if needed
    try:
        model = load_model(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {model.count_params():,}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

def evaluate_model(model_path, test_on_train=False, create_visualizations=True):
    """Comprehensive model evaluation"""
    
    print("🎨 Model Evaluation")
    print("=" * 50)
    
    # Load model
    model = load_trained_model(model_path)
    
    # Load data
    print("📂 Loading test data...")
    preprocessor = QuickDrawPreprocessor()
    (X_train, X_val, X_test), (y_train, y_val, y_test), info = preprocessor.load_preprocessed_data()
    
    class_names = info["class_names"]
    
    # Choose evaluation set
    if test_on_train:
        print("⚠️  Evaluating on training set (for debugging)")
        X_eval, y_eval = X_train, y_train
        eval_name = "train"
    else:
        print("📊 Evaluating on test set")
        X_eval, y_eval = X_test, y_test
        eval_name = "test"
    
    print(f"   Evaluation samples: {len(X_eval):,}")
    print(f"   Classes: {len(class_names)}")
    
    # Make predictions
    print("🔮 Making predictions...")
    y_pred = model.predict(X_eval, batch_size=128, verbose=1)
    
    # Calculate comprehensive metrics
    print("\n📊 Calculating metrics...")
    metrics = calculate_metrics(y_eval, y_pred, class_names)
    
    # Print results
    print(f"\n🎯 Overall Results:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1-Score: {metrics['f1_score']:.4f}")
    
    if metrics['top_3_accuracy']:
        print(f"   Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
    if metrics['top_5_accuracy']:
        print(f"   Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")
    
    # Print detailed classification report
    print_classification_report(y_eval, y_pred, class_names)
    
    # Confidence analysis
    confidence_stats = calculate_confidence_metrics(y_pred)
    if confidence_stats:
        print(f"\n🎯 Confidence Analysis:")
        print(f"   Mean confidence: {confidence_stats['mean_confidence']:.4f}")
        print(f"   High confidence (>0.8): {confidence_stats['high_confidence_ratio']:.1%}")
        print(f"   Low confidence (<0.5): {confidence_stats['low_confidence_ratio']:.1%}")
        print(f"   Mean entropy: {confidence_stats['mean_entropy']:.4f}")
    
    # Misclassification analysis
    print(f"\n🔍 Misclassification Analysis:")
    misclass_analysis = analyze_misclassifications(y_eval, y_pred, class_names)
    
    # Class confusion analysis
    confusion_analysis = calculate_class_confusion(y_eval, y_pred, class_names)
    
    # Find best and worst performing classes
    class_accuracies = [(name, data['accuracy']) 
                       for name, data in metrics['per_class_metrics'].items()]
    class_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Best Performing Classes:")
    for name, acc in class_accuracies[:5]:
        print(f"   {name}: {acc:.4f}")
    
    print(f"\n⚠️  Worst Performing Classes:")
    for name, acc in class_accuracies[-5:]:
        print(f"   {name}: {acc:.4f}")
    
    # Model robustness test
    print(f"\n🧪 Testing Model Robustness...")
    robustness_results = evaluate_model_robustness(model, X_eval[:1000], y_eval[:1000])
    
    # Create visualizations
    if create_visualizations:
        print(f"\n📊 Creating visualizations...")
        
        model_name = os.path.basename(model_path).replace('.h5', '')
        
        # Confusion matrix
        cm_path = os.path.join(PATHS["plots_dir"], f"{model_name}_confusion_matrix.png")
        plot_confusion_matrix(y_eval, y_pred, class_names, cm_path)
        
        # Sample predictions
        pred_path = os.path.join(PATHS["plots_dir"], f"{model_name}_sample_predictions.png")
        plot_sample_predictions(X_eval[:100], y_eval[:100], y_pred[:100], class_names, pred_path)
        
        # Class accuracy
        class_acc_path = os.path.join(PATHS["plots_dir"], f"{model_name}_class_accuracy.png")
        plot_class_accuracy(y_eval, y_pred, class_names, class_acc_path)
        
        # Top-k accuracy
        topk_path = os.path.join(PATHS["plots_dir"], f"{model_name}_topk_accuracy.png")
        plot_top_k_accuracy(y_eval, y_pred, class_names, save_path=topk_path)
        
        print(f"✅ Visualizations saved to {PATHS['plots_dir']}")
    
    # Compile comprehensive results
    evaluation_results = {
        'model_path': model_path,
        'evaluation_set': eval_name,
        'model_parameters': int(model.count_params()),
        'evaluation_samples': len(X_eval),
        'metrics': metrics,
        'confidence_stats': confidence_stats,
        'misclassification_analysis': misclass_analysis,
        'confusion_analysis': confusion_analysis,
        'robustness_results': robustness_results,
        'best_classes': class_accuracies[:5],
        'worst_classes': class_accuracies[-5:]
    }
    
    # Save evaluation report
    report_path = os.path.join(PATHS["models_dir"], f"{model_name}_evaluation_report.json")
    save_evaluation_report(evaluation_results, report_path)
    
    print(f"\n🎉 Evaluation Complete!")
    print(f"📄 Detailed report saved: {report_path}")
    
    return evaluation_results

def compare_models(model_paths):
    """Compare multiple models"""
    print("🔍 Model Comparison")
    print("=" * 50)
    
    results = {}
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found: {model_path}")
            continue
        
        model_name = os.path.basename(model_path).replace('.h5', '')
        print(f"\n📊 Evaluating {model_name}...")
        
        try:
            result = evaluate_model(model_path, create_visualizations=False)
            results[model_name] = result
        except Exception as e:
            print(f"❌ Failed to evaluate {model_name}: {e}")
    
    # Create comparison table
    if results:
        print(f"\n📊 Model Comparison Summary:")
        print("-" * 80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Top-3':<10} {'Params':<12} {'Size (MB)':<10}")
        print("-" * 80)
        
        for model_name, result in results.items():
            accuracy = result['metrics']['accuracy']
            top3_acc = result['metrics'].get('top_3_accuracy', 0) or 0
            params = result['model_parameters']
            
            # Estimate model size (rough approximation)
            size_mb = params * 4 / (1024 * 1024)  # 4 bytes per parameter
            
            print(f"{model_name:<20} {accuracy:<10.4f} {top3_acc:<10.4f} {params:<12,} {size_mb:<10.1f}")
        
        # Save comparison report
        comparison_path = os.path.join(PATHS["models_dir"], "model_comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Comparison report saved: {comparison_path}")
    
    return results

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Doodle Recognition Model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.h5 file)')
    parser.add_argument('--train', action='store_true',
                       help='Evaluate on training set instead of test set')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization creation')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple models')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple models
        compare_models(args.compare)
    else:
        # Evaluate single model
        if not os.path.exists(args.model):
            print(f"❌ Model not found: {args.model}")
            return
        
        evaluate_model(
            args.model,
            test_on_train=args.train,
            create_visualizations=not args.no_viz
        )

if __name__ == "__main__":
    main()