import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image
import json

PROJECT_ROOT = Path(r"D:\School\python\AINailSys")
DATA_DIR = PROJECT_ROOT / "data" / "05_stage2_multiclass"  # ← FIXED: Use correct data
MODEL_PATH = PROJECT_ROOT / "models" / "stage2_multiclass" / "best_model.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage2_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Mapping abnormalities to deficiencies (NO HEALTHY!)
ABNORMALITY_TO_DEFICIENCY = {
    'spooning': 'Iron Deficiency',
    'onycholysis': 'Iron Deficiency',
    'onychorrhexis': 'Iron Deficiency',
    'beaus_lines': 'Folate Deficiency',
    'onychoschizia': 'Folate Deficiency',
    'melanonychia': 'B12 Deficiency',
    'blue_nails': 'B12 Deficiency'
}

def load_test_data(data_dir):
    """Load test dataset"""
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(
        data_dir / 'test',
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    return test_loader, test_dataset


def load_model(model_path, num_classes, device):
    """Load trained model"""
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_model(model, test_loader, device):
    """
    Evaluate model and get predictions with confidence scores
    
    Returns:
        dict: Contains predictions, confidences, labels, etc.
    """
    all_preds = []
    all_labels = []
    all_confidences = []
    all_probs = []
    
    print(" Evaluating model on test set...")
    print("=" * 80)
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Get predictions
            outputs = model(inputs)
            
            # Get probabilities (confidence scores)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Show progress
            if (batch_idx + 1) % 5 == 0:
                print(f"   Processed {(batch_idx + 1) * BATCH_SIZE} images...")
    
    results = {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'confidences': np.array(all_confidences),
        'probabilities': np.array(all_probs)
    }
    
    return results

def calculate_metrics(results, class_names):
    """Calculate and display metrics"""
    preds = results['predictions']
    labels = results['labels']
    confidences = results['confidences']
    
    # Overall accuracy
    accuracy = np.mean(preds == labels)
    
    # Average confidence
    avg_confidence = np.mean(confidences)
    
    # Confidence for correct vs incorrect predictions
    correct_mask = preds == labels
    correct_confidence = np.mean(confidences[correct_mask])
    incorrect_confidence = np.mean(confidences[~correct_mask]) if np.sum(~correct_mask) > 0 else 0
    
    # Per-class accuracy
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        class_mask = labels == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(preds[class_mask] == labels[class_mask])
            per_class_acc[class_name] = class_acc
    
    print("\n EVALUATION RESULTS")
    print("=" * 80)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total Test Images: {len(labels)}")
    print(f"Correct Predictions: {np.sum(correct_mask)}")
    print(f"Incorrect Predictions: {np.sum(~correct_mask)}")
    print(f"\nAverage Confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
    print(f"Confidence (Correct): {correct_confidence:.4f} ({correct_confidence*100:.2f}%)")
    if np.sum(~correct_mask) > 0:
        print(f"Confidence (Incorrect): {incorrect_confidence:.4f} ({incorrect_confidence*100:.2f}%)")
    
    # Per-class accuracy
    print("\n PER-CLASS ACCURACY")
    print("=" * 80)
    for class_name, acc in sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True):
        deficiency = ABNORMALITY_TO_DEFICIENCY.get(class_name, 'Unknown')
        print(f"{class_name:15} {acc:.4f} ({acc*100:.2f}%)  → {deficiency}")
    
    # Classification report
    print("\n DETAILED CLASSIFICATION REPORT")
    print("=" * 80)
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print(report)
    
    return {
        'accuracy': float(accuracy),
        'avg_confidence': float(avg_confidence),
        'correct_confidence': float(correct_confidence),
        'incorrect_confidence': float(incorrect_confidence),
        'total': int(len(labels)),
        'correct': int(np.sum(correct_mask)),
        'incorrect': int(np.sum(~correct_mask)),
        'per_class_acc': {k: float(v) for k, v in per_class_acc.items()}
    }

def calculate_deficiency_accuracy(results, class_names):
    """Calculate accuracy at deficiency level"""
    preds = results['predictions']
    labels = results['labels']
    
    # Map to deficiencies
    pred_deficiencies = [ABNORMALITY_TO_DEFICIENCY[class_names[p]] for p in preds]
    label_deficiencies = [ABNORMALITY_TO_DEFICIENCY[class_names[l]] for l in labels]
    
    # Calculate deficiency-level accuracy
    deficiency_correct = sum(p == l for p, l in zip(pred_deficiencies, label_deficiencies))
    deficiency_acc = deficiency_correct / len(labels)
    
    print("\n DEFICIENCY-LEVEL ACCURACY")
    print("=" * 80)
    print(f"Deficiency Accuracy: {deficiency_acc:.4f} ({deficiency_acc*100:.2f}%)")
    print(f"Correct Deficiency: {deficiency_correct}/{len(labels)}")

    
    # Per-deficiency breakdown
    print("\n Per-Deficiency Breakdown:")
    deficiency_types = list(set(ABNORMALITY_TO_DEFICIENCY.values()))
    
    for deficiency in sorted(deficiency_types):
        # Get indices for this deficiency
        true_indices = [i for i, l in enumerate(label_deficiencies) if l == deficiency]
        if len(true_indices) == 0:
            continue
        
        # Count correct predictions for this deficiency
        correct_count = sum(1 for i in true_indices if pred_deficiencies[i] == deficiency)
        acc = correct_count / len(true_indices)
        
        print(f"   {deficiency:20} {correct_count}/{len(true_indices):3} ({acc*100:.1f}%)")
    
    return float(deficiency_acc)

def plot_confusion_matrix(results, class_names, save_path):
    """Create confusion matrix"""
    cm = confusion_matrix(results['labels'], results['predictions'])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Stage 2 (7 Nail Abnormalities)', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n Confusion matrix saved to: {save_path}")
    plt.close()


def plot_per_class_performance(metrics, save_path):
    """Plot per-class accuracy with deficiency grouping"""
    per_class_acc = metrics['per_class_acc']
    
    # Sort by accuracy
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
    classes = [c[0] for c in sorted_classes]
    accs = [c[1] for c in sorted_classes]
    
    # Color by deficiency
    colors = []
    for class_name in classes:
        deficiency = ABNORMALITY_TO_DEFICIENCY[class_name]
        if 'Iron' in deficiency:
            colors.append('#e74c3c')  # Red
        elif 'Folate' in deficiency:
            colors.append('#3498db')  # Blue
        else:  # B12
            colors.append('#2ecc71')  # Green
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(classes)), accs, color=colors, edgecolor='black', linewidth=1.5)
    
    plt.xlabel('Nail Abnormality', fontsize=12, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    plt.title('Per-Class Accuracy - Stage 2 (7 Abnormalities)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    # Add accuracy labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Iron Deficiency'),
        Patch(facecolor='#3498db', label='Folate Deficiency'),
        Patch(facecolor='#2ecc71', label='B12 Deficiency')
    ]
    plt.legend(handles=legend_elements, loc='lower left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Per-class performance saved to: {save_path}")
    plt.close()


def plot_confidence_distribution(results, save_path):
    """Plot confidence score distribution"""
    confidences = results['confidences']
    correct_mask = results['predictions'] == results['labels']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Overall confidence distribution
    ax1.hist(confidences, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Confidence Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Correct vs Incorrect confidence
    ax2.hist(confidences[correct_mask], bins=30, color='green', 
             alpha=0.6, label=f'Correct ({np.sum(correct_mask)})', edgecolor='black')
    if np.sum(~correct_mask) > 0:
        ax2.hist(confidences[~correct_mask], bins=30, color='red', 
                 alpha=0.6, label=f'Incorrect ({np.sum(~correct_mask)})', edgecolor='black')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence: Correct vs Incorrect', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Confidence distribution saved to: {save_path}")
    plt.close()


def save_detailed_results(results, class_names, save_path):
    """Save detailed per-image results to JSON"""
    detailed_results = []
    
    for i in range(len(results['labels'])):
        true_label = class_names[results['labels'][i]]
        pred_label = class_names[results['predictions'][i]]
        
        detailed_results.append({
            'image_index': int(i),
            'true_label': true_label,
            'true_deficiency': ABNORMALITY_TO_DEFICIENCY[true_label],
            'predicted_label': pred_label,
            'predicted_deficiency': ABNORMALITY_TO_DEFICIENCY[pred_label],
            'confidence': float(results['confidences'][i]),
            'probabilities': {
                class_names[j]: float(results['probabilities'][i][j]) 
                for j in range(len(class_names))
            },
            'correct_abnormality': bool(results['predictions'][i] == results['labels'][i]),
            'correct_deficiency': ABNORMALITY_TO_DEFICIENCY[true_label] == ABNORMALITY_TO_DEFICIENCY[pred_label]
        })
    
    with open(save_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f" Detailed results saved to: {save_path}")


if __name__ == "__main__":
    print("=" * 80)
    print(" AINAILSYS - STAGE 2 MODEL EVALUATION")
    print("   7-Class Abnormality Classifier")
    print("=" * 80)
    print()
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f" Model not found at: {MODEL_PATH}")
        print("   Please train the model first!")
        exit()
    
    # Check if data exists
    if not DATA_DIR.exists():
        print(f" Data directory not found: {DATA_DIR}")
        print("   Please run: python scripts/09a_prepare_stage2_data.py")
        exit()
    
    print(f" Loading test data from: {DATA_DIR / 'test'}")
    test_loader, test_dataset = load_test_data(DATA_DIR)
    class_names = test_dataset.classes
    
    print(f"   Classes ({len(class_names)}): {class_names}")
    print(f"   Test images: {len(test_dataset)}")
    
    # Verify no 'healthy' class
    if 'healthy' in class_names:
        print("\n ERROR: 'healthy' found in test data!")
        print("   Stage 2 should only have 7 abnormality classes.")
        exit()
    
    print(f"   ✓ Correct! Only abnormality classes (no healthy)")
    print()
    
    print(f" Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH, len(class_names), DEVICE)
    print(f"   Device: {DEVICE}")
    print()
    
    # Evaluate
    results = evaluate_model(model, test_loader, DEVICE)
    
    # Calculate metrics
    metrics = calculate_metrics(results, class_names)
    
    # Calculate deficiency-level accuracy
    deficiency_acc = calculate_deficiency_accuracy(results, class_names)
    metrics['deficiency_accuracy'] = deficiency_acc
    
    # Create visualizations
    print("\n Creating visualizations...")
    plot_confusion_matrix(results, class_names, 
                         OUTPUT_DIR / 'confusion_matrix.png')
    
    plot_per_class_performance(metrics,
                              OUTPUT_DIR / 'per_class_accuracy.png')
    
    plot_confidence_distribution(results, 
                                OUTPUT_DIR / 'confidence_distribution.png')
    
    # Save detailed results
    print("\n Saving detailed results...")
    save_detailed_results(results, class_names, 
                         OUTPUT_DIR / 'detailed_predictions.json')
    
    # Save summary
    summary = {
        'model': 'ResNet18',
        'task': 'Multi-class Classification (7 Abnormalities - No Healthy)',
        'test_size': int(len(test_dataset)),
        'num_classes': len(class_names),
        'classes': class_names,
        'abnormality_to_deficiency': ABNORMALITY_TO_DEFICIENCY,
        'metrics': metrics
    }
    
    with open(OUTPUT_DIR / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f" Evaluation summary saved to: {OUTPUT_DIR / 'evaluation_summary.json'}")
    
    # Final summary
    print("\n" + "=" * 80)
    print(" EVALUATION COMPLETE!")
    print("=" * 80)
    
    print(f"\n Results saved in: {OUTPUT_DIR}")
    
    print("\n KEY METRICS:")
    print(f"   Abnormality Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   Deficiency Accuracy:  {deficiency_acc*100:.2f}%")
    print(f"   Average Confidence:   {metrics['avg_confidence']*100:.2f}%")
    
    print("\n" + "=" * 80)