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
DATA_DIR = PROJECT_ROOT / "data" / "04_stage1_binary"
MODEL_PATH = PROJECT_ROOT / "models" / "stage1_binary" / "best_model.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage1_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    model.load_state_dict(torch.load(model_path))
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
    
    print("Evaluating model on test set...")
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
    
    # Classification report
    print("\n CLASSIFICATION REPORT")
    print("=" * 80)
    report = classification_report(labels, preds, target_names=class_names)
    print(report)
    
    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'correct_confidence': correct_confidence,
        'incorrect_confidence': incorrect_confidence,
        'total': len(labels),
        'correct': int(np.sum(correct_mask)),
        'incorrect': int(np.sum(~correct_mask))
    }


def plot_confusion_matrix(results, class_names, save_path):
    """Create confusion matrix"""
    cm = confusion_matrix(results['labels'], results['predictions'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Stage 1 (Healthy vs Anemic)', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n Confusion matrix saved to: {save_path}")
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
             alpha=0.6, label='Correct Predictions', edgecolor='black')
    ax2.hist(confidences[~correct_mask], bins=30, color='red', 
             alpha=0.6, label='Incorrect Predictions', edgecolor='black')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence: Correct vs Incorrect', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Confidence distribution saved to: {save_path}")
    plt.close()


def show_sample_predictions(model, test_dataset, class_names, device, save_path, num_samples=16):
    """Show sample predictions with images and confidence scores"""
    model.eval()
    
    # Get random samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    for idx, ax in enumerate(axes):
        img_idx = indices[idx]
        img_path = test_dataset.imgs[img_idx][0]
        true_label = test_dataset.imgs[img_idx][1]
        
        # Load and display original image
        img_display = Image.open(img_path).convert('RGB')
        
        # Prepare for model
        img_tensor = test_dataset.transform(img_display).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)
            
            pred = pred.item()
            confidence = confidence.item()
            probs_values = probs[0].cpu().numpy()
        
        # Display
        ax.imshow(img_display)
        ax.axis('off')
        
        # Color: green if correct, red if wrong
        color = 'green' if pred == true_label else 'red'
        
        # Title with prediction and confidence
        title = f"True: {class_names[true_label]}\n"
        title += f"Pred: {class_names[pred]} ({confidence*100:.1f}%)"
        
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        
        # Add probability text
        prob_text = f"{class_names[0]}: {probs_values[0]*100:.1f}%\n"
        prob_text += f"{class_names[1]}: {probs_values[1]*100:.1f}%"
        ax.text(0.02, 0.98, prob_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Sample Predictions with Confidence Scores', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Sample predictions saved to: {save_path}")
    plt.close()


def save_detailed_results(results, class_names, save_path):
    """Save detailed per-image results to JSON"""
    detailed_results = []
    
    for i in range(len(results['labels'])):
        detailed_results.append({
            'image_index': i,
            'true_label': class_names[results['labels'][i]],
            'predicted_label': class_names[results['predictions'][i]],
            'confidence': float(results['confidences'][i]),
            'probabilities': {
                class_names[j]: float(results['probabilities'][i][j]) 
                for j in range(len(class_names))
            },
            'correct': bool(results['predictions'][i] == results['labels'][i])
        })
    
    with open(save_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f" Detailed results saved to: {save_path}")



if __name__ == "__main__":
    print("=" * 80)
    print(" AINAILSYS - STAGE 1 MODEL EVALUATION")
    print("=" * 80)
    print()
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f" Model not found at: {MODEL_PATH}")
        print("   Please train the model first!")
        exit()
    
    print(f" Loading test data from: {DATA_DIR / 'test'}")
    test_loader, test_dataset = load_test_data(DATA_DIR)
    class_names = test_dataset.classes
    
    print(f"   Classes: {class_names}")
    print(f"   Test images: {len(test_dataset)}")
    print()
    
    print(f" Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH, len(class_names), DEVICE)
    print(f"   Device: {DEVICE}")
    print()
    
    # Evaluate
    results = evaluate_model(model, test_loader, DEVICE)
    
    # Calculate metrics
    metrics = calculate_metrics(results, class_names)
    
    # Create visualizations
    print("\n Creating visualizations...")
    plot_confusion_matrix(results, class_names, 
                         OUTPUT_DIR / 'confusion_matrix.png')
    
    plot_confidence_distribution(results, 
                                OUTPUT_DIR / 'confidence_distribution.png')
    
    show_sample_predictions(model, test_dataset, class_names, DEVICE,
                           OUTPUT_DIR / 'sample_predictions.png')
    
    # Save detailed results
    print("\n Saving detailed results...")
    save_detailed_results(results, class_names, 
                         OUTPUT_DIR / 'detailed_predictions.json')
    
    # Save summary (FIXED)
    summary = {
        'model': 'ResNet18',
        'task': 'Binary Classification (Healthy vs Anemic)',
        'test_size': int(len(test_dataset)),
        'classes': class_names,
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'avg_confidence': float(metrics['avg_confidence']),
            'correct_confidence': float(metrics['correct_confidence']),
            'incorrect_confidence': float(metrics['incorrect_confidence']),
            'total': int(metrics['total']),
            'correct': int(metrics['correct']),
            'incorrect': int(metrics['incorrect'])
        }
    }
    
    with open(OUTPUT_DIR / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f" Evaluation summary saved to: {OUTPUT_DIR / 'evaluation_summary.json'}")
    
    # Final summary
    print("\n" + "=" * 80)
    print(" EVALUATION COMPLETE!")
    print("=" * 80)
    
    print(f"\n Results saved in: {OUTPUT_DIR}")
    
    print("\n To see confidence scores:")
    print(f"   Open: {OUTPUT_DIR / 'detailed_predictions.json'}")
    
    print("\n" + "=" * 80)