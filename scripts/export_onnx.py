import torch
import torch.onnx
from torchvision import models
import torch.nn as nn
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np

PROJECT_ROOT = Path(r"D:\School\python\AINailSys")

# Model paths
STAGE1_MODEL_PATH = PROJECT_ROOT / "models" / "stage1_binary" / "best_model.pth"
STAGE2_MODEL_PATH = PROJECT_ROOT / "models" / "stage2_multiclass" / "best_model.pth"

# Output paths
DEPLOYMENT_DIR = PROJECT_ROOT / "models" / "deployment"
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_ONNX_PATH = DEPLOYMENT_DIR / "stage1_binary.onnx"
STAGE2_ONNX_PATH = DEPLOYMENT_DIR / "stage2_multiclass.onnx"

# Model settings
IMAGE_SIZE = 224
STAGE1_NUM_CLASSES = 2  # healthy, anemic
STAGE2_NUM_CLASSES = 7  # 7 abnormalities

def load_pytorch_model(model_path, num_classes):
    """
    Load PyTorch model
    
    Args:
        model_path: Path to .pth file
        num_classes: Number of output classes
    
    Returns:
        model: Loaded PyTorch model
    """
    # Create model architecture
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    
    return model

def export_to_onnx(model, onnx_path, model_name, num_classes):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model
        onnx_path: Output path for ONNX file
        model_name: Name for display
        num_classes: Number of output classes
    """
    print(f"\n Converting {model_name} to ONNX...")
    print("-" * 80)
    
    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    
    # Class names for metadata
    if num_classes == 2:
        class_names = ['anemic', 'healthy']
    else:
        class_names = ['beaus_lines', 'blue_nails', 'melanonychia', 
                      'onycholysis', 'onychorrhexis', 'onychoschizia', 'spooning']
    
    # Export to ONNX
    torch.onnx.export(
        model,                          # Model to export
        dummy_input,                    # Model input (or a tuple for multiple inputs)
        onnx_path,                      # Where to save the model
        export_params=True,             # Store trained parameter weights
        opset_version=14,               # ONNX version to export to
        do_constant_folding=True,       # Optimize constant folding
        input_names=['input'],          # Input tensor name
        output_names=['output'],        # Output tensor name
        dynamic_axes={
            'input': {0: 'batch_size'},  # Variable batch size
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"   ✓ Exported to: {onnx_path}")
    
    # Check ONNX model
    print(f"   Checking ONNX model validity...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"   ✓ ONNX model is valid!")
    
    # Get model size
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"   Model size: {size_mb:.2f} MB")
    
    return class_names

def test_onnx_inference(onnx_path, model_name, class_names):
    """
    Test ONNX model inference
    
    Args:
        onnx_path: Path to ONNX model
        model_name: Name for display
        class_names: List of class names
    """
    print(f"\n Testing {model_name} ONNX inference...")
    print("-" * 80)
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(str(onnx_path))
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"   Input name: {input_name}")
    print(f"   Output name: {output_name}")
    print(f"   Input shape: {session.get_inputs()[0].shape}")
    print(f"   Output shape: {session.get_outputs()[0].shape}")
    
    # Create dummy input
    dummy_input = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    
    # Run inference
    import time
    start = time.time()
    outputs = session.run([output_name], {input_name: dummy_input})
    inference_time = (time.time() - start) * 1000  # Convert to ms
    
    print(f"\n   ✓ Inference successful!")
    print(f"   Inference time: {inference_time:.2f} ms")
    print(f"   Output shape: {outputs[0].shape}")
    
    # Get prediction
    logits = outputs[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
    pred_class = np.argmax(probs)
    confidence = probs[pred_class]
    
    print(f"\n   Sample prediction:")
    print(f"      Predicted class: {class_names[pred_class]}")
    print(f"      Confidence: {confidence*100:.2f}%")
    print(f"      All probabilities:")
    for i, (name, prob) in enumerate(zip(class_names, probs)):
        print(f"         {name:15} {prob*100:6.2f}%")

def save_model_metadata(onnx_path, class_names, stage_name):
    """
    Save model metadata (class names, preprocessing info)
    
    Args:
        onnx_path: Path to ONNX model
        class_names: List of class names
        stage_name: Stage identifier
    """
    metadata = {
        'model_name': f'AINAILSYS - {stage_name}',
        'framework': 'PyTorch → ONNX',
        'architecture': 'ResNet18',
        'input_shape': [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        'input_name': 'input',
        'output_name': 'output',
        'num_classes': len(class_names),
        'class_names': class_names,
        'preprocessing': {
            'resize': [IMAGE_SIZE, IMAGE_SIZE],
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225],
            'input_range': [0, 1],  # After ToTensor
            'color_mode': 'RGB'
        },
        'postprocessing': {
            'output_type': 'logits',
            'apply_softmax': True
        }
    }
    
    # Save as JSON
    import json
    metadata_path = onnx_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("🔧 AINAILSYS - EXPORT MODELS TO ONNX")
    print("=" * 80)
    print()
    
    print(" Export Configuration:")
    print(f"   Stage 1 PyTorch: {STAGE1_MODEL_PATH}")
    print(f"   Stage 2 PyTorch: {STAGE2_MODEL_PATH}")
    print(f"   Output directory: {DEPLOYMENT_DIR}")
    print(f"   Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print()
    
    # ==========================================
    # STAGE 1 EXPORT
    # ==========================================
    
    print("=" * 80)
    print("STAGE 1: Binary Classifier (Healthy vs Anemic)")
    print("=" * 80)
    
    # Load PyTorch model
    print("\n Loading Stage 1 PyTorch model...")
    stage1_model = load_pytorch_model(STAGE1_MODEL_PATH, STAGE1_NUM_CLASSES)
    print("   ✓ Model loaded successfully")
    
    # Export to ONNX
    stage1_classes = export_to_onnx(
        stage1_model, 
        STAGE1_ONNX_PATH, 
        "Stage 1",
        STAGE1_NUM_CLASSES
    )
    
    # Save metadata
    save_model_metadata(STAGE1_ONNX_PATH, stage1_classes, "Stage 1 Binary")
    
    # Test inference
    test_onnx_inference(STAGE1_ONNX_PATH, "Stage 1", stage1_classes)
    
    
    print("\n" + "=" * 80)
    print("STAGE 2: Multi-class Classifier (7 Abnormalities)")
    print("=" * 80)
    
    # Load PyTorch model
    print("\n Loading Stage 2 PyTorch model...")
    stage2_model = load_pytorch_model(STAGE2_MODEL_PATH, STAGE2_NUM_CLASSES)
    print("   ✓ Model loaded successfully")
    
    # Export to ONNX
    stage2_classes = export_to_onnx(
        stage2_model, 
        STAGE2_ONNX_PATH, 
        "Stage 2",
        STAGE2_NUM_CLASSES
    )
    
    # Save metadata
    save_model_metadata(STAGE2_ONNX_PATH, stage2_classes, "Stage 2 Multiclass")
    
    # Test inference
    test_onnx_inference(STAGE2_ONNX_PATH, "Stage 2", stage2_classes)
    
    
    print("\n" + "=" * 80)
    print(" EXPORT COMPLETE!")
    print("=" * 80)
    
    print(f"\n Exported Models:")
    print(f"   Stage 1: {STAGE1_ONNX_PATH}")
    print(f"   Stage 2: {STAGE2_ONNX_PATH}")
    
    print(f"\n Metadata Files:")
    print(f"   Stage 1: {STAGE1_ONNX_PATH.with_suffix('.json')}")
    print(f"   Stage 2: {STAGE2_ONNX_PATH.with_suffix('.json')}")
    
    print("\n" + "=" * 80)