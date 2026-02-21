#!/usr/bin/env python3
"""
AINAILSYS - Camera Capture + Inference Pipeline
Captures from USB webcam and runs two-stage classification
"""

import cv2
import onnxruntime as ort
import numpy as np
from pathlib import Path
import time
import json

# ============================================
# CONFIGURATION
# ============================================

MODEL_DIR = Path.home() / "capstone" / "models"
STAGE1_MODEL = MODEL_DIR / "stage1_binary.onnx"
STAGE2_MODEL = MODEL_DIR / "stage2_multiclass.onnx"

# Load metadata
with open(MODEL_DIR / "stage1_binary.json") as f:
    stage1_meta = json.load(f)

with open(MODEL_DIR / "stage2_multiclass.json") as f:
    stage2_meta = json.load(f)

# Deficiency mapping
DEFICIENCY_MAP = {
    'spooning': 'Iron Deficiency',
    'onycholysis': 'Iron Deficiency',
    'onychorrhexis': 'Iron Deficiency',
    'beaus_lines': 'Folate Deficiency',
    'onychoschizia': 'Folate Deficiency',
    'melanonychia': 'B12 Deficiency',
    'blue_nails': 'B12 Deficiency'
}

# ============================================
# MODEL LOADING
# ============================================

print("=" * 80)
print("🔍 AINAILSYS - Camera Inference System")
print("=" * 80)
print()

print("📦 Loading models...")
stage1_session = ort.InferenceSession(str(STAGE1_MODEL))
stage2_session = ort.InferenceSession(str(STAGE2_MODEL))
print("✅ Models loaded successfully!")
print()

# ============================================
# PREPROCESSING
# ============================================

def preprocess_image(image):
    """
    Preprocess image for model inference
    
    Args:
        image: OpenCV image (BGR format)
    
    Returns:
        tensor: Preprocessed image tensor
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224
    image_resized = cv2.resize(image_rgb, (224, 224))
    
    # Convert to float and normalize [0, 255] -> [0, 1]
    image_float = image_resized.astype(np.float32) / 255.0
    
    # Normalize using ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_normalized = (image_float - mean) / std
    
    # Transpose to NCHW format (batch, channels, height, width)
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    
    # Add batch dimension
    image_batch = np.expand_dims(image_transposed, axis=0)
    
    return image_batch

# ============================================
# INFERENCE FUNCTIONS
# ============================================

def run_stage1(image_tensor):
    """Run Stage 1: Binary classification"""
    input_name = stage1_session.get_inputs()[0].name
    output_name = stage1_session.get_outputs()[0].name
    
    # Run inference
    outputs = stage1_session.run([output_name], {input_name: image_tensor})
    
    # Get probabilities
    logits = outputs[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    pred_class = np.argmax(probs)
    confidence = probs[pred_class]
    
    class_names = stage1_meta['class_names']
    
    return {
        'prediction': class_names[pred_class],
        'confidence': float(confidence),
        'probabilities': {name: float(prob) for name, prob in zip(class_names, probs)}
    }

def run_stage2(image_tensor):
    """Run Stage 2: Multi-class classification"""
    input_name = stage2_session.get_inputs()[0].name
    output_name = stage2_session.get_outputs()[0].name
    
    # Run inference
    outputs = stage2_session.run([output_name], {input_name: image_tensor})
    
    # Get probabilities
    logits = outputs[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    pred_class = np.argmax(probs)
    confidence = probs[pred_class]
    
    class_names = stage2_meta['class_names']
    predicted_abnormality = class_names[pred_class]
    predicted_deficiency = DEFICIENCY_MAP[predicted_abnormality]
    
    return {
        'abnormality': predicted_abnormality,
        'deficiency': predicted_deficiency,
        'confidence': float(confidence),
        'probabilities': {name: float(prob) for name, prob in zip(class_names, probs)}
    }

# ============================================
# MAIN INFERENCE PIPELINE
# ============================================

def analyze_nail(image):
    """
    Complete two-stage analysis pipeline
    
    Args:
        image: OpenCV image
    
    Returns:
        dict: Analysis results
    """
    start_time = time.time()
    
    # Preprocess
    image_tensor = preprocess_image(image)
    preprocess_time = (time.time() - start_time) * 1000
    
    # Stage 1: Healthy vs Anemic
    stage1_start = time.time()
    stage1_result = run_stage1(image_tensor)
    stage1_time = (time.time() - stage1_start) * 1000
    
    result = {
        'stage1': stage1_result,
        'preprocessing_time': preprocess_time,
        'stage1_time': stage1_time
    }
    
    # If anemic, run Stage 2
    if stage1_result['prediction'] == 'anemic':
        stage2_start = time.time()
        stage2_result = run_stage2(image_tensor)
        stage2_time = (time.time() - stage2_start) * 1000
        
        result['stage2'] = stage2_result
        result['stage2_time'] = stage2_time
        result['total_time'] = preprocess_time + stage1_time + stage2_time
    else:
        result['total_time'] = preprocess_time + stage1_time
    
    return result

# ============================================
# CAMERA CAPTURE
# ============================================

def main():
    """Main function with camera capture"""
    
    # Open webcam
    print("📷 Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam!")
        print("   Make sure USB webcam is connected")
        return
    
    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Webcam opened successfully!")
    print()
    print("=" * 80)
    print("📸 INSTRUCTIONS:")
    print("   - Position fingernail in front of camera")
    print("   - Press SPACE to capture and analyze")
    print("   - Press Q to quit")
    print("=" * 80)
    print()
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to capture frame")
            break
        
        # Display frame
        cv2.imshow('AINAILSYS - Camera Feed (Press SPACE to analyze, Q to quit)', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Exiting...")
            break
        
        elif key == ord(' '):  # Space bar
            print("\n" + "=" * 80)
            print("🔍 ANALYZING IMAGE...")
            print("=" * 80)
            
            # Run analysis
            results = analyze_nail(frame)
            
            # Display results
            print(f"\n📊 STAGE 1 RESULTS:")
            print(f"   Classification: {results['stage1']['prediction'].upper()}")
            print(f"   Confidence: {results['stage1']['confidence']*100:.2f}%")
            print(f"   Inference time: {results['stage1_time']:.1f} ms")
            
            if 'stage2' in results:
                print(f"\n📊 STAGE 2 RESULTS:")
                print(f"   Abnormality: {results['stage2']['abnormality']}")
                print(f"   Deficiency: {results['stage2']['deficiency']}")
                print(f"   Confidence: {results['stage2']['confidence']*100:.2f}%")
                print(f"   Inference time: {results['stage2_time']:.1f} ms")
            
            print(f"\n⏱️  PERFORMANCE:")
            print(f"   Preprocessing: {results['preprocessing_time']:.1f} ms")
            print(f"   Total time: {results['total_time']:.1f} ms")
            
            print("\n" + "=" * 80)
            print("Press SPACE for another analysis, Q to quit")
            print("=" * 80)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Camera released. Goodbye!")

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    main()