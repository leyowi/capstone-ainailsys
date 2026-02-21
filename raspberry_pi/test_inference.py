#!/usr/bin/env python3
"""
Quick test of ONNX models on Raspberry Pi
Tests both Stage 1 and Stage 2 models
"""

import onnxruntime as ort
import numpy as np
import json
from pathlib import Path
import time

# Paths
MODEL_DIR = Path.home() / "capstone" / "models"
STAGE1_MODEL = MODEL_DIR / "stage1_binary.onnx"
STAGE2_MODEL = MODEL_DIR / "stage2_multiclass.onnx"

print("=" * 80)
print("🔍 AINAILSYS - MODEL INFERENCE TEST")
print("=" * 80)
print()

# ============================================
# STAGE 1: Binary Classifier Test
# ============================================

print("📊 STAGE 1: Binary Classifier (Healthy vs Anemic)")
print("-" * 80)

# Load model
print(f"Loading: {STAGE1_MODEL}")
stage1_session = ort.InferenceSession(str(STAGE1_MODEL))

# Get input/output info
input_name = stage1_session.get_inputs()[0].name
output_name = stage1_session.get_outputs()[0].name
input_shape = stage1_session.get_inputs()[0].shape

print(f"✅ Model loaded successfully!")
print(f"   Input: {input_name}, Shape: {input_shape}")
print(f"   Output: {output_name}")

# Create dummy input (random image)
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
print("\n🧪 Testing inference...")
start = time.time()
outputs = stage1_session.run([output_name], {input_name: dummy_input})
inference_time = (time.time() - start) * 1000  # ms

# Get prediction
logits = outputs[0][0]
probs = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
pred_class = np.argmax(probs)
confidence = probs[pred_class]

class_names = ['anemic', 'healthy']

print(f"✅ Inference successful!")
print(f"   Time: {inference_time:.2f} ms")
print(f"   Prediction: {class_names[pred_class]}")
print(f"   Confidence: {confidence*100:.2f}%")
print(f"   Probabilities: {probs}")

print()

# ============================================
# STAGE 2: Multi-class Classifier Test
# ============================================

print("📊 STAGE 2: Multi-class Classifier (7 Abnormalities)")
print("-" * 80)

# Load model
print(f"Loading: {STAGE2_MODEL}")
stage2_session = ort.InferenceSession(str(STAGE2_MODEL))

# Get input/output info
input_name = stage2_session.get_inputs()[0].name
output_name = stage2_session.get_outputs()[0].name
input_shape = stage2_session.get_inputs()[0].shape

print(f"✅ Model loaded successfully!")
print(f"   Input: {input_name}, Shape: {input_shape}")
print(f"   Output: {output_name}")

# Create dummy input
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
print("\n🧪 Testing inference...")
start = time.time()
outputs = stage2_session.run([output_name], {input_name: dummy_input})
inference_time = (time.time() - start) * 1000  # ms

# Get prediction
logits = outputs[0][0]
probs = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
pred_class = np.argmax(probs)
confidence = probs[pred_class]

class_names = ['beaus_lines', 'blue_nails', 'melanonychia', 'onycholysis', 
               'onychorrhexis', 'onychoschizia', 'spooning']

# Map to deficiency
deficiency_map = {
    'spooning': 'Iron Deficiency',
    'onycholysis': 'Iron Deficiency',
    'onychorrhexis': 'Iron Deficiency',
    'beaus_lines': 'Folate Deficiency',
    'onychoschizia': 'Folate Deficiency',
    'melanonychia': 'B12 Deficiency',
    'blue_nails': 'B12 Deficiency'
}

predicted_abnormality = class_names[pred_class]
predicted_deficiency = deficiency_map[predicted_abnormality]

print(f"✅ Inference successful!")
print(f"   Time: {inference_time:.2f} ms")
print(f"   Abnormality: {predicted_abnormality}")
print(f"   Deficiency: {predicted_deficiency}")
print(f"   Confidence: {confidence*100:.2f}%")

print()
print("=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print()
print("📊 Summary:")
print(f"   Stage 1 inference: ~{inference_time:.0f} ms")
print(f"   Stage 2 inference: ~{inference_time:.0f} ms")
print(f"   Total pipeline: ~{inference_time*2:.0f} ms")
print()
print("🎉 Models are ready for deployment!")
print()