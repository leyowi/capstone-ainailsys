# AINAILSYS - AI-Powered Anemia Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-c51a4a)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-powered system for non-invasive anemia detection through fingernail image analysis, deployed on Raspberry Pi 5 for portable, accessible healthcare screening.

## Project Overview

AINAILSYS uses deep learning to detect anemia and identify specific nutrient deficiencies (Iron, Folate, B12) by analyzing fingernail images. The system achieves 97.45% accuracy in deficiency type classification and runs entirely on edge hardware, making it suitable for resource-limited settings.

## Key Features

- **Two-Stage Architecture**: Binary classification (Healthy/Anemic) followed by multi-class abnormality detection
- **High Accuracy**: 98.41% Stage 1, 96.43% Stage 2, 97.45% deficiency detection
- **Edge Deployment**: Optimized for Raspberry Pi 5 with ONNX Runtime
- **Real-Time Inference**: ~98ms total inference time
- **Portable**: Standalone device with 7" touchscreen interface
- **No Internet Required**: Fully offline operation

## System Architecture
```
┌─────────────────┐
│  Camera Input   │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Stage 1 │ (Healthy vs Anemic)
    │ ResNet18│ → 98.41% accuracy
    └────┬────┘
         │
    ┌────▼────┐
    │ Stage 2 │ (7 Abnormalities)
    │ ResNet18│ → 96.43% accuracy
    └────┬────┘
         │
    ┌────▼────────────┐
    │ Deficiency Type │
    │   Iron/Folate/  │
    │      B12        │
    └─────────────────┘
```

## Performance Metrics

| Metric | Stage 1 | Stage 2 | Deficiency |
|--------|---------|---------|------------|
| **Accuracy** | 98.41% | 96.43% | 97.45% |
| **Inference Time** | ~49ms | ~49ms | - |
| **Model Size** | 0.10 MB | 0.10 MB | - |

### Per-Class Accuracy (Stage 2)
- Onychoschizia: 100.00%
- Onycholysis: 98.80%
- Melanonychia: 97.59%
- Blue Nails: 96.39%
- Onychorrhexis: 96.39%
- Beau's Lines: 94.44%
- Spooning: 91.57%

## Technology Stack

### Training
- **Framework**: PyTorch 2.x
- **Architecture**: ResNet18 (Transfer Learning)
- **Optimization**: Adam optimizer
- **Data**: 4,585 augmented images across 8 classes

### Deployment
- **Hardware**: Raspberry Pi 5 (8GB RAM)
- **Display**: 7" DSI Touchscreen
- **Camera**: USB Webcam
- **Runtime**: ONNX Runtime
- **Interface**: Python Tkinter GUI

## Project Structure
```
ainailsys/
├── data/                   # Dataset organization
├── models/                 # Trained models (PyTorch & ONNX)
├── scripts/               # Training & evaluation scripts
├── raspberry_pi/          # RPi deployment code
├── docs/                  # Documentation
└── outputs/               # Results & visualizations
```

## Quick Start

### Training (PC)
```bash
# Clone repository
git clone https://github.com/leyowi/ainailsys.git
cd ainailsys

# Install dependencies
pip install -r requirements.txt

# Train models
python scripts/training/03_train_stage1.py
python scripts/training/04_train_stage2.py

# Export to ONNX
python scripts/training/05_export_onnx.py
```

### Deployment (Raspberry Pi)
```bash
# Install dependencies
pip3 install -r raspberry_pi/requirements.txt --break-system-packages

# Transfer models to RPi
scp models/deployment/*.onnx pi@raspberrypi.local:~/ainailsys/models/

# Run GUI
python3 raspberry_pi/ainailsys_gui.py
```

## Research Context

This project was developed as a thesis for Polytechnic University of the Philippines, Bachelor of Science in Electronics Engineering. The goal is to create an accessible, affordable anemia screening tool for use in resource-limited healthcare settings.

**Academic Year**: 2025-2026

## Dataset

- **Total Images**: 4,585 (after augmentation)
- **Original Images**: 2,675
- **Classes**: 8 (healthy + 7 abnormalities)
- **Split**: 70% train, 15% validation, 15% test
- **Augmentation**: Rotation, flip, color jitter, affine transforms

## Future Work

- [ ] Integrate object detection for improved nail localization
- [ ] Expand dataset with more diverse skin tones
- [ ] Add support for multiple deficiency combinations
- [ ] Develop mobile app version (Android/iOS)
- [ ] Clinical validation study with dermatologists

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ResNet18 architecture from [PyTorch torchvision](https://pytorch.org/vision/stable/models.html)
- Dataset sourced from [Kaggle](https://www.kaggle.com/) and [Roboflow](https://roboflow.com/)
- Inspired by research in non-invasive health screening

---
