from pathlib import Path
import os

PROJECT_ROOT = Path(r"D:\School\python\AINailSys")

FOLDERS = [
    "data/00_original/healthy",
    "data/00_original/iron/spooning",
    "data/00_original/iron/onycholysis",
    "data/00_original/iron/onychorrhexis",
    "data/00_original/folate/beaus_lines",
    "data/00_original/folate/onychoschizia",
    "data/00_original/b12/melanonychia",
    "data/00_original/b12/blue_nails",
    
    "data/01_cleaned/healthy",
    "data/01_cleaned/spooning",
    "data/01_cleaned/onycholysis",
    "data/01_cleaned/onychorrhexis",
    "data/01_cleaned/beaus_lines",
    "data/01_cleaned/onychoschizia",
    "data/01_cleaned/melanonychia",
    "data/01_cleaned/blue_nails",
    
    "data/02_augmented/healthy",
    "data/02_augmented/spooning",
    "data/02_augmented/onycholysis",
    "data/02_augmented/onychorrhexis",
    "data/02_augmented/beaus_lines",
    "data/02_augmented/onychoschizia",
    "data/02_augmented/melanonychia",
    "data/02_augmented/blue_nails",
    
    "data/03_splits/train/healthy",
    "data/03_splits/train/spooning",
    "data/03_splits/train/onycholysis",
    "data/03_splits/train/onychorrhexis",
    "data/03_splits/train/beaus_lines",
    "data/03_splits/train/onychoschizia",
    "data/03_splits/train/melanonychia",
    "data/03_splits/train/blue_nails",
    
    "data/03_splits/val/healthy",
    "data/03_splits/val/spooning",
    "data/03_splits/val/onycholysis",
    "data/03_splits/val/onychorrhexis",
    "data/03_splits/val/beaus_lines",
    "data/03_splits/val/onychoschizia",
    "data/03_splits/val/melanonychia",
    "data/03_splits/val/blue_nails",
    
    "data/03_splits/test/healthy",
    "data/03_splits/test/spooning",
    "data/03_splits/test/onycholysis",
    "data/03_splits/test/onychorrhexis",
    "data/03_splits/test/beaus_lines",
    "data/03_splits/test/onychoschizia",
    "data/03_splits/test/melanonychia",
    "data/03_splits/test/blue_nails",
    
    "models/stage1_binary/checkpoints",
    "models/stage2_multiclass/checkpoints",
    "models/deployment",
    
    "outputs/reports",
    "outputs/visualizations",
    
    "raspberry_pi/models",
    
    "documentation",
]

# Create all folders
print("Creating folders...")
for folder in FOLDERS:
    full_path = PROJECT_ROOT / folder
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ {folder}")

print(f"\nDone. Created {len(FOLDERS)} folders")