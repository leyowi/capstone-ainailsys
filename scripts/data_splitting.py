import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

PROJECT_ROOT = Path(r"D:\School\python\AINailSys")
SOURCE = PROJECT_ROOT / "data" / "02_augmented"
DESTINATION = PROJECT_ROOT / "data" / "03_splits"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility 
RANDOM_SEED = 42


def split_files(files, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Split list of files into train/val/test
    
    Args:
        files: List of file paths
        train_ratio: Proportion for training (e.g., 0.70)
        val_ratio: Proportion for validation (e.g., 0.15)
        test_ratio: Proportion for test (e.g., 0.15)
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_files, val_files, test_files)
    """
    # Set random seed
    random.seed(seed)
    
    # Shuffle files
    files_copy = files.copy()
    random.shuffle(files_copy)
    
    # Calculate split indices
    total = len(files_copy)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split
    train_files = files_copy[:train_end]
    val_files = files_copy[train_end:val_end]
    test_files = files_copy[val_end:]
    
    return train_files, val_files, test_files


def create_splits(src, dst, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Create train/val/test splits for all classes
    
    Args:
        src: Source folder (02_augmented)
        dst: Destination folder (03_splits)
        train_ratio: Train proportion
        val_ratio: Validation proportion
        test_ratio: Test proportion
        seed: Random seed
    
    Returns:
        dict: Statistics
    """
    src = Path(src)
    dst = Path(dst)
    
    stats = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'test': defaultdict(int),
        'total': defaultdict(int)
    }
    
    print("Creating train/validation/test splits...")
    print("=" * 80)
    
    # Process each class
    for class_folder in sorted(src.iterdir()):
        if not class_folder.is_dir():
            continue
        
        class_name = class_folder.name
        print(f"\nProcessing: {class_name}")
        
        # Get all images
        all_images = list(class_folder.glob('*.jpg'))
        total_count = len(all_images)
        
        if total_count == 0:
            print(f"   No images found, skipping")
            continue
        
        print(f"   Total images: {total_count}")
        
        # Split files
        train_files, val_files, test_files = split_files(
            all_images,
            train_ratio,
            val_ratio,
            test_ratio,
            seed
        )
        
        print(f"   Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
        # Create destination folders
        train_dst = dst / 'train' / class_name
        val_dst = dst / 'val' / class_name
        test_dst = dst / 'test' / class_name
        
        train_dst.mkdir(parents=True, exist_ok=True)
        val_dst.mkdir(parents=True, exist_ok=True)
        test_dst.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        print(f"   Copying files...")
        
        # Copy train files
        for file_path in train_files:
            shutil.copy2(file_path, train_dst / file_path.name)
        
        # Copy validation files
        for file_path in val_files:
            shutil.copy2(file_path, val_dst / file_path.name)
        
        # Copy test files
        for file_path in test_files:
            shutil.copy2(file_path, test_dst / file_path.name)
        
        # Update stats
        stats['train'][class_name] = len(train_files)
        stats['val'][class_name] = len(val_files)
        stats['test'][class_name] = len(test_files)
        stats['total'][class_name] = total_count
        
        print(f"   ✓ Completed")
    
    return stats


def verify_splits(dst):
    """
    Verify splits and show summary
    
    Args:
        dst: Splits folder
    
    Returns:
        dict: Counts per split
    """
    dst = Path(dst)
    
    print("\nVerifying splits...")
    print("=" * 80)
    
    results = {}
    
    for split in ['train', 'val', 'test']:
        split_path = dst / split
        
        if not split_path.exists():
            print(f"{split}/ folder not found")
            continue
        
        print(f"\n{split.upper()}:")
        print("-" * 80)
        
        split_total = 0
        class_counts = {}
        
        for class_folder in sorted(split_path.iterdir()):
            if class_folder.is_dir():
                class_name = class_folder.name
                count = len(list(class_folder.glob('*.jpg')))
                class_counts[class_name] = count
                split_total += count
                print(f"   {class_name:15} {count:4} images")
        
        print(f"   {'-' * 76}")
        print(f"   TOTAL:          {split_total:4} images")
        
        results[split] = {
            'by_class': class_counts,
            'total': split_total
        }
    
    return results


def print_summary(stats, verify_results):
    """
    Print comprehensive summary
    
    Args:
        stats: Statistics from splitting
        verify_results: Verification results
    """
    print("\n" + "=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)
    
    # Overall totals
    train_total = verify_results['train']['total']
    val_total = verify_results['val']['total']
    test_total = verify_results['test']['total']
    grand_total = train_total + val_total + test_total
    
    print(f"\nOverall Distribution:")
    print(f"   Train:      {train_total:4} images ({train_total/grand_total*100:.1f}%)")
    print(f"   Validation: {val_total:4} images ({val_total/grand_total*100:.1f}%)")
    print(f"   Test:       {test_total:4} images ({test_total/grand_total*100:.1f}%)")
    print(f"   {'─' * 40}")
    print(f"   TOTAL:      {grand_total:4} images")
    
    # Per-class distribution
    print(f"\nPer-Class Distribution:")
    print("-" * 80)
    print(f"   {'Class':<15} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 80)
    
    for class_name in sorted(stats['total'].keys()):
        train_count = stats['train'][class_name]
        val_count = stats['val'][class_name]
        test_count = stats['test'][class_name]
        total_count = stats['total'][class_name]
        
        print(f"   {class_name:<15} {train_count:<8} {val_count:<8} {test_count:<8} {total_count:<8}")
    
    print("-" * 80)
    print(f"   {'TOTAL':<15} {train_total:<8} {val_total:<8} {test_total:<8} {grand_total:<8}")

if __name__ == "__main__":
    print("=" * 80)
    print("AINAILSYS - CREATE DATA SPLITS")
    print("=" * 80)
    print()
    
    # Show configuration
    print("Configuration:")
    print(f"   Source:      {SOURCE}")
    print(f"   Destination: {DESTINATION}")
    print(f"   Split ratio: {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val / {TEST_RATIO:.0%} test")
    print(f"   Random seed: {RANDOM_SEED}")
    print()
    
    # Check if splits exist
    if (DESTINATION / 'train').exists():
        print("Splits folder already exists")
        response = input("Do you want to recreate splits? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("Operation cancelled.")
            exit()
        
        print("Removing old splits...")
        shutil.rmtree(DESTINATION)
        print("   ✓ Old splits removed\n")
    
    # Create splits
    stats = create_splits(
        SOURCE,
        DESTINATION,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        RANDOM_SEED
    )
    
    # Verify
    verify_results = verify_splits(DESTINATION)
    
    # Print summary
    print_summary(stats, verify_results)
    
    # Final message
    print("\n" + "=" * 80)
    print("SPLITS CREATED SUCCESSFULLY!")
    print("=" * 80)
    
    print("\nYour data is ready in:")
    print(f"   {DESTINATION / 'train'}")
    print(f"   {DESTINATION / 'val'}")
    print(f"   {DESTINATION / 'test'}")
    
    print("\n" + "=" * 80)