import shutil
from pathlib import Path


PROJECT_ROOT = Path(r"D:\School\python\AINailSys")
SOURCE = PROJECT_ROOT / "data" / "03_splits"
DESTINATION = PROJECT_ROOT / "data" / "05_stage2_multiclass"


ABNORMALITY_CLASSES = [
    'spooning',
    'onycholysis',
    'onychorrhexis',
    'beaus_lines',
    'onychoschizia',
    'melanonychia',
    'blue_nails'
]


def prepare_stage2_data(src, dst, classes):
    """
    Copy only abnormality classes (exclude healthy)
    
    Args:
        src: Source folder (03_splits)
        dst: Destination folder (05_stage2_multiclass)
        classes: List of classes to include
    
    Returns:
        dict: Statistics
    """
    src = Path(src)
    dst = Path(dst)
    
    stats = {}
    
    print(" Preparing Stage 2 data (7 abnormalities only)...")
    print("=" * 80)
    
    for split in ['train', 'val', 'test']:
        print(f"\n Processing {split.upper()} split:")
        
        src_split = src / split
        dst_split = dst / split
        
        if not src_split.exists():
            print(f"     {split} folder not found, skipping")
            continue
        
        split_stats = {}
        split_total = 0
        
        # Copy each abnormality class
        for class_name in classes:
            src_class = src_split / class_name
            dst_class = dst_split / class_name
            
            if not src_class.exists():
                print(f"     {class_name} not found, skipping")
                continue
            
            # Create destination
            dst_class.mkdir(parents=True, exist_ok=True)
            
            # Copy all images
            images = list(src_class.glob('*.jpg'))
            for img in images:
                shutil.copy2(img, dst_class / img.name)
            
            count = len(images)
            split_stats[class_name] = count
            split_total += count
            
            print(f"   ✓ {class_name:15} {count:4} images")
        
        print(f"   {'─' * 76}")
        print(f"   TOTAL:          {split_total:4} images")
        
        stats[split] = split_stats
    
    return stats


def verify_data(dst):
    """Verify the prepared data"""
    dst = Path(dst)
    
    print("\n Verifying Stage 2 data...")
    print("=" * 80)
    
    for split in ['train', 'val', 'test']:
        split_path = dst / split
        
        if not split_path.exists():
            print(f"  {split} folder not found")
            continue
        
        print(f"\n {split.upper()}:")
        
        # Check for healthy (should NOT exist)
        if (split_path / 'healthy').exists():
            print("    ERROR: 'healthy' folder found! (should not be here)")
        else:
            print("   ✓ No 'healthy' folder (correct!)")
        
        # Count classes
        class_folders = [f for f in split_path.iterdir() if f.is_dir()]
        print(f"   ✓ Number of classes: {len(class_folders)}")
        
        if len(class_folders) != 7:
            print(f"     Expected 7 classes, found {len(class_folders)}")



if __name__ == "__main__":
    print("=" * 80)
    print(" AINAILSYS - PREPARE STAGE 2 DATA (7 CLASSES ONLY)")
    print("=" * 80)
    print()
    
    print("  Configuration:")
    print(f"   Source:      {SOURCE}")
    print(f"   Destination: {DESTINATION}")
    print(f"   Classes to include:")
    for cls in ABNORMALITY_CLASSES:
        print(f"      - {cls}")
    print(f"   Classes to EXCLUDE:")
    print(f"      - healthy (filtered by Stage 1)")
    print()
    
    # Check if destination exists
    if DESTINATION.exists() and any(DESTINATION.iterdir()):
        print("  Stage 2 data folder already exists")
        response = input("Do you want to recreate it? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print(" Operation cancelled.")
            exit()
        
        print("  Removing old data...")
        shutil.rmtree(DESTINATION)
        print("   ✓ Removed\n")
    
    # Prepare data
    stats = prepare_stage2_data(SOURCE, DESTINATION, ABNORMALITY_CLASSES)
    
    # Verify
    verify_data(DESTINATION)
    
    # Summary
    print("\n" + "=" * 80)
    print(" STAGE 2 DATA READY!")
    print("=" * 80)
    
    print("\n Summary:")
    total_train = sum(stats['train'].values())
    total_val = sum(stats['val'].values())
    total_test = sum(stats['test'].values())
    
    print(f"\n   TRAIN: {total_train} images (7 abnormality classes)")
    print(f"   VAL:   {total_val} images (7 abnormality classes)")
    print(f"   TEST:  {total_test} images (7 abnormality classes)")
    print(f"   TOTAL: {total_train + total_val + total_test} images")
    
    
    print("\n" + "=" * 80)