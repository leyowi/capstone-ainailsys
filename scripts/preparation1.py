import shutil
from pathlib import Path

PROJECT_ROOT = Path(r"D:\School\python\AINailSys")
SOURCE = PROJECT_ROOT / "data" / "03_splits"
DESTINATION = PROJECT_ROOT / "data" / "04_stage1_binary"

# Classes to combine as "anemic"
ANEMIC_CLASSES = [
    'spooning',
    'onycholysis',
    'onychorrhexis',
    'beaus_lines',
    'onychoschizia',
    'melanonychia',
    'blue_nails'
]


def combine_anemic_classes(src_split, dst_split):
    """
    Combine all anemic classes into one 'anemic' folder
    
    Args:
        src_split: Source split path (e.g., 03_splits/train)
        dst_split: Destination split path (e.g., 04_stage1_binary/train)
    
    Returns:
        dict: Statistics
    """
    src_split = Path(src_split)
    dst_split = Path(dst_split)
    
    stats = {
        'healthy': 0,
        'anemic': 0
    }
    
    # Copy healthy images (stays the same)
    healthy_src = src_split / 'healthy'
    healthy_dst = dst_split / 'healthy'
    
    if healthy_src.exists():
        healthy_dst.mkdir(parents=True, exist_ok=True)
        
        for img in healthy_src.glob('*.jpg'):
            shutil.copy2(img, healthy_dst / img.name)
            stats['healthy'] += 1
    
    # Combine all anemic classes
    anemic_dst = dst_split / 'anemic'
    anemic_dst.mkdir(parents=True, exist_ok=True)
    
    for class_name in ANEMIC_CLASSES:
        class_src = src_split / class_name
        
        if not class_src.exists():
            continue
        
        # Copy images with prefix to avoid name conflicts
        for img in class_src.glob('*.jpg'):
            # Add class prefix to filename
            new_name = f"{class_name}_{img.name}"
            shutil.copy2(img, anemic_dst / new_name)
            stats['anemic'] += 1
    
    return stats


def prepare_stage1_data(src, dst):
    """
    Prepare all splits for Stage 1
    
    Args:
        src: Source folder (03_splits)
        dst: Destination folder (04_stage1_binary)
    
    Returns:
        dict: Statistics for all splits
    """
    src = Path(src)
    dst = Path(dst)
    
    print("🔄 Preparing Stage 1 binary classification data...")
    print("=" * 80)
    
    all_stats = {}
    
    for split in ['train', 'val', 'test']:
        print(f"\n📂 Processing {split.upper()} split:")
        
        src_split = src / split
        dst_split = dst / split
        
        if not src_split.exists():
            print(f"   ⚠️  {split} folder not found, skipping")
            continue
        
        stats = combine_anemic_classes(src_split, dst_split)
        all_stats[split] = stats
        
        print(f"   ✓ Healthy: {stats['healthy']} images")
        print(f"   ✓ Anemic:  {stats['anemic']} images")
        print(f"   ✓ Total:   {stats['healthy'] + stats['anemic']} images")
    
    return all_stats


if __name__ == "__main__":
    print("=" * 80)
    print("🔄 AINAILSYS - PREPARE STAGE 1 DATA (BINARY)")
    print("=" * 80)
    print()
    
    print("⚙️  Configuration:")
    print(f"   Source:      {SOURCE}")
    print(f"   Destination: {DESTINATION}")
    print(f"   Classes:")
    print(f"      - healthy (stays as is)")
    print(f"      - anemic (combines: {', '.join(ANEMIC_CLASSES)})")
    print()
    
    # Check if destination exists
    if DESTINATION.exists() and any(DESTINATION.iterdir()):
        print("⚠️  Stage 1 data folder already exists")
        response = input("Do you want to recreate it? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("❌ Operation cancelled.")
            exit()
        
        print("🗑️  Removing old data...")
        shutil.rmtree(DESTINATION)
        print("   ✓ Removed\n")
    
    # Prepare data
    stats = prepare_stage1_data(SOURCE, DESTINATION)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ STAGE 1 DATA READY!")
    print("=" * 80)
    
    print("\n📊 Summary:")
    for split, split_stats in stats.items():
        total = split_stats['healthy'] + split_stats['anemic']
        healthy_pct = (split_stats['healthy'] / total * 100) if total > 0 else 0
        anemic_pct = (split_stats['anemic'] / total * 100) if total > 0 else 0
        
        print(f"\n   {split.upper()}:")
        print(f"      Healthy: {split_stats['healthy']:4} images ({healthy_pct:.1f}%)")
        print(f"      Anemic:  {split_stats['anemic']:4} images ({anemic_pct:.1f}%)")
        print(f"      Total:   {total:4} images")
    
    print("\n💡 Data structure created:")
    print(f"   {DESTINATION / 'train' / 'healthy'}")
    print(f"   {DESTINATION / 'train' / 'anemic'}")
    print(f"   {DESTINATION / 'val' / 'healthy'}")
    print(f"   {DESTINATION / 'val' / 'anemic'}")
    print(f"   {DESTINATION / 'test' / 'healthy'}")
    print(f"   {DESTINATION / 'test' / 'anemic'}")
    
    
    print("\n" + "=" * 80)