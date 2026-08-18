import shutil
from pathlib import Path
from PIL import Image
import os

PROJECT_ROOT = Path(r"D:\School\python\AINailSys")
SOURCE = PROJECT_ROOT / "data" / "00_original"
DESTINATION = PROJECT_ROOT / "data" / "01_cleaned"

# Target image size (set to None to keep original size)
TARGET_SIZE = (512, 512)  # Resize to 512x512, or set to None

# JPEG quality (1-100, higher = better quality but larger file)
JPEG_QUALITY = 95

# Mapping: source_path -> destination_folder
REORGANIZE_MAP = {
    'healthy': 'healthy',
    'iron/spooning': 'spooning',
    'iron/onycholysis': 'onycholysis',
    'iron/onychorrhexis': 'onychorrhexis',
    'folate/beaus_lines': 'beaus_lines',
    'folate/onychoschizia': 'onychoschizia',
    'b12/melanonychia': 'melanonychia',
    'b12/blue_nails': 'blue_nails'
}


def convert_to_jpg(image_path, output_path, target_size=None, quality=95):
    """
    Convert any image to JPG format
    
    Args:
        image_path: Path to source image
        output_path: Path to save JPG
        target_size: Tuple (width, height) or None to keep original
        quality: JPEG quality (1-100)
    
    Returns:
        bool: True if successful
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary (PNG might have alpha channel)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if requested
        if target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Save as JPG
        img.save(output_path, 'JPEG', quality=quality)
        
        return True
        
    except Exception as e:
        print(f"   Error converting {image_path.name}: {e}")
        return False


def reorganize_and_clean(src, dst, reorganize_map, target_size=None):
    """
    Reorganize folder structure and convert images
    
    Args:
        src: Source folder (00_original)
        dst: Destination folder (01_cleaned)
        reorganize_map: Dictionary mapping source paths to destination folders
        target_size: Target image size or None
    
    Returns:
        dict: Statistics
    """
    src = Path(src)
    dst = Path(dst)
    
    stats = {
        'total_processed': 0,
        'total_success': 0,
        'total_failed': 0,
        'by_class': {}
    }
    
    print(" Reorganizing and cleaning data...\n")
    
    # Process each mapping
    for source_path, dest_folder in reorganize_map.items():
        source_full = src / source_path
        dest_full = dst / dest_folder
        
        # Create destination folder
        dest_full.mkdir(parents=True, exist_ok=True)
        
        if not source_full.exists():
            print(f"  Skipping {source_path} - folder not found")
            continue
        
        print(f" Processing: {source_path} → {dest_folder}")
        
        # Count files in source
        files = [f for f in source_full.iterdir() if f.is_file()]
        total_files = len(files)
        
        success_count = 0
        failed_count = 0
        
        # Process each file
        for i, file_path in enumerate(files, 1):
            # Show progress
            if i % 50 == 0:
                print(f"   ... {i}/{total_files} files processed")
            
            # Generate output filename (always .jpg)
            output_filename = file_path.stem + '.jpg'
            output_path = dest_full / output_filename
            
            # Convert and save
            if convert_to_jpg(file_path, output_path, target_size, JPEG_QUALITY):
                success_count += 1
            else:
                failed_count += 1
        
        # Update stats
        stats['total_processed'] += total_files
        stats['total_success'] += success_count
        stats['total_failed'] += failed_count
        stats['by_class'][dest_folder] = {
            'total': total_files,
            'success': success_count,
            'failed': failed_count
        }
        
        print(f"    Completed: {success_count}/{total_files} successful")
        if failed_count > 0:
            print(f"     Failed: {failed_count} files")
        print()
    
    return stats


def verify_reorganization(dst, expected_counts):
    """
    Verify reorganization was successful
    
    Args:
        dst: Destination folder
        expected_counts: Dictionary of expected file counts per class
    
    Returns:
        bool: True if verification passed
    """
    dst = Path(dst)
    
    print(" Verifying reorganization...\n")
    
    all_match = True
    
    for class_name, expected in expected_counts.items():
        class_path = dst / class_name
        
        if not class_path.exists():
            print(f"    {class_name}: Folder not found")
            all_match = False
            continue
        
        actual = len(list(class_path.glob('*.jpg')))
        
        if actual == expected:
            print(f"    {class_name}: {actual} files (matches expected)")
        else:
            print(f"     {class_name}: {actual} files (expected {expected})")
            all_match = False
    
    return all_match


if __name__ == "__main__":
    print("=" * 80)
    print("  AINAILSYS - REORGANIZE & CLEAN DATA")
    print("=" * 80)
    print()
    
    # Show configuration
    print("  Configuration:")
    print(f"   Source:       {SOURCE}")
    print(f"   Destination:  {DESTINATION}")
    print(f"   Target size:  {TARGET_SIZE if TARGET_SIZE else 'Keep original'}")
    print(f"   JPEG quality: {JPEG_QUALITY}")
    print()
    
    # Check if destination exists
    if any(DESTINATION.iterdir()) if DESTINATION.exists() else False:
        print("  01_cleaned/ folder is not empty")
        response = input("Do you want to continue? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print(" Operation cancelled.")
            exit()
        print()
    
    # Process
    stats = reorganize_and_clean(SOURCE, DESTINATION, REORGANIZE_MAP, TARGET_SIZE)
    
    # Verify (expected counts based on your audit)
    expected_counts = {
        'healthy': 685,
        'spooning': 336,
        'onycholysis': 249,
        'onychorrhexis': 93,
        'beaus_lines': 600,
        'onychoschizia': 50,
        'melanonychia': 207,
        'blue_nails': 455
    }
    
    print()
    verified = verify_reorganization(DESTINATION, expected_counts)
    
    # Summary
    print("\n" + "=" * 80)
    if stats['total_failed'] == 0 and verified:
        print(" REORGANIZATION COMPLETE!")
    else:
        print("  REORGANIZATION COMPLETED WITH WARNINGS")
    print("=" * 80)
    
    print(f"\n Statistics:")
    print(f"   Total files processed: {stats['total_processed']}")
    print(f"   Successful:            {stats['total_success']}")
    print(f"   Failed:                {stats['total_failed']}")
    
    print(f"\n By class:")
    for class_name, class_stats in stats['by_class'].items():
        print(f"   {class_name:15} {class_stats['success']:4} files")

    print("   ✓ Flattened folder structure (deficiency → abnormality)")
    print("   ✓ Converted all images to .jpg format")
    if TARGET_SIZE:
        print(f"   ✓ Resized all images to {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print("   ✓ Standardized JPEG quality")
    
    print("\n" + "=" * 80)