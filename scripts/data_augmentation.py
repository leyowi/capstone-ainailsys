import os
from pathlib import Path
from PIL import Image, ImageEnhance
import random
import shutil


PROJECT_ROOT = Path(r"D:\School\python\AINailSys")
SOURCE = PROJECT_ROOT / "data" / "01_cleaned"
DESTINATION = PROJECT_ROOT / "data" / "02_augmented"

TARGET_COUNT = 550

AUGMENTATION_CONFIG = {
    'rotation_range': 15,          # Rotate ±15 degrees
    'brightness_range': (0.8, 1.2), # 80% to 120% brightness
    'contrast_range': (0.8, 1.2),   # 80% to 120% contrast
    'flip_horizontal': True,        # Allow horizontal flip
    'zoom_range': (0.9, 1.1)       # 90% to 110% zoom
}

def rotate_image(img, angle):
    """Rotate image by given angle"""
    return img.rotate(angle, fillcolor='white', expand=False)


def adjust_brightness(img, factor):
    """Adjust image brightness"""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def adjust_contrast(img, factor):
    """Adjust image contrast"""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def flip_horizontal(img):
    """Flip image horizontally"""
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def zoom_image(img, factor):
    """Zoom in/out on image"""
    width, height = img.size
    new_width = int(width * factor)
    new_height = int(height * factor)
    
    # Resize
    zoomed = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Crop or pad to original size
    if factor > 1:  # Zoom in 
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        zoomed = zoomed.crop((left, top, left + width, top + height))
    else:  # Zoom out 
        background = Image.new('RGB', (width, height), (255, 255, 255))
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        background.paste(zoomed, (left, top))
        zoomed = background
    
    return zoomed


def augment_image(img, config):

    # Start with original
    aug_img = img.copy()
    
    # Random rotation
    if random.random() > 0.5:
        angle = random.uniform(-config['rotation_range'], config['rotation_range'])
        aug_img = rotate_image(aug_img, angle)
    
    # Random brightness
    if random.random() > 0.5:
        factor = random.uniform(*config['brightness_range'])
        aug_img = adjust_brightness(aug_img, factor)
    
    # Random contrast
    if random.random() > 0.5:
        factor = random.uniform(*config['contrast_range'])
        aug_img = adjust_contrast(aug_img, factor)
    
    # Random horizontal flip
    if config['flip_horizontal'] and random.random() > 0.5:
        aug_img = flip_horizontal(aug_img)
    
    # Random zoom
    if random.random() > 0.5:
        factor = random.uniform(*config['zoom_range'])
        aug_img = zoom_image(aug_img, factor)
    
    return aug_img


def process_class(class_name, src_path, dst_path, current_count, target_count, config):

    # Create destination folder
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Get all source images
    source_images = list(src_path.glob('*.jpg'))
    
    if len(source_images) == 0:
        return {'copied': 0, 'augmented': 0}
    
    print(f"\n📁 {class_name}:")
    print(f"   Current: {current_count} images")
    print(f"   Target:  {target_count} images")
    
    # Copy all original images first
    print(f"   Copying originals...")
    for i, img_path in enumerate(source_images):
        dst_img_path = dst_path / img_path.name
        shutil.copy2(img_path, dst_img_path)
    
    copied_count = len(source_images)
    print(f"   ✓ Copied {copied_count} original images")
    
    # Calculate how many augmented images we need
    needed = target_count - current_count
    
    if needed <= 0:
        print(f"   ✓ No augmentation needed (already at target)")
        return {'copied': copied_count, 'augmented': 0}
    
    print(f"   Creating {needed} augmented images...")
    
    # Create augmented images
    augmented_count = 0
    
    for i in range(needed):
        # Pick random source image
        source_img_path = random.choice(source_images)
        
        # Load image
        img = Image.open(source_img_path)
        
        # Augment
        aug_img = augment_image(img, config)
        
        # Save with unique name
        base_name = source_img_path.stem
        aug_name = f"{base_name}_aug_{i:04d}.jpg"
        aug_path = dst_path / aug_name
        
        aug_img.save(aug_path, 'JPEG', quality=95)
        augmented_count += 1
        
        # Show progress
        if (i + 1) % 50 == 0:
            print(f"      ... {i + 1}/{needed} augmented images created")
    
    print(f"   ✓ Created {augmented_count} augmented images")
    print(f"   ✓ Total: {copied_count + augmented_count} images")
    
    return {'copied': copied_count, 'augmented': augmented_count}


def augment_dataset(src, dst, target_count, config):

    src = Path(src)
    dst = Path(dst)
    
    # Get all class folders
    class_folders = [f for f in src.iterdir() if f.is_dir()]
    
    stats = {}
    total_copied = 0
    total_augmented = 0
    
    print("🔄 Augmenting dataset...")
    print("=" * 80)
    
    for class_folder in sorted(class_folders):
        class_name = class_folder.name
        src_path = src / class_name
        dst_path = dst / class_name
        
        # Count current images
        current_count = len(list(src_path.glob('*.jpg')))
        
        # Process class
        class_stats = process_class(
            class_name,
            src_path,
            dst_path,
            current_count,
            target_count,
            config
        )
        
        stats[class_name] = class_stats
        total_copied += class_stats['copied']
        total_augmented += class_stats['augmented']
    
    return {
        'by_class': stats,
        'total_copied': total_copied,
        'total_augmented': total_augmented,
        'total_final': total_copied + total_augmented
    }


def verify_augmentation(dst):

    dst = Path(dst)
    
    print("\n🔍 Verifying augmentation...")
    print("-" * 80)
    
    counts = {}
    
    for class_folder in sorted(dst.iterdir()):
        if class_folder.is_dir():
            class_name = class_folder.name
            count = len(list(class_folder.glob('*.jpg')))
            counts[class_name] = count
            print(f"   {class_name:15} {count:4} images")
    
    return counts


if __name__ == "__main__":
    print("=" * 80)
    print("🔄 AINAILSYS - DATA AUGMENTATION")
    print("=" * 80)
    print()
    
    # Show configuration
    print("⚙️  Configuration:")
    print(f"   Source:       {SOURCE}")
    print(f"   Destination:  {DESTINATION}")
    print(f"   Target count: {TARGET_COUNT} images per class")
    print(f"   Augmentation:")
    print(f"      - Rotation:   ±{AUGMENTATION_CONFIG['rotation_range']}°")
    print(f"      - Brightness: {AUGMENTATION_CONFIG['brightness_range']}")
    print(f"      - Contrast:   {AUGMENTATION_CONFIG['contrast_range']}")
    print(f"      - H-Flip:     {AUGMENTATION_CONFIG['flip_horizontal']}")
    print(f"      - Zoom:       {AUGMENTATION_CONFIG['zoom_range']}")
    print()
    
    # Augment dataset
    stats = augment_dataset(SOURCE, DESTINATION, TARGET_COUNT, AUGMENTATION_CONFIG)
    
    # Verify
    final_counts = verify_augmentation(DESTINATION)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ AUGMENTATION COMPLETE!")
    print("=" * 80)
    
    print(f"\n📊 Statistics:")
    print(f"   Original images copied: {stats['total_copied']}")
    print(f"   Augmented images added: {stats['total_augmented']}")
    print(f"   Total final images:     {stats['total_final']}")
    
    print(f"\n📁 Final counts per class:")
    for class_name, count in sorted(final_counts.items()):
        print(f"   {class_name:15} {count:4} images")
    
    print("\n" + "=" * 80)