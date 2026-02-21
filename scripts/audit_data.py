import os
from pathlib import Path

RAW_DATA_PATH = r"D:\School\python\AINailSys\data\raw"  

def count_files_detailed(folder_path):
    """
    Count files and show details
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder_path}")
        return
    
    print(f"\n📂 Scanning: {folder_path}")
    print("=" * 80)
    
    # Get all files
    all_files = []
    for item in folder_path.iterdir():
        if item.is_file():
            all_files.append(item)
    
    # Group by extension
    extensions = {}
    for file in all_files:
        ext = file.suffix.lower()
        if ext not in extensions:
            extensions[ext] = []
        extensions[ext].append(file.name)
    
    # Print results
    print(f"\n📊 Total files found: {len(all_files)}")
    print("\nBreakdown by extension:")
    print("-" * 80)
    
    for ext, files in sorted(extensions.items()):
        print(f"\n{ext if ext else '(no extension)'}: {len(files)} files")
        
        # Show first 5 filenames as examples
        print("  Examples:")
        for filename in files[:5]:
            print(f"    - {filename}")
        if len(files) > 5:
            print(f"    ... and {len(files) - 5} more")
    
    return len(all_files)


def scan_all_folders(raw_path):
    """
    Scan all folders in the structure
    """
    raw_path = Path(raw_path)
    
    if not raw_path.exists():
        print(f"❌ Raw folder not found: {raw_path}")
        return
    
    print("🔍 DETAILED FILE COUNT REPORT")
    print("=" * 80)
    
    total = 0
    
    # Check healthy folder
    healthy_path = raw_path / "healthy"
    if healthy_path.exists():
        count = count_files_detailed(healthy_path)
        total += count
        print(f"\n✅ Healthy total: {count}")
    else:
        print(f"\n❌ 'healthy' folder not found")
    
    # Check iron folder
    iron_path = raw_path / "iron"
    if iron_path.exists():
        print("\n" + "=" * 80)
        print("IRON FOLDER")
        print("=" * 80)
        
        iron_total = 0
        
        for subfolder in ['spooning', 'onycholysis', 'onychorrhexis']:
            subfolder_path = iron_path / subfolder
            if subfolder_path.exists():
                count = count_files_detailed(subfolder_path)
                iron_total += count
                print(f"\n✅ {subfolder} total: {count}")
            else:
                print(f"\n❌ '{subfolder}' subfolder not found in iron/")
        
        total += iron_total
        print(f"\n📦 IRON GRAND TOTAL: {iron_total}")
    else:
        print(f"\n❌ 'iron' folder not found")
    
    # Check folate folder
    folate_path = raw_path / "folate"
    if folate_path.exists():
        print("\n" + "=" * 80)
        print("FOLATE FOLDER")
        print("=" * 80)
        
        folate_total = 0
        
        for subfolder in ['beaus_lines', 'onychoschizia']:
            subfolder_path = folate_path / subfolder
            if subfolder_path.exists():
                count = count_files_detailed(subfolder_path)
                folate_total += count
                print(f"\n✅ {subfolder} total: {count}")
            else:
                print(f"\n❌ '{subfolder}' subfolder not found in folate/")
        
        total += folate_total
        print(f"\n📦 FOLATE GRAND TOTAL: {folate_total}")
    else:
        print(f"\n❌ 'folate' folder not found")
    
    # Check b12 folder
    b12_path = raw_path / "b12"
    if b12_path.exists():
        print("\n" + "=" * 80)
        print("B12 FOLDER")
        print("=" * 80)
        
        b12_total = 0
        
        for subfolder in ['melanonychia', 'blue_nails']:
            subfolder_path = b12_path / subfolder
            if subfolder_path.exists():
                count = count_files_detailed(subfolder_path)
                b12_total += count
                print(f"\n✅ {subfolder} total: {count}")
            else:
                print(f"\n❌ '{subfolder}' subfolder not found in b12/")
        
        total += b12_total
        print(f"\n📦 B12 GRAND TOTAL: {b12_total}")
    else:
        print(f"\n❌ 'b12' folder not found")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎯 FINAL SUMMARY")
    print("=" * 80)
    print(f"\n🔢 TOTAL FILES ACROSS ALL FOLDERS: {total}")
    print("\nExpected counts:")
    print("  healthy: 685")
    print("  spooning: 336")
    print("  onycholysis: 249")
    print("  onychorrhexis: 93")
    print("  beaus_lines: 600")
    print("  onychoschizia: 50")
    print("  melanonychia: 207")
    print("  blue_nails: 455")
    print("  ─────────────")
    print("  EXPECTED TOTAL: 2675")
    print(f"\n  ACTUAL TOTAL: {total}")
    print("")


if __name__ == "__main__":
    scan_all_folders(RAW_DATA_PATH)