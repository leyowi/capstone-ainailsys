import shutil
from pathlib import Path

PROJECT_ROOT = Path(r"D:\School\python\AINailSys")
SOURCE = PROJECT_ROOT / "data" / "raw"
DESTINATION = PROJECT_ROOT / "data" / "00_original"

def copy_folder_structure(src, dst):

    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        print(f"Source not found: {src}")
        return
    
    print(f"Copying from: {src}")
    print(f"Copying to:   {dst}\n")
    
    total_files = 0
    total_folders = 0
    
    # Walk through source directory
    for item in src.rglob('*'):
        # Calculate relative path
        relative_path = item.relative_to(src)
        destination_path = dst / relative_path
        
        if item.is_dir():
            # Create directory
            destination_path.mkdir(parents=True, exist_ok=True)
            total_folders += 1
            print(f"Created folder: {relative_path}")
        
        elif item.is_file():
            # Create parent directory if needed
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(item, destination_path)
            total_files += 1
            
            # Show progress every 50 files
            if total_files % 50 == 0:
                print(f"   ... copied {total_files} files so far")
    
    return total_files, total_folders


def verify_backup(src, dst):
    src = Path(src)
    dst = Path(dst)
    
    print("\nVerifying backup...")
    
    # Count files in source
    src_files = list(src.rglob('*'))
    src_file_count = sum(1 for f in src_files if f.is_file())
    
    # Count files in destination
    dst_files = list(dst.rglob('*'))
    dst_file_count = sum(1 for f in dst_files if f.is_file())
    
    print(f"   Source files:      {src_file_count}")
    print(f"   Destination files: {dst_file_count}")
    
    if src_file_count == dst_file_count:
        print("   File counts match!")
        return True
    else:
        print("   File counts don't match!")
        return False



if __name__ == "__main__":
    print("=" * 80)
    print("AINAILSYS - BACKUP RAW DATA")
    print("=" * 80)
    print()
    
    # Check if backup already exists
    if (DESTINATION / "healthy").exists():
        print("Backup already exists in 00_original/")
        response = input("Do you want to overwrite it? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("Backup cancelled.")
            exit()
        else:
            print("Removing old backup...")
            shutil.rmtree(DESTINATION)
            print("Old backup removed\n")
    
    # Copy data
    print("Starting backup...\n")
    files_copied, folders_created = copy_folder_structure(SOURCE, DESTINATION)
    
    # Verify
    success = verify_backup(SOURCE, DESTINATION)
    
    # Summary
    print("\n" + "=" * 80)
    if success:
        print("BACKUP COMPLETE!")
    else:
        print(" BACKUP COMPLETED WITH WARNINGS")
    print("=" * 80)
    
    print(f"\nSummary:")
    print(f"   Files copied:    {files_copied}")
    print(f"   Folders created: {folders_created}")
    print(f"   Source:          {SOURCE}")
    print(f"   Destination:     {DESTINATION}")
    
    print("\n" + "=" * 80)