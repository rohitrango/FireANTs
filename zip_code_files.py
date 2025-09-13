#!/usr/bin/env python3
import os
import zipfile
import argparse
from pathlib import Path

def should_include_file(file_path):
    """Check if file should be included based on extension."""
    allowed_extensions = {'.py', '.cu', '.h', '.cpp', '.ipynb', '.txt'}
    return file_path.suffix.lower() in allowed_extensions

def zip_code_files(source_dir, output_zip):
    """Create zip file with only code and text files."""
    source_path = Path(source_dir)
    
    # First, collect all files that will be included
    files_to_zip = []
    for file_path in source_path.rglob('*'):
        if file_path.is_file() and should_include_file(file_path):
            arcname = file_path.relative_to(source_path)
            files_to_zip.append((file_path, arcname))
    
    # Show files that will be zipped
    print(f"Found {len(files_to_zip)} files to zip:")
    for _, arcname in files_to_zip:
        print(f"  {arcname}")
    
    # Ask for confirmation
    response = input(f"\nProceed to create {output_zip}? (y/N): ").strip().lower()
    if response not in ('y', 'yes'):
        print("Operation cancelled.")
        return
    
    # Create the zip file
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path, arcname in files_to_zip:
            zipf.write(file_path, arcname)
            print(f"Added: {arcname}")
    
    print(f"\nZip file created: {output_zip}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zip only code and text files")
    parser.add_argument("source", nargs="?", default=".", help="Source directory (default: current directory)")
    parser.add_argument("-o", "--output", default="code_files.zip", help="Output zip file name")
    
    args = parser.parse_args()
    
    zip_code_files(args.source, args.output)