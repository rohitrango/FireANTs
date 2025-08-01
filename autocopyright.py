#!/usr/bin/env python3

import os
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

COPYRIGHT_NOTICE = '''# Copyright (c) {year} Rohit Jena. All rights reserved.
# 
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.
#
# IMPORTANT: This code is part of FireANTs and its use, reproduction, or
# distribution must comply with the full license terms, including:
# - Maintaining all copyright notices and bibliography references
# - Using only approved (re)-distribution channels 
# - Proper attribution in derivative works
#
# For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE \n\n
'''

def check_copyright(content: str) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    Check if file has a copyright notice and extract its position and year if present.
    Returns (has_copyright, year, end_line_idx)
    """
    lines = content.split('\n')
    for i, line in enumerate(lines[:10]):  # Check first 10 lines
        if 'copyright' in line.lower():
            # Find the end of the copyright block
            end_idx = i
            while end_idx < len(lines) and end_idx < i + 15:  # Look at most 15 lines after copyright
                if not lines[end_idx].strip().startswith('#'):
                    break
                end_idx += 1
            # Try to extract year
            year_match = re.search(r'Copyright \(c\) (\d{4})', line)
            if year_match:
                return True, int(year_match.group(1)), end_idx
            return True, None, end_idx
    return False, None, None

def find_python_files(root_dir: str) -> List[str]:
    """Find all Python files in the directory and subdirectories."""
    python_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def process_file(filepath: str, dry_run: bool = False, overwrite: bool = False) -> Optional[str]:
    """Process a file to add or update copyright notice."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        has_copyright, existing_year, end_idx = check_copyright(content)
        current_year = datetime.now().year
        
        if has_copyright and not overwrite:
            if existing_year is None:
                return f"Skipping {filepath} - has copyright notice but year format unknown"
            elif existing_year == current_year:
                return f"Skipping {filepath} - copyright year is current ({current_year})"
            else:
                if dry_run:
                    return f"Would update copyright year in {filepath} from {existing_year} to {current_year}"
                # Update the year in the existing copyright notice
                updated_content = re.sub(
                    f'Copyright \\(c\\) {existing_year}',
                    f'Copyright (c) {current_year}',
                    content
                )
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                return f"Updated copyright year in {filepath} from {existing_year} to {current_year}"
        else:
            # Either no copyright or overwrite is True
            if dry_run:
                action = "overwrite" if has_copyright else "add"
                return f"Would {action} copyright notice in {filepath}"
            
            if has_copyright:
                # Remove old copyright notice and add new one
                lines = content.split('\n')
                remaining_content = '\n'.join(lines[end_idx:])
                new_content = COPYRIGHT_NOTICE.format(year=current_year) + remaining_content
            else:
                new_content = COPYRIGHT_NOTICE.format(year=current_year) + content
                
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            action = "Overwrote" if has_copyright else "Added"
            return f"{action} copyright notice in {filepath}"
        
    except Exception as e:
        return f"Error processing {filepath}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Add or update copyright notices in Python files')
    parser.add_argument('--dry-run', action='store_true', 
                      help='Show what would be done without making any changes')
    parser.add_argument('--overwrite', action='store_true',
                      help='Overwrite existing copyright notices with the current template')
    args = parser.parse_args()
    
    # Get the repository root directory (assuming script is in root)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    
    # Process all Python files
    python_files = find_python_files(os.path.join(repo_root, 'fireants'))
    for filepath in python_files:
        result = process_file(filepath, dry_run=args.dry_run, overwrite=args.overwrite)
        print(result)

if __name__ == '__main__':
    main() 