"""
File utility functions
"""
import os
from typing import List, Optional

def list_files_recursive(dir_path: str, ext: str) -> List[str]:
    """Return sorted list of full paths under dir_path with extension ext"""
    out = []
    if not dir_path or not os.path.exists(dir_path):
        return out
    
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(ext.lower()):
                out.append(os.path.join(root, f))
    
    out.sort()
    return out

def normalize_basename(path_or_name, suffixes_to_strip: Optional[List[str]] = None) -> str:
    """Return the base filename in lowercase without specified suffixes"""
    if path_or_name is None:
        return None
    
    # If bytes, decode
    if isinstance(path_or_name, (bytes, bytearray)):
        try:
            path_or_name = path_or_name.decode('utf-8', errors='ignore')
        except:
            path_or_name = str(path_or_name)
    
    # Get base name without extension, lowercase
    base = os.path.splitext(os.path.basename(str(path_or_name)))[0]
    base_lower = base.lower()
    
    # Strip specified suffixes
    if suffixes_to_strip:
        sorted_suffixes = sorted(suffixes_to_strip, key=len, reverse=True)
        for suffix in sorted_suffixes:
            suffix_lower = suffix.lower()
            if base_lower.endswith(suffix_lower):
                base_lower = base_lower[:-len(suffix_lower)]
                break
    
    return base_lower