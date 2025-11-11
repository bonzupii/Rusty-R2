#!/usr/bin/env python3
# FILE: scripts/list_datasets.py
# Copyright (C) Micah L. Ostrow <bonzupii@protonmail.com>
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
#
# This file is part of Rusty-R2: A Scrapyard Language Model (Next Generation).
#
# Rusty-R2 is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Rusty-R2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

"""
Utility script to list datasets that should be present in the data directory
based on the download_datasets.py configuration.
"""

import argparse
import sys
from pathlib import Path
import importlib.util

# Import DATASETS_TO_DOWNLOAD from download_datasets.py
spec = importlib.util.spec_from_file_location("download_datasets", "./scripts/download_datasets.py")
download_datasets_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_datasets_module)
DATASETS_TO_DOWNLOAD = download_datasets_module.DATASETS_TO_DOWNLOAD

def main():
    parser = argparse.ArgumentParser(description="List datasets configured for download and their status.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("./data/hf/"),
        help="Directory to check for datasets.",
    )
    args = parser.parse_args()

    print("Rusty-R2 Dataset Status Report")
    print("=" * 60)
    print(f"Checking datasets in: {args.data_dir}")
    print()
    
    present_count = 0
    missing_count = 0
    
    print(f"{'Dataset Name':<30} {'Category':<15} {'Status':<10} {'Path':<30}")
    print("-" * 85)
    
    for dataset_info in DATASETS_TO_DOWNLOAD:
        local_name = dataset_info["local_name"]
        category = dataset_info["category"]
        dataset_path = args.data_dir / local_name
        status = "PRESENT" if dataset_path.exists() else "MISSING"
        
        if dataset_path.exists():
            present_count += 1
        else:
            missing_count += 1
            
        print(f"{local_name:<30} {category:<15} {status:<10} {str(dataset_path):<30}")
    
    print("-" * 85)
    print(f"Total datasets: {len(DATASETS_TO_DOWNLOAD)}")
    print(f"Present: {present_count}")
    print(f"Missing: {missing_count}")
    print()
    
    if missing_count > 0:
        print("To download missing datasets, run:")
        print("  python scripts/download_datasets.py")
        print()
    
    # Print license information
    print("Dataset License Information:")
    print("=" * 60)
    for dataset_info in DATASETS_TO_DOWNLOAD:
        local_name = dataset_info["local_name"]
        category = dataset_info["category"]
        license_ok = dataset_info["license_ok"]
        notes = dataset_info["notes"]
        
        print(f"- {local_name} ({category}): {license_ok}")
        if notes:
            print(f"  Notes: {notes}")
        print()

if __name__ == "__main__":
    main()