# FILE: scripts/download_datasets.py
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

import argparse
from pathlib import Path
from datasets import load_dataset

# Datasets to download
# Format: (huggingface_name, subset, local_name, trust_remote_code)
DATASETS_TO_DOWNLOAD = [
    ("glaiveai/glaive-code-assistant-v2", None, "glaive_v2", False),
    ("Nan-Do/code-search-net-javascript", None, "codesearchnet_js", False),
    ("codeparrot/apps", None, "apps", True),
    ("huybery/repair", None, "code_repair", False),
    ("MuskumPillerum/General-Knowledge", None, "general_knowledge", False),
]

def download_and_save_dataset(
    hf_name: str,
    subset: str | None,
    local_name: str,
    trust_remote_code: bool,
    output_dir: Path,
):
    """Downloads a dataset from Hugging Face and saves it to disk."""
    save_path = output_dir / local_name
    if save_path.exists():
        print(f"Dataset '{local_name}' already exists at {save_path}. Skipping.")
        return

    print(f"Downloading '{hf_name}' (subset: {subset or 'all'})...")
    try:
        dataset = load_dataset(
            hf_name,
            subset,
            streaming=False,
            trust_remote_code=trust_remote_code
        )
        dataset.save_to_disk(str(save_path))
        print(f"Successfully saved '{local_name}' to {save_path}")
    except Exception as e:
        print(f"Failed to download or save '{hf_name}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Download curated code datasets from Hugging Face.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./data/hf/"),
        help="Directory to save the datasets.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for hf_name, subset, local_name, trust_remote_code in DATASETS_TO_DOWNLOAD:
        download_and_save_dataset(hf_name, subset, local_name, trust_remote_code, args.output_dir)

    print("\n--- Dataset download process complete. ---")

if __name__ == "__main__":
    main()
