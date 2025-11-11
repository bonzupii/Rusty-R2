#!/usr/bin/env python3
# FILE: scripts/download_datasets.py
# Copyright (C) Micah L. Ostrow
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
#
# Simple dataset downloader for Rusty-R2.
# Just downloads the datasets. No pre-checks, no size limits, no license checks.

import argparse
import sys
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from datasets import load_dataset

# List of datasets to download
DATASETS = [
    # Your 7 successfully downloaded datasets
    {
        "hf_name": "codeparrot/apps",
        "subset": None,
        "local_name": "apps",
    },
    {
        "hf_name": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "local_name": "wikipedia_en_20231101",
    },
    {
        "hf_name": "openai/gsm8k",
        "subset": "main",
        "local_name": "gsm8k_openai",
    },
    {
        "hf_name": "OpenAssistant/oasst1",
        "subset": None,
        "local_name": "oasst1",
    },
    # NEW: General multi-language coding knowledge dataset (~15-18GB)
    # the-stack-smol-xl: 87 programming languages, 10k samples each
    {
        "hf_name": "bigcode/the-stack-smol-xl",
        "subset": None,
        "local_name": "the_stack_smol_xl",
    },
]


def download_datasets(output_dir: Path):
    """Download all datasets in the list."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in DATASETS:
        hf_name = spec["hf_name"]
        subset = spec.get("subset")
        local_name = spec["local_name"]
        save_path = output_dir / local_name

        print(f"\n{'=' * 60}")
        print(f"Dataset: {hf_name} [{subset}]")
        print(f"Saving to: {save_path}")
        print(f"{'=' * 60}")

        if save_path.exists():
            print(f"✓ Already exists, skipping")
            continue

        try:
            print(f"Downloading...")

            # Special handling for specific datasets
            if hf_name == "Muennighoff/mbpp":
                ds = load_dataset(
                    hf_name, subset, streaming=False, trust_remote_code=True
                )
            else:
                ds = load_dataset(hf_name, subset, streaming=False)

            print(f"Saving to disk...")
            ds.save_to_disk(str(save_path))
            print(f"✓ Successfully downloaded and saved")

        except Exception as e:
            print(f"✗ ERROR: {e}")
            print(f"Continuing to next dataset...")
            continue

    print(f"\n{'=' * 60}")
    print("Download process complete!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for Rusty-R2. Simple and straightforward."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./data/hf/"),
        help="Directory to save the datasets.",
    )
    args = parser.parse_args()

    download_datasets(args.output_dir)


if __name__ == "__main__":
    sys.exit(main())

