# FILE: create_dump.py
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

import os
from pathlib import Path

# List of all important source code files to include in the dump
FILES_TO_DUMP = [
    # Documentation & Metadata
    "README.md",
    "requirements.txt",

    # Core Training Scripts
    "train_supervised.py",
    "agent/train_agentic.py",
    
    # Core Environment & Utilities
    "agent/env.py",
    "rusty_terminal.py",
    "utils/checkpoint.py",
    "utils/rl.py",

    # Model (keep tokenizer out as it's generated)
    "rusty_r2/model/model_rwkv.py",
    
    # Scripts
    "scripts/rebuild_tokenizer.py",
    "scripts/download_datasets.py",
    "inference_runtime.py",

    # Tasks & Tests
    "tasks/sum_list/prompt.txt",
    "tasks/sum_list/template.py", 
    "tasks/sum_list/tests.py",
    "tests/test_r2_equivalence.py",

    # Documentation
    "docs/rusty_r2_upgrade.md",

    # Main Pipeline Script
    "train_rusty_r2_pipeline.py",
]

OUTPUT_FILE = "dump.txt"

def create_dump():
    """
    Concatenates the content of specified project files into a single
    text file for easy review.
    """
    print(f"Creating file dump in '{OUTPUT_FILE}'...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as dump_file:
        for file_path_str in FILES_TO_DUMP:
            file_path = Path(file_path_str)
            
            header = f"--- FILE: {file_path_str} ---\n"
            print(f"Dumping {file_path_str}...")
            
            dump_file.write(header)
            
            try:
                content = file_path.read_text(encoding="utf-8")
                dump_file.write(content)
            except FileNotFoundError:
                dump_file.write("!!! FILE NOT FOUND !!!\n")
                print(f"  Warning: Could not find file {file_path_str}")
            except Exception as e:
                dump_file.write(f"!!! ERROR READING FILE: {e} !!!\n")
                print(f"  Error reading file {file_path_str}: {e}")

            dump_file.write("\n\n")

    print(f"\nDump complete. All specified files have been written to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    create_dump()