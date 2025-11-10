# FILE: scripts/rebuild_tokenizer.py
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
import os
from pathlib import Path
from typing import Iterator, List

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders

from tqdm import tqdm

# --- R2 Tokenizer Configuration ---
VOCAB_SIZE = 24000  # Match README's "24K vocabulary" description
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
# Limit to text files for tokenizer training, don't process binary Arrow files
ALLOWED_EXTENSIONS = {'.py', '.md', '.txt', '.json', '.toml', '.yaml', '.sh'}

def find_source_files(root_path: Path) -> List[Path]:
    """Recursively finds all files with allowed extensions in the given directory."""
    if not root_path.is_dir():
        print(f"Error: Data directory not found at '{root_path}'")
        return []

    files = []
    for ext in ALLOWED_EXTENSIONS:
        files.extend(root_path.rglob(f"*{ext}"))
    
    # Also search inside the HF datasets directories for source files
    hf_dir = root_path / "hf"
    if hf_dir.is_dir():
        for dataset_dir in hf_dir.iterdir():
            if dataset_dir.is_dir():
                for ext in ALLOWED_EXTENSIONS:
                    files.extend(dataset_dir.rglob(f"*{ext}"))
    
    return files

def extract_text_from_hf_dataset(hf_dir: Path) -> Iterator[str]:
    """Extract text content from HuggingFace datasets."""
    from datasets import load_from_disk
    
    for dataset_dir in hf_dir.iterdir():
        if dataset_dir.is_dir():
            try:
                print(f"Loading dataset from: {dataset_dir.name}")
                ds = load_from_disk(str(dataset_dir))
                
                # Process each split (train, test, etc.)
                for split_name in ds.keys():
                    print(f"Processing split: {split_name}")
                    dataset_split = ds[split_name]
                    
                    # Process each example in the dataset
                    for example_idx in range(min(len(dataset_split), 1000)):  # Limit to prevent taking too long
                        example = dataset_split[example_idx]
                        
                        # Extract text content from various possible keys
                        text_content = ""
                        
                        # Try various standard keys where text might be stored
                        if 'text' in example:
                            text_content = str(example['text'])
                        elif 'code' in example:
                            text_content = str(example['code'])
                        elif 'content' in example:
                            text_content = str(example['content'])
                        elif 'question' in example:
                            text_content = str(example['question'])
                            if 'solutions' in example:
                                # Add solutions if they exist
                                if isinstance(example['solutions'], list):
                                    for sol in example['solutions']:
                                        text_content += " " + str(sol)
                                else:
                                    text_content += " " + str(example['solutions'])
                        elif 'problem' in example and 'solution' in example:
                            text_content = str(example['problem']) + " " + str(example['solution'])
                        elif 'instruction' in example and 'output' in example:
                            text_content = str(example['instruction']) + " " + str(example['output'])
                        elif 'prompt' in example and 'completion' in example:
                            text_content = str(example['prompt']) + " " + str(example['completion'])
                        elif 'translation' in example:
                            # Handle translation datasets
                            translation = example['translation']
                            if isinstance(translation, dict):
                                text_parts = [str(v) for v in translation.values()]
                                text_content = " ".join(text_parts)
                            else:
                                text_content = str(translation)
                        elif 'conversations' in example:
                            # Handle conversation datasets
                            conversations = example['conversations']
                            if isinstance(conversations, list):
                                text_parts = []
                                for msg in conversations:
                                    if isinstance(msg, dict) and 'content' in msg:
                                        text_parts.append(str(msg['content']))
                                    elif isinstance(msg, str):
                                        text_parts.append(msg)
                                text_content = " ".join(text_parts)
                            else:
                                text_content = str(conversations)
                        elif 'messages' in example:
                            # Handle message-based datasets
                            messages = example['messages']
                            if isinstance(messages, list):
                                text_parts = []
                                for msg in messages:
                                    if isinstance(msg, dict) and 'content' in msg:
                                        text_parts.append(str(msg['content']))
                                text_content = " ".join(text_parts)
                            else:
                                text_content = str(messages)
                        elif len(example) > 0:
                            # Generic fallback: extract first string value
                            for key, value in example.items():
                                if isinstance(value, str) and len(value) > 10:  # Skip short/metadata values
                                    text_content = str(value)
                                    break
                        
                        # Yield if we found meaningful content
                        if text_content and text_content.strip():
                            yield text_content
            except Exception as e:
                print(f"Could not load dataset from {dataset_dir.name}: {e}")
                continue

def get_text_iterator(files: List[Path]) -> Iterator[str]:
    """
    Yields the content of text files, skipping unreadable or empty ones.
    """
    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if content:
                    yield content
        except (IOError, OSError):
            # Silently skip files that can't be read
            continue

def build_and_save_tokenizer(data_dir: Path, output_dir: Path):
    """Builds a Byte-Level BPE tokenizer with prefix-space handling and saves it."""
    print(f"Searching for source files in: {data_dir.resolve()}")
    source_files = find_source_files(data_dir)
    
    # Get text content from HF datasets
    hf_dir = data_dir / "hf"
    hf_texts = []
    if hf_dir.is_dir():
        print("Extracting text from HuggingFace datasets...")
        hf_texts = list(extract_text_from_hf_dataset(hf_dir))
        print(f"Extracted {len(hf_texts)} text samples from HF datasets")

    print(f"Found {len(source_files)} files to train tokenizer on.")
    print(f"Found {len(hf_texts)} text samples from HF datasets.")

    # If no source files found, at least use HF text samples
    if not source_files and not hf_texts:
        raise RuntimeError(f"No source files or HF dataset text found in {data_dir}. Aborting.")

    # 1. Initialize the Tokenizer with a BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # 2. Set up the normalizer (NFC is a good default)
    tokenizer.normalizer = normalizers.NFC()

    # 3. Set up the pre-tokenizer with ByteLevel and prefix-space
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    # 4. Configure and run the trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    print("Training tokenizer...")
    
    # Create iterator for text content
    def all_text_iterator():
        # First yield HF dataset text content
        for text in hf_texts:
            if text.strip():
                yield text
        # Then yield from source files
        for text in get_text_iterator(source_files):
            if text.strip():
                yield text

    text_iterator = all_text_iterator()
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    print("Training complete.")

    # 5. Save the tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")

    # 6. Verification
    print("\n--- Verification ---")
    print(f"Final vocab size: {tokenizer.get_vocab_size()}")
    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    unk_id = tokenizer.token_to_id("<unk>")

    print(f"Special token IDs: <pad>={pad_id}, <bos>={bos_id}, <eos>={eos_id}, <unk>={unk_id}")
    if (pad_id, bos_id, eos_id, unk_id) != (0, 1, 2, 3):
        raise RuntimeError("Special token IDs are not in the expected order (0, 1, 2, 3).")
    print("Special token IDs are correctly assigned.")

def main():
    parser = argparse.ArgumentParser(description="Build the Rusty-R2 BPE tokenizer.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/"),
        help="Path to the directory containing raw training data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./rusty_r2/tokenizer/"),
        help="Path to save the tokenizer files.",
    )
    args = parser.parse_args()

    try:
        build_and_save_tokenizer(args.data_dir, args.output_dir)
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    import sys
    main()