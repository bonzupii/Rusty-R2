
import argparse
import os
from pathlib import Path
from typing import Iterator, List

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders

from tqdm import tqdm

# --- R2 Tokenizer Configuration ---
VOCAB_SIZE = 24000
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
    return files

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
    
    if not source_files:
        print(f"No source files found in {data_dir}, but this may be OK if there are files inside subdirectories.")
        # Find text files deeper in the directory tree
        for root, dirs, files in os.walk(data_dir):
            for ext in ALLOWED_EXTENSIONS:
                source_files.extend([Path(root) / f for f in files if f.endswith(ext)])
    
    print(f"Found {len(source_files)} files to train tokenizer on.")
    
    if not source_files:
        raise RuntimeError(f"No source files found in {data_dir} after searching all subdirectories. Aborting.")

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
    text_iterator = get_text_iterator(source_files)
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
