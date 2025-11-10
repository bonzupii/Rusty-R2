import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
import warnings
# Suppress specific PyTorch deprecation warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.GradScaler.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from tqdm import tqdm

# --- Checkpoint Utilities ---
from utils.checkpoint import save_checkpoint, load_checkpoint

# --- Model Imports ---
from rusty_r2.model.model_rwkv import TinyRWKVLM

# --- Tokenizer ---
from tokenizers import Tokenizer

# --- HuggingFace Datasets ---
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("huggingface_hub not found. Install with 'pip install datasets' to use HuggingFace datasets.")

# --- HuggingFace Datasets ---
try:
    from datasets import load_dataset, Dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("huggingface-hub not found. Install with 'pip install datasets' to use HuggingFace datasets.")

# --- 8-bit Optimizer ---
try:
    import bitsandbytes.optim as bnb_optim
    BITSANDBYTES_AVAILABLE = True
    print("bitsandbytes is available. Using 8-bit AdamW.")
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("bitsandbytes not found. Falling back to torch.optim.AdamW.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ConcatenatedTextDataset(IterableDataset):
    def __init__(self, file_paths: List[Path], tokenizer: Tokenizer, seq_len: int):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[torch.Tensor]:
        buffer = []
        for file_path in self.file_paths:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if content:
                    buffer.extend(self.tokenizer.encode(content).ids)
                    while len(buffer) >= self.seq_len:
                        sequence = buffer[:self.seq_len]
                        yield torch.tensor(sequence, dtype=torch.long)
                        buffer = buffer[self.seq_len:]
            except (IOError, OSError):
                continue

class HFDatasetWrapper(IterableDataset):
    """
    Efficient streaming wrapper for HuggingFace datasets to avoid OOM issues.
    Processes datasets one example at a time instead of loading everything into memory.
    """
    def __init__(self, hf_dirs: List[Path], tokenizer: Tokenizer, seq_len: int):
        self.hf_dirs = hf_dirs
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[torch.Tensor]:
        # Process each dataset directory one by one to avoid loading everything
        buffer = []
        
        for hf_dir in self.hf_dirs:
            try:
                print(f"Streaming from HuggingFace dataset: {hf_dir.name}")
                # Load the dataset in streaming mode to avoid memory issues
                dataset = load_dataset(str(hf_dir), streaming=True)
                
                # Process each split (train, test, etc.) separately
                for split_name in dataset.keys():
                    print(f"Processing split: {split_name}")
                    ds_stream = dataset[split_name]
                    
                    # Iterate through the stream without loading it all into memory
                    processed_examples = 0  # Initialize variable properly
                    max_examples_per_split = 10000  # Limit per dataset split to avoid spending forever on one
                    for example in ds_stream:
                        # Extract text content from the example using common keys
                        text_content = None
                        
                        # Try various standard keys where text might be stored
                        if 'text' in example:
                            text_content = example['text']
                        elif 'code' in example:
                            text_content = example['code']
                        elif 'content' in example:
                            text_content = example['content']
                        elif 'question' in example and 'answer' in example:
                            text_content = example['question'] + " " + example['answer']
                        elif 'question' in example:
                            text_content = example['question']
                        elif 'answer' in example:
                            text_content = example['answer']
                        elif 'problem' in example and 'solution' in example:
                            text_content = example['problem'] + " " + example['solution']
                        elif 'instruction' in example and 'output' in example:
                            text_content = example['instruction'] + " " + example['output']
                        elif 'prompt' in example and 'completion' in example:
                            text_content = example['prompt'] + " " + example['completion']
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
                        
                        # Add extracted content to buffer if valid
                        if text_content and isinstance(text_content, str) and text_content.strip():
                            # Tokenize and add to buffer
                            tokenized = self.tokenizer.encode(str(text_content)).ids
                            buffer.extend(tokenized)
                            
                            # Yield sequences when buffer is large enough
                            while len(buffer) >= self.seq_len:
                                sequence = buffer[:self.seq_len]
                                yield torch.tensor(sequence, dtype=torch.long)
                                buffer = buffer[self.seq_len:]
                        
                        processed_examples += 1
                        # Limit examples per dataset split to avoid spending too long on one
                        if processed_examples >= max_examples_per_split:
                            break
                            
            except Exception as e:
                print(f"Could not load dataset from {hf_dir.name}, skipping: {e}")
                continue
        
        # Process any remaining tokens in the buffer (pad with pad_token if needed)
        if len(buffer) > self.seq_len // 2:  # Only yield if we have a reasonable amount of content
            # Pad or truncate to exactly seq_len
            if len(buffer) >= self.seq_len:
                sequence = buffer[:self.seq_len]
            else:
                # Pad with pad token (assuming pad token is 0 or get from tokenizer)
                pad_token_id = self.tokenizer.token_to_id("<pad>")
                if pad_token_id is None:
                    pad_token_id = 0  # Default to 0 if no pad token
                sequence = buffer + [pad_token_id] * (self.seq_len - len(buffer))
            
            yield torch.tensor(sequence, dtype=torch.long)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)

def find_text_files(root_dir: Path) -> List[Path]:
    extensions = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".sh"}
    return [p for ext in extensions for p in root_dir.rglob(f"*{ext}")]


class HFDatasetWrapper(IterableDataset):
    """
    Wrapper to load and process HuggingFace datasets from subdirectories.
    Streams data efficiently to avoid OOM issues.
    """
    def __init__(self, hf_dirs: List[Path], tokenizer: Tokenizer, seq_len: int):
        self.hf_dirs = hf_dirs
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Stream data from HuggingFace datasets."""
        for hf_dir in self.hf_dirs:
            if HF_AVAILABLE:
                try:
                    print(f"Loading HuggingFace dataset from: {hf_dir.name}")
                    ds = load_dataset(str(hf_dir))
                    for split_name in ds.keys():
                        dataset_split = ds[split_name]
                        buffer = []
                        for example_idx in range(len(dataset_split)):
                            example = dataset_split[example_idx]
                            # Extract text content from various possible keys
                            text_content = None
                            if 'text' in example:
                                text_content = example['text']
                            elif 'code' in example:
                                text_content = example['code']
                            elif 'content' in example:
                                text_content = example['content']
                            elif 'question' in example:
                                text_content = example['question']
                                if 'answer' in example:
                                    text_content += " " + example['answer']
                            elif 'problem' in example and 'solution' in example:
                                text_content = example['problem'] + " " + example['solution']
                            elif 'instruction' in example and 'output' in example:
                                text_content = example['instruction'] + " " + example['output']
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
                                    text_parts = [msg.get('value', '') if isinstance(msg, dict) else str(msg) for msg in conversations]
                                    text_content = " ".join(filter(None, text_parts))
                                else:
                                    text_content = str(conversations)
                            elif len(example) > 0:
                                # Generic fallback: join all values
                                example_values = [str(v) for v in example.values() if v is not None]
                                text_content = " ".join(example_values)
                            
                            if text_content and str(text_content).strip():
                                # Tokenize the content
                                tokenized = self.tokenizer.encode(str(text_content)).ids
                                buffer.extend(tokenized)
                                
                                # Yield full sequences
                                while len(buffer) >= self.seq_len:
                                    sequence = buffer[:self.seq_len]
                                    yield torch.tensor(sequence, dtype=torch.long)
                                    buffer = buffer[self.seq_len:]
                except Exception as e:
                    print(f"Could not load dataset from {hf_dir.name}: {e}")
                    continue


def find_hf_datasets(root_dir: Path) -> List[Path]:
    """Find directories that look like HuggingFace datasets."""
    hf_dirs = []
    for item in root_dir.iterdir():
        if item.is_dir():
            # Check if it looks like a HuggingFace dataset directory
            has_metadata = any(item.glob("dataset_dict.json")) or any(item.glob("train/")) or any(item.glob("test/"))
            if has_metadata:
                hf_dirs.append(item)
    return hf_dirs

def load_hf_datasets(data_dir: Path) -> List[Dataset]:
    """Load HuggingFace datasets from subdirectories in data_dir."""
    datasets = []
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            try:
                ds = load_dataset(str(subdir))
                if 'train' in ds:
                    datasets.append(ds['train'])
                else:
                    # If no train split, use the first available split
                    first_split = next(iter(ds))
                    datasets.append(ds[first_split])
                print(f"Loaded dataset from {subdir.name}")
            except Exception as e:
                print(f"Could not load dataset from {subdir.name}: {e}")
    return datasets

def main():
    parser = argparse.ArgumentParser(description="Supervised Training for Rusty-R2")
    # Model & Data
    parser.add_argument("--tokenizer_path", type=str, default="rusty_r2/tokenizer/tokenizer.json", help="Path to the tokenizer.")
    parser.add_argument("--data_dir", type=str, default="./data/", help="Directory with raw text data or HuggingFace datasets.")
    # Training
    parser.add_argument("--batch_size", type=int, default=8, help="Initial batch size.")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Steps for gradient accumulation.")
    parser.add_argument("--max_steps", type=int, default=100000, help="Total training steps.")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    # Optimizer & Scheduler
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--warmup_pct", type=float, default=0.05, help="Percentage of steps for warmup.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")
    # System
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints_r2", help="Directory for R2 checkpoints.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--log_interval", type=int, default=20, help="Logging interval.")
    parser.add_argument("--ckpt_interval", type=int, default=1000, help="Checkpointing interval.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile().")
    parser.add_argument("--device", type=str, default="cuda", help="Device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    use_amp = False  # Changed from "device == 'cuda'" to False for numerical stability
    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    pad_token_id = tokenizer.token_to_id("<pad>")

    # Prepare model config for RWKV (filter out incompatible params)
    # Prepare model config for RWKV and filter out incompatible params
    # Get the actual vocabulary size from the tokenizer
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Actual tokenizer vocab size: {actual_vocab_size}")
    
    model_config = {
        "vocab_size": actual_vocab_size,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    
    model = TinyRWKVLM(**model_config).to(device)
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.compile:
        print("Compiling model with torch.compile()...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile() failed: {e}. Continuing without compilation.")

    print(f"Model: RWKV with {model.count_parameters():,} parameters.")
    
    if BITSANDBYTES_AVAILABLE:
        optimizer = bnb_optim.AdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    warmup_steps = int(args.max_steps * args.warmup_pct)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, args.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    global_step = 0
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        ckpt_data = load_checkpoint(
            Path(args.resume_from_checkpoint),
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device
        )
        global_step = ckpt_data.get('global_step', 0)

    # Find HuggingFace dataset directories and text files
    data_path = Path(args.data_dir)
    
    # Check if HF datasets exist in main data directory
    hf_dirs = find_hf_datasets(data_path)
    text_files = find_text_files(data_path)
    
    # Use HuggingFace datasets if available, otherwise use text files
    if hf_dirs and HF_AVAILABLE:
        print(f"Found {len(hf_dirs)} HuggingFace dataset directories")
        # Use HuggingFace datasets with streaming to avoid OOM
        dataset = HFDatasetWrapper(hf_dirs, tokenizer, args.seq_len)
    elif text_files:
        print(f"Found {len(text_files)} raw text files")
        # Use raw text files
        dataset = ConcatenatedTextDataset(text_files, tokenizer, args.seq_len)
    else:
        print(f"No datasets found in {args.data_dir}")
        print("Looking for text files in subdirectories...")
        # Try looking in subdirectories for text files
        for subdir in data_path.iterdir():
            if subdir.is_dir():
                text_files.extend(find_text_files(subdir))
        if text_files:
            print(f"Found {len(text_files)} text files in subdirectories")
            dataset = ConcatenatedTextDataset(text_files, tokenizer, args.seq_len)
        else:
            # Look for HF datasets in subdirs too 
            hf_dirs = []  # Reset to empty list
            for subdir in data_path.iterdir():
                if subdir.is_dir():
                    hf_dirs.extend(find_hf_datasets(subdir))
            if hf_dirs and HF_AVAILABLE:
                print(f"Found {len(hf_dirs)} HuggingFace datasets in subdirectories")
                dataset = HFDatasetWrapper(hf_dirs, tokenizer, args.seq_len)
            else:
                raise ValueError(f"No text files or HuggingFace datasets found in {args.data_dir}")
    
    current_batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=current_batch_size, num_workers=2)
    data_iterator = iter(dataloader)

    optimizer.zero_grad()
    
    # Initialize progress bar
    progress_bar = tqdm(range(args.max_steps), desc="Training Steps", unit="step")
    
    while global_step < args.max_steps:
        try:
            for micro_step in range(args.grad_accum_steps):
                try:
                    batch = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloader)
                    batch = next(data_iterator)

                batch = batch.to(device)
                input_ids = batch
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = pad_token_id

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, _ = model(input_ids)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=pad_token_id)
                    loss = loss / args.grad_accum_steps

                scaler.scale(loss).backward()

            # Use the last loss for tracking
            final_loss = loss * args.grad_accum_steps  # Undo grad_accum_steps division for display

            scaler.unscale_(optimizer)
            
            # Check if loss is finite before proceeding with optimization
            if torch.isfinite(loss):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
            else:
                print(f"Warning: Non-finite loss ({loss.item() * args.grad_accum_steps}) at step {global_step}, skipping optimizer step.")
                # Continue with scaler update but skip the optimizer step
            
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Update progress bar at each step
            current_loss_val = loss.item() * args.grad_accum_steps if torch.isfinite(loss) else float('nan')
            progress_bar.set_postfix({
                "loss": f"{current_loss_val:.4f}" if torch.isfinite(torch.tensor(current_loss_val)) else "nan", 
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            progress_bar.update(1)
            
            if global_step % args.log_interval == 0:
                loss_str = f"{current_loss_val:.4f}" if torch.isfinite(torch.tensor(current_loss_val)) else "nan"
                print(f"Step: {global_step:6d} | Loss: {loss_str} | LR: {scheduler.get_last_lr()[0]:.2e}")

            if global_step % args.ckpt_interval == 0:
                save_checkpoint(
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    global_step=global_step,
                    model_config=model_config,
                    checkpoint_dir=Path(args.checkpoints_dir),
                    filename=f"step_{global_step}.pt"
                )

        except torch.cuda.OutOfMemoryError:
            print(f"OOM at batch size {current_batch_size}. Halving and retrying.")
            torch.cuda.empty_cache()
            current_batch_size //= 2
            if current_batch_size < 1:
                print("Batch size is zero. Aborting.")
                break
            dataloader = DataLoader(dataset, batch_size=current_batch_size, num_workers=1)
            data_iterator = iter(dataloader)

        # Save a final checkpoint at the end of training if no checkpoints were saved during due to interval
        # This ensures there's always a checkpoint for the next pipeline step
        if global_step > 0 and (global_step % args.ckpt_interval) != 0:
            save_checkpoint(
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
                model_config=model_config,
                checkpoint_dir=Path(args.checkpoints_dir),
                filename=f"step_{global_step}_final.pt"  # Different name to distinguish from regular checkpoints
            )
            print(f"Saved final checkpoint at step {global_step}")

    # Close progress bar
    progress_bar.close()
    print("--- Training Complete ---")

if __name__ == "__main__":
    main()