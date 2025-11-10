#!/usr/bin/env python3
"""
Rusty-R2 Full Training Pipeline Script
======================================

This interactive script guides you through the complete Rusty-R2 training pipeline:
1. Tokenizer building
2. Supervised training
3. Agentic fine-tuning
4. Model testing

Before running this script, ensure you have:
- Python dependencies installed (pip install -r requirements.txt)
- Optional: bitsandbytes for 8-bit optimizers (pip install bitsandbytes)
- Data in the appropriate directories
"""

import os
import sys
# FILE: train_rusty_r2_pipeline.py
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

import subprocess
import argparse
import time
from pathlib import Path


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)


def confirm_action(message):
    """Ask for user confirmation."""
    response = input(f"\n{message} (y/n): ").lower().strip()
    return response in ['y', 'yes']


def run_command(cmd, description, check_exit_code=True):
    """Run a shell command with a description."""
    print(f"\n{description}")
    print(f"Command: {cmd}")
    
    if not confirm_action("Execute this command?"):
        print("Skipping...")
        return False
    
    try:
        result = subprocess.run(cmd, shell=True, check=check_exit_code)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully!")
            return True
        else:
            print(f"✗ {description} failed with exit code {result.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error: {e}")
        if check_exit_code:
            return False
        else:
            return True


def check_prerequisites():
    """Check if required directories and files exist."""
    print_header("Checking Prerequisites")
    
    # Check requirements
    try:
        import torch
        import tokenizers
        import numpy as np
        print("✓ Required Python packages are available")
    except ImportError as e:
        print(f"✗ Missing Python package: {e}")
        return False
    
    # Check data directories
    data_dirs = ["data/hf", "data/raw"]
    for data_dir in data_dirs:
        if not Path(data_dir).exists():
            print(f"! Warning: {data_dir} directory does not exist")
            print("  You may need to create it and add training data")
    
    return True


def build_tokenizer():
    """Build the tokenizer."""
    print_header("Step 1: Building Tokenizer")
    print("This step will create a 24K vocabulary tokenizer from your data.")
    
    cmd = "python scripts/rebuild_tokenizer.py"
    desc = "Building tokenizer from data/raw and data/hf"
    
    success = run_command(cmd, desc)
    if success:
        if Path("rusty_r2/tokenizer/tokenizer.json").exists():
            print("✓ Tokenizer built successfully!")
            return True
        else:
            print("✗ Tokenizer file not found after building")
            return False
    return False


def prepare_data():
    """Check and prepare data for training."""
    print_header("Data Preparation Check")
    
    print("The training pipeline requires data in your data/ directory.")
    print("You can either:")
    print("  1. Use existing data in data/hf/")
    print("  2. Add your own text/data files to data/raw/")
    
    has_data = False
    
    # Check for data in hf directory
    hf_dir = Path("data/hf")
    if hf_dir.exists() and any(hf_dir.iterdir()):
        print(f"✓ Found data in {hf_dir}")
        has_data = True
    
    # Check for raw data
    raw_dir = Path("data/raw")
    if raw_dir.exists() and any(raw_dir.iterdir()):
        print(f"✓ Found raw data in {raw_dir}")
        has_data = True
    
    if not has_data:
        print(f"✗ No training data found in data/hf/ or data/raw/")
        print("Please add your training data before proceeding.")
        print("You can add .txt, .py, or other text files to data/raw/")
        return False
    
    print("✓ Data check completed")
    return True


def supervised_training():
    """Run supervised training."""
    print_header("Step 2: Supervised Training")
    print("This trains the base RWKV model on your text data.")
    
    print("\nTraining parameters:")
    print("  - Model: TinyRWKVLM")
    print("  - Architecture: 4-layer RWKV with parallelized TimeMix")
    print("  - Vocabulary: 24K (from tokenizer)")
    print("  - Embedding: 256, Hidden: 512")
    print("  - Gradient checkpointing enabled")
    
    # Get training parameters from user
    print("\nDefault training settings (you can customize):")
    max_steps = input("Max training steps (default 10000, press Enter for default): ").strip()
    max_steps = max_steps if max_steps else "10000"
    
    batch_size = input("Batch size (default 4, press Enter for default): ").strip()
    batch_size = batch_size if batch_size else "4"
    
    grad_acc_steps = input("Gradient accumulation steps (default 8, press Enter for default): ").strip()
    grad_acc_steps = grad_acc_steps if grad_acc_steps else "8"
    
    device = input("Device (cuda/cpu) (default cuda, press Enter for default): ").strip()
    device = device if device else "cuda"
    
    checkpoints_dir = input("Checkpoints directory (default ./checkpoints_r2_supervised, press Enter for default): ").strip()
    checkpoints_dir = checkpoints_dir if checkpoints_dir else "./checkpoints_r2_supervised"
    
    # Construct the command
    cmd = f"""python train_supervised.py \\
    --tokenizer_path rusty_r2/tokenizer/tokenizer.json \\
    --data_dir data/hf/ \\
    --max_steps {max_steps} \\
    --batch_size {batch_size} \\
    --grad_accum_steps {grad_acc_steps} \\
    --seq_len 512 \\
    --gradient_checkpointing \\
    --checkpoints_dir {checkpoints_dir} \\
    --device {device}"""

    desc = f"Running supervised training for {max_steps} steps"
    
    success = run_command(cmd, desc)
    if success:
        # Check if checkpoints were created
        checkpoints_path = Path(checkpoints_dir)
        if checkpoints_path.exists() and any(checkpoints_path.glob("*.pt")):
            checkpoint_files = list(checkpoints_path.glob("step_*.pt"))
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                print(f"✓ Supervised training completed! Latest checkpoint: {latest_checkpoint}")
                return str(latest_checkpoint)
        print("✗ No checkpoint files found after training")
        return None
    return None


def agentic_fine_tuning(supervised_checkpoint):
    """Run agentic fine-tuning."""
    print_header("Step 3: Agentic Fine-Tuning")
    print("This fine-tunes the model using PPO for agentic behavior.")
    
    if not supervised_checkpoint:
        print("✗ No supervised checkpoint provided. Cannot proceed with agentic fine-tuning.")
        return None
        
    print(f"Using checkpoint: {supervised_checkpoint}")
    
    # Get fine-tuning parameters
    print("\nDefault fine-tuning settings (you can customize):")
    total_timesteps = input("Total timesteps (default 10000, press Enter for default): ").strip()
    total_timesteps = total_timesteps if total_timesteps else "10000"
    
    num_steps = input("Number of steps per update (default 256, press Enter for default): ").strip()
    num_steps = num_steps if num_steps else "256"
    
    ppo_epochs = input("PPO epochs (default 4, press Enter for default): ").strip()
    ppo_epochs = ppo_epochs if ppo_epochs else "4"
    
    device = input("Device (cuda/cpu) (default cuda, press Enter for default): ").strip()
    device = device if device else "cuda"
    
    log_dir = input("Log directory for TensorBoard (default runs_ppo_r2, press Enter for default): ").strip()
    log_dir = log_dir if log_dir else "runs_ppo_r2"
    
    agentic_checkpoints_dir = input("Agentic checkpoints directory (default ./checkpoints_r2_agentic, press Enter for default): ").strip()
    agentic_checkpoints_dir = agentic_checkpoints_dir if agentic_checkpoints_dir else "./checkpoints_r2_agentic"
    
    cmd = f"""python agent/train_agentic.py \\
    --init_checkpoint {supervised_checkpoint} \\
    --tokenizer_path rusty_r2/tokenizer/tokenizer.json \\
    --total_timesteps {total_timesteps} \\
    --num_steps {num_steps} \\
    --ppo_epochs {ppo_epochs} \\
    --num_minibatches 4 \\
    --log_dir {log_dir} \\
    --checkpoints_dir {agentic_checkpoints_dir} \\
    --device {device}"""

    desc = f"Running agentic fine-tuning for {total_timesteps} timesteps"
    
    success = run_command(cmd, desc)
    if success:
        # Check if agentic checkpoints were created
        agentic_checkpoints_path = Path(agentic_checkpoints_dir)
        if agentic_checkpoints_path.exists() and any(agentic_checkpoints_path.glob("*agent*.pt")):
            checkpoint_files = list(agentic_checkpoints_path.glob("*agent*.pt"))
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                print(f"✓ Agentic fine-tuning completed! Latest checkpoint: {latest_checkpoint}")
                return str(latest_checkpoint)
        print("✗ No agentic checkpoint files found after fine-tuning")
        return None
    return None


def test_model(model_checkpoint):
    """Test the trained model."""
    print_header("Step 4: Testing the Model")
    
    if not model_checkpoint:
        print("✗ No model checkpoint provided for testing.")
        return False
    
    print(f"Testing model checkpoint: {model_checkpoint}")
    
    # Quick test command
    test_prompt = input("Enter a test prompt (or press Enter for default): ").strip()
    if not test_prompt:
        test_prompt = "USER: Write a python function to reverse a string."
    
    cmd = f"""python inference_runtime.py \\
    --checkpoint {model_checkpoint} \\
    --quantization none \\
    --prompt "{test_prompt}" \\
    --max_new_tokens 50 \\
    --temperature 0.8 \\
    --device cuda"""

    desc = f"Testing model with prompt: {test_prompt[:50]}..."
    
    success = run_command(cmd, desc)
    if success:
        print("✓ Model test completed!")
        return True
    return False


def interactive_terminal(model_checkpoint):
    """Start the interactive terminal with the trained model."""
    print_header("Step 5: Interactive Terminal")
    
    if not model_checkpoint:
        print("✗ No model checkpoint provided for the terminal.")
        return False
    
    print(f"Starting interactive terminal with checkpoint: {model_checkpoint}")
    print("Commands in the terminal:")
    print("  :q or exit - Quit terminal")
    print("  :dry - Toggle dry-run mode")
    
    if confirm_action("Start interactive terminal?"):
        cmd = f"python rusty_terminal.py --checkpoint {model_checkpoint}"
        try:
            print(f"\nStarting terminal...")
            print("Type 'exit' or ':q' to quit the terminal when done.")
            subprocess.run(cmd, shell=True)
            return True
        except KeyboardInterrupt:
            print("\nTerminal interrupted by user.")
            return True
    return False


def main():
    """Main pipeline execution."""
    print_header("Rusty-R2 Complete Training Pipeline")
    print("This script will guide you through training your RWKV-based assistant.")
    
    if not confirm_action("Start the complete training pipeline?"):
        print("Pipeline cancelled.")
        return
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\nPrerequisites check failed. Please resolve the issues and try again.")
        return
    
    # Step 2: Prepare data
    if not prepare_data():
        print("\nData preparation failed. Please add training data and try again.")
        return
    
    # Step 3: Build tokenizer
    tokenizer_success = build_tokenizer()
    if not tokenizer_success:
        print("\nTokenizer building failed. Please resolve the issues and try again.")
        return
    
    # Step 4: Supervised training
    print("\nNow proceeding with supervised training...")
    supervised_checkpoint = None
    if confirm_action("Run supervised training?"):
        supervised_checkpoint = supervised_training()
        if not supervised_checkpoint:
            print("\nSupervised training failed.")
            return
    else:
        print("Skipping supervised training.")
        # User can provide existing checkpoint
        checkpoint_path = input("Provide path to existing supervised checkpoint (or press Enter to skip): ").strip()
        if checkpoint_path and Path(checkpoint_path).exists():
            supervised_checkpoint = checkpoint_path
        else:
            print("No checkpoint provided, cannot proceed with agentic fine-tuning.")
            return
    
    # Step 5: Agentic fine-tuning
    agentic_checkpoint = None
    if confirm_action("Run agentic fine-tuning?"):
        agentic_checkpoint = agentic_fine_tuning(supervised_checkpoint)
        if not agentic_checkpoint:
            print("\nAgentic fine-tuning failed.")
            # Still try to use the supervised checkpoint for testing
            agentic_checkpoint = supervised_checkpoint
    else:
        print("Skipping agentic fine-tuning.")
        agentic_checkpoint = supervised_checkpoint
    
    # Step 6: Test the model
    if confirm_action("Test the trained model?"):
        test_model(agentic_checkpoint)
    
    # Step 7: Interactive terminal
    if confirm_action("Start interactive terminal with the trained model?"):
        interactive_terminal(agentic_checkpoint)
    
    print_header("Training Pipeline Complete!")
    print("Your Rusty-R2 model has been trained and tested.")
    print(f"Final checkpoint: {agentic_checkpoint}")
    print("\nNext steps:")
    print("- Review your trained model performance")
    print("- Fine-tune hyperparameters if needed")
    print("- Deploy for your specific use cases")


if __name__ == "__main__":
    main()