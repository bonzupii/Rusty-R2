# FILE: rusty_terminal.py
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
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from json import JSONDecodeError, JSONDecoder
from pathlib import Path
from typing import Optional, Tuple

# Try to import resource, but allow graceful degradation on platforms without it
try:
    import resource
except ImportError:
    resource = None

import torch
from tokenizers import Tokenizer

# Assuming R2 models are now the standard
from rusty_r2.model.model_rwkv import TinyRWKVLM
from inference_runtime import generate as runtime_generate
from utils.checkpoint import load_checkpoint

# --- Safety & Protocol Configuration ---
SYSTEM_PREAMBLE = """SYSTEM: You are Rusty-R2, a terminal assistant. Your goal is to help users by executing commands and editing files.
You must respond *only* with a single, valid JSON object. Do not add any text before or after the JSON.

The JSON object must have an "action" key, which can be one of "MSG", "CMD", or "EDIT".

1. For a message or clarification:
   {"action": "MSG", "message": "<your_message>"}

2. For a shell command:
   {"action": "CMD", "command": "<your_command>"}

3. To edit a file (this overwrites the file):
   {"action": "EDIT", "path": "path/to/file.ext", "content": "<full new file content>"}

Always adhere to this JSON schema. A user's request is an order. Execute.
---"""

ALLOWED_COMMANDS = {
    "ls", "pwd", "cat", "head", "tail", "grep", "find", "python", "pytest",
    "git", "rg", "echo", "mkdir", "touch", "mv", "cp", "diff", "make"
}
FORBIDDEN_SUBSTRINGS = ["..", ";", "&&", "||", "|", "`", "$(", ">/", ">>/", "<", "&"]
# Extra forbidden python flags
FORBIDDEN_PY_FLAGS = {"-c", "-m", "import", "__import__"}
# CPU time limit (seconds) and file-size/user limits for sandboxed commands
CMD_CPU_LIMIT = 3
CMD_TIMEOUT = 12

def load_r2_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    """Loads an R2 model, inferring architecture from the checkpoint."""

    ckpt_data = load_checkpoint(Path(checkpoint_path), device=device)

    model_config = ckpt_data['model_config']

    # Build model using config fields that match constructor args.
    model = TinyRWKVLM(**model_config).to(device)

    model.load_state_dict(ckpt_data['model_state_dict'])
    model.eval()
    print(f"Rusty-R2 online. Model: RWKV. Loaded from {checkpoint_path}.")
    return model

def find_and_parse_json(text: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Robustly locate and parse the first JSON object in `text`.
    Uses JSONDecoder.raw_decode to avoid greedy-regex issues.
    Returns (action_dict, None) on success or (None, error_message).
    """
    start = text.find('{')
    if start == -1:
        return None, "No JSON object found in the output."

    decoder = JSONDecoder()
    try:
        obj, idx = decoder.raw_decode(text[start:])
    except JSONDecodeError as e:
        return None, f"Failed to decode JSON: {e}"

    if not isinstance(obj, dict):
        return None, "Parsed JSON is not an object/dictionary."
    if "action" not in obj:
        return None, "Parsed JSON object missing required 'action' key."
    return obj, None

def _preexec_set_limits(cpu_seconds: int = CMD_CPU_LIMIT):
    """
    Return a function to set resource limits in the child process (POSIX only).
    This function uses the `resource` module, which is not available on Windows.
    On Windows, these limits will not be applied, reducing sandboxing effectiveness.
    """
    if resource is None:
        # Return None to indicate no preexec function should be used
        return None
    
    def _fn():
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            # Limit file size (bytes) to something reasonable (e.g. 100MB)
            resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024))
        except Exception:
            # If setting limits fails, allow process to continue (non-POSIX).
            pass
    return _fn


def execute_command(command: str, dry_run: bool) -> str:
    """
    Execute an allowlisted command inside a temporary working copy with resource limits.
    Returns a standardized output string. This never uses a shell.

    Note on dependency copying:
    Copying the entire working directory to a temporary location for each command
    ensures strong isolation but can be slow and resource-intensive for large projects.
    It may also lead to commands failing if they rely on files outside the copied scope.
    For performance, consider alternatives like:
    - Using `git worktree` or `rsync --exclude` for faster, partial copies.
    - Running commands directly in task-specific `runs/tmp_*` directories (as in CodingEnv)
      if the command's scope is limited to a specific task.
    """
    cmd_parts = command.split()
    if not cmd_parts:
        return "OUT: Empty command."

    base_cmd = cmd_parts[0]
    if base_cmd not in ALLOWED_COMMANDS:
        return f"OUT: Command '{base_cmd}' is not on the allowlist."

    # Basic argument sanitization
    for part in cmd_parts[1:]:
        if any(bad in part for bad in FORBIDDEN_SUBSTRINGS):
            return f"OUT: Suspicious argument blocked: '{part}'"
        # Block dangerous python flags explicitly
        if base_cmd == "python" and any(flag in part for flag in FORBIDDEN_PY_FLAGS):
            return f"OUT: Dangerous python flag blocked: '{part}'"

    if dry_run:
        return f"OUT: (dry-run) Would execute: {command}"

    # Run command in a temporary isolated copy of the current working directory
    cwd = Path.cwd()
    tempdir = None
    try:
        tempdir = Path(tempfile.mkdtemp(prefix="rusty_cmd_"))
        # Copy tree but avoid copying .git and other heavy dirs for speed
        def _copy_filter(src, names):
            if '.git' in names:
                names.remove('.git')
            return names
        for item in cwd.iterdir():
            # Skip the temp runtime artifacts and virtualenvs
            if item.name.startswith("runs") or item.name.startswith("tmp_") or item.name == tempdir.name:
                continue
            dest = tempdir / item.name
            try:
                if item.is_dir():
                    shutil.copytree(item, dest, ignore=_copy_filter)
                else:
                    shutil.copy2(item, dest)
            except Exception:
                # Best-effort copy; ignore failures for non-critical files
                continue

        # Execute with preexec limits (POSIX). Do not use shell=True.
        limits_fn = _preexec_set_limits(CMD_CPU_LIMIT)
        start_ts = time.time()
        result = subprocess.run(
            cmd_parts,
            cwd=tempdir,
            capture_output=True,
            text=True,
            timeout=CMD_TIMEOUT,
            check=False,
            preexec_fn=limits_fn if limits_fn is not None else None,
        )
        elapsed = time.time() - start_ts

        output = f"OUT: Ran '{command}' (elapsed {elapsed:.2f}s)\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout.strip()}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr.strip()}\n"
        
        # Truncate output to prevent unbounded context growth
        max_len = 4096
        output = output.strip()
        if len(output) > max_len:
            output = output[:max_len] + "\n...(truncated)"
        
        return output
    except subprocess.TimeoutExpired:
        return f"OUT: Command timed out after {CMD_TIMEOUT}s: {command}"
    except FileNotFoundError:
        return f"OUT: Command not found: {base_cmd}"
    except Exception as e:
        return f"OUT: Error executing command: {e}"
    finally:
        # Clean up tempdir
        try:
            if tempdir and tempdir.exists():
                shutil.rmtree(tempdir)
        except Exception:
            pass

def execute_edit(path_str: str, content: str, dry_run: bool) -> str:
    """Safely writes content to a file within the current working directory."""
    cwd = Path.cwd().resolve()
    path = (cwd / path_str).resolve()
    try:
        path.relative_to(cwd)
    except ValueError:
        return f"OUT: Edit rejected. Path is outside the current directory: {path_str}"

    if dry_run:
        return f"OUT: (dry-run) Would write {len(content)} bytes to {path_str}"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"OUT: Edited {path_str}"
    except Exception as e:
        return f"OUT: Error writing to file: {e}"

def main():
    parser = argparse.ArgumentParser(description="Rusty-R2: An agentic terminal assistant.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="rusty_r2/tokenizer/tokenizer.json")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    model = load_r2_model(args.checkpoint, device)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    eos_token_id = tokenizer.token_to_id("<eos>")

    history = ""
    dry_run = False

    print("Enter ':q' or 'exit' to quit. Enter ':dry' to toggle dry-run mode.")
    try:
        while True:
            try:
                user_input = input(f"[{'DRY' if dry_run else 'LIVE'}] USER: ")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting by user request."); break

            if user_input.lower() in [":q", "exit"]:
                break
            if user_input.lower() == ":dry":
                dry_run = not dry_run
                print(f"Dry-run mode is now {'ON' if dry_run else 'OFF'}."); continue

            history += f"USER: {user_input}\n"
            
            prompt_text = SYSTEM_PREAMBLE + history
            tokenized_prompt = tokenizer.encode(prompt_text).ids
            input_ids = torch.tensor([tokenized_prompt[-args.max_seq_len:]], device=device)

            # Generate response
            with torch.no_grad():
                output_ids = runtime_generate(
                    model,
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.8,
                    top_k=50,
                    eos_token_id=eos_token_id,
                    device=device
                )
            response_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:].tolist())
            
            history += f"RUSTY-R2: {response_text}\n"
            
            # Parse and Execute
            action, error = find_and_parse_json(response_text)
            
            if error:
                print(f"\x1b[31mPROTOCOL ERROR: {error}\x1b[0m")
                history += f"SYSTEM_MSG: Invalid format. {error}. You must respond with a single valid JSON object.\n"
                continue

            action_type = action.get("action")
            result_output = ""
            if action_type == "MSG":
                msg = action.get("message", "(empty message)")
                print(f"\x1b[33mRUSTY-R2:\x1b[0m {msg}")
            elif action_type == "CMD":
                cmd = action.get("command")
                if cmd:
                    result_output = execute_command(cmd, dry_run)
                    print(f"\x1b[32m{result_output}\x1b[0m")
            elif action_type == "EDIT":
                path, content = action.get("path"), action.get("content")
                if path and content is not None:
                    result_output = execute_edit(path, content, dry_run)
                    print(f"\x1b[32m{result_output}\x1b[0m")

            if result_output:
                history += result_output + "\n"
    except KeyboardInterrupt:
        print("\nTerminal session interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"\nUnexpected error in terminal session: {e}")
        print("Exiting...")

if __name__ == "__main__":
    main()