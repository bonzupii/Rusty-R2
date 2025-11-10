# utils/checkpoint.py
# FILE: utils/checkpoint.py
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

import torch
from pathlib import Path
from typing import Optional

def save_checkpoint(model, checkpoint_dir: Path, filename: str, optimizer=None, scheduler=None, scaler=None, global_step: int = 0, model_config: dict = None, **extra):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config or getattr(model, "config", {}),
        "global_step": int(global_step),
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        try:
            ckpt["scaler_state_dict"] = scaler.state_dict()
        except Exception:
            pass
    ckpt.update(extra)
    torch.save(ckpt, checkpoint_dir / filename)
    print(f"Checkpoint saved to {checkpoint_dir / filename}")

def load_checkpoint(path: Path, device="cpu"):
    """
    Loads a checkpoint from path and returns the raw dictionary.
    The caller is responsible for loading the state into models, optimizers, etc.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{path}'")
    ckpt = torch.load(path, map_location=device)
    # return raw ckpt; caller may load parts as needed
    return ckpt