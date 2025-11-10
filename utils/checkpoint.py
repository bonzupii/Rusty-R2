# utils/checkpoint.py
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