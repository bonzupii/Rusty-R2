import argparse
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import time
from pathlib import Path

from rusty_r2.model.model_rwkv import TinyRWKVLM
from tokenizers import Tokenizer
from utils.checkpoint import load_checkpoint

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("bitsandbytes not found. 8-bit and 4-bit quantization will not be available.")

def default_top_k_filter(logits: torch.Tensor, top_k: int):
    """In-place top-k filter on logits (batch x vocab)."""
    if top_k <= 0:
        return logits
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    minv = v[:, [-1]]
    logits[logits < minv] = -float('Inf')
    return logits


@torch.no_grad()
def generate(
    model,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    eos_token_id: Optional[int] = None,
    device: Optional[str] = None,
):
    """
    Autoregressive generation wrapper to be used by terminal and inference scripts.

    Args:
        model: model returning (logits, value) or (logits, hidden) where logits shape is (B, S, V)
        input_ids: (batch, seq_len) torch.LongTensor
        max_new_tokens: how many new tokens to sample
        temperature: sampling temperature
        top_k: apply top-k sampling filter if > 0
        eos_token_id: optional eos to stop early
        device: device to place tensors on; if None, use model device

    Returns:
        torch.LongTensor: full sequence including generated tokens (batch, seq_len + n)
    """
    if device is None:
        device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    model.eval()

    # Support models that accept (input_ids) and return (logits, value) or (logits, hidden)
    for _ in range(max_new_tokens):
        outputs = model(input_ids)
        # Try to extract logits as first returned element (common pattern in R2 models)
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            logits = outputs[0]
        else:
            logits = outputs

        last_logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        if top_k > 0:
            last_logits = default_top_k_filter(last_logits, top_k)

        probs = F.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_token_id is not None:
            # if all sequences in batch emitted eos, stop early
            if (next_token == eos_token_id).all():
                break

    return input_ids


def load_quantized_model(checkpoint_path: str, quantization: str, device: str):
    """Loads a model with optional quantization."""
    if quantization != "none" and not BITSANDBYTES_AVAILABLE:
        raise ImportError("bitsandbytes is required for 4-bit or 8-bit quantization.")

    ckpt_data = load_checkpoint(Path(checkpoint_path), device="cpu") # Load to CPU first for quantization
    model_config = ckpt_data['model_config']
    model_type = model_config.get("model_type", "transformer")

    # Instantiate model
    model = TinyRWKVLM(**model_config)
    
    # Apply quantization if requested
    if quantization == "4bit":
        # This is a simplified stub. Real 4-bit quantization involves more complex setup.
        print("4-bit quantization is a stub. Falling back to 8-bit if possible.")
        quantization = "8bit" # Fallback for now
    
    if quantization == "8bit":
        if not BITSANDBYTES_AVAILABLE:
            raise ImportError("bitsandbytes is required for 8-bit quantization.")
        
        # Recursively replace all nn.Linear layers with bnb.Linear8bitLt
        # This is a basic implementation. For production, consider using
        # a dedicated quantization library like Hugging Face's `accelerate` or `peft`
        # which handle more complex scenarios (e.g., layer exclusions, custom modules).
        def replace_linear_with_bnb_linear(module):
            for name, child in module.named_children():
                if isinstance(child, torch.nn.Linear):
                    setattr(module, name, bnb.Linear8bitLt(
                        child.in_features,
                        child.out_features,
                        has_bias=child.bias is not None,
                        threshold=6.0 # Default threshold from bitsandbytes
                    ))
                else:
                    replace_linear_with_bnb_linear(child)
        
        replace_linear_with_bnb_linear(model)
        model.load_state_dict(ckpt_data['model_state_dict'], strict=False) # strict=False because of bnb layer changes
        model = model.to(device)
    else: # "none" or fallback from 4bit
        model.load_state_dict(ckpt_data['model_state_dict'])
        model = model.to(device)

    model.eval()
    print(f"Loaded RWKV model with {quantization} quantization.")
    return model

def main():
    parser = argparse.ArgumentParser(description="Inference and Benchmark Runtime for Rusty-R2")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--quantization", type=str, default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--prompt", type=str, default="USER: Write a python function to find the nth fibonacci number.")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile()")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    tokenizer = Tokenizer.from_file("rusty_r2/tokenizer/tokenizer.json")
    
    # --- Load Model ---
    model = load_quantized_model(args.checkpoint, args.quantization, args.device)
    
    # --- Compile Model ---
    if args.compile:
        print("Compiling model with torch.compile()...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"torch.compile() failed: {e}. Continuing without compilation.")

    # --- Benchmark & Generate ---
    # Prepare input_ids for generation
    input_ids = torch.tensor([tokenizer.encode(args.prompt).ids], device=args.device)

    # Warmup run
    _ = generate(model, input_ids, max_new_tokens=5)
    
    # Timed run
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(args.device)
    
    start_time = time.time()
    
    generated_ids = generate(
        model, input_ids, args.max_new_tokens,
        args.temperature, args.top_k, tokenizer.token_to_id("<eos>"), args.device
    )
    
    if args.device == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    peak_memory_mb = 0
    if args.device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(args.device) / (1024 * 1024)

    generated_text = tokenizer.decode(generated_ids[0].tolist())
    num_generated = generated_ids.shape[1] - input_ids.shape[1]
    elapsed_time = end_time - start_time
    tokens_per_sec = num_generated / elapsed_time if elapsed_time > 0 else 0

    print("\n--- Generation Complete ---")
    print(generated_text)
    print("\n--- Benchmarks ---")
    print(f"Generated {num_generated} tokens in {elapsed_time:.2f} seconds.")
    print(f"Speed: {tokens_per_sec:.2f} tokens/sec")
    print(f"Peak Memory Usage: {peak_memory_mb:.2f} MB")