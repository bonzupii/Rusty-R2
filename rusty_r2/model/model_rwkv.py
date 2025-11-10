# FILE: rusty_r2/model/model_rwkv.py
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
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch.utils.checkpoint import checkpoint

class TimeMix(nn.Module):
    """
    The RWKV Time-mixing block, which replaces self-attention.
    This is a parallel implementation for training.
    """
    def __init__(self, d_hidden: int, d_state: int = 16):
        super().__init__()
        self.time_decay = nn.Parameter(torch.ones(d_hidden))
        self.time_first = nn.Parameter(torch.ones(d_hidden))
        
        self.key = nn.Linear(d_hidden, d_hidden, bias=False)
        self.value = nn.Linear(d_hidden, d_hidden, bias=False)
        self.receptance = nn.Linear(d_hidden, d_hidden, bias=False)
        self.output = nn.Linear(d_hidden, d_hidden, bias=False)
        
        # Initialize time_decay and time_first properly for stability
        nn.init.normal_(self.time_decay, mean=0.0, std=0.02)
        nn.init.normal_(self.time_first, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Time-shift the input to get x from the previous timestep
        x_shifted = F.pad(x, (0, 0, 1, 0))[:, :-1, :]
        
        k = self.key(x_shifted)
        v = self.value(x_shifted)
        r = torch.sigmoid(self.receptance(x))

        # Time-mixing parameters
        w = -torch.exp(self.time_decay) # Decay factor, should be negative
        u = self.time_first # Bonus for the first token

        # --- Parallel WKV Scan ---
        # This is a numerically stable parallel implementation of the WKV recurrence.
        # Proper broadcasting: w(C) * arange(T) -> ws(1, T, C)
        
        # Clamp k to prevent exp(k) from overflowing
        k_clamped = torch.clamp(k, min=-15.0, max=15.0)
        
        # Generate time-decay weights with stability
        ws = w.view(1, 1, C) * torch.arange(T, 0, -1, device=x.device).view(1, T, 1)
        
        # Clamp ws to prevent exp overflow
        ws = torch.clamp(ws, min=-20.0, max=0.0)
        
        e_k = torch.exp(k_clamped)
        # Clamp the sum to prevent exp overflow in e_w_k
        ws_plus_k = ws + k_clamped
        ws_plus_k = torch.clamp(ws_plus_k, max=20.0)  # Prevent exp from exploding
        e_w_k = torch.exp(ws_plus_k)
        
        wkv_num = torch.cumsum(e_w_k * v, dim=1)
        wkv_den = torch.cumsum(e_w_k, dim=1) + 1e-8  # Add small epsilon to prevent division by zero
        
        # Calculate the initial state contribution with stability
        u_expanded = u.view(1, 1, C) + ws + k_clamped  # Use k_clamped instead of k
        u_expanded = torch.clamp(u_expanded, max=20.0)  # Prevent exp from exploding
        exp_u_ws_k = torch.exp(u_expanded)
        
        numerator = exp_u_ws_k * v + wkv_num
        denominator = exp_u_ws_k + wkv_den
        
        # Additional numerical safety to prevent NaN/Inf
        wkv = numerator / denominator

        return self.output(r * wkv)

class ChannelMix(nn.Module):
    """
    The RWKV Channel-mixing block, a simple feed-forward network.
    """
    def __init__(self, d_hidden: int, d_ffn: int):
        super().__init__()
        self.key = nn.Linear(d_hidden, d_ffn, bias=False)
        self.receptance = nn.Linear(d_hidden, d_hidden, bias=False)
        self.value = nn.Linear(d_ffn, d_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.key(x)
        k = torch.square(torch.relu(k)) # SwiGLU-like activation
        r = torch.sigmoid(self.receptance(x))
        
        return r * self.value(k)

class RWKVBlock(nn.Module):
    def __init__(self, d_hidden: int, d_ffn: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_hidden)
        self.ln2 = nn.LayerNorm(d_hidden)
        self.attn = TimeMix(d_hidden)
        self.ffn = ChannelMix(d_hidden, d_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TinyRWKVLM(nn.Module):
    """
    A compact, RWKV-style model for Rusty-R2.
    """
    def __init__(
        self,
        vocab_size: int,
        d_embed: int = 256,
        d_hidden: int = 512,
        n_layers: int = 4,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.gradient_checkpointing = gradient_checkpointing

        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        self.embed_proj = nn.Linear(d_embed, d_hidden)
        self.ln_in = nn.LayerNorm(d_hidden)
        self.drop_in = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [RWKVBlock(d_hidden, d_hidden * 4) for _ in range(n_layers)]
        )

        self.ln_out = nn.LayerNorm(d_hidden)
        # Note: In the original implementation, weights were tied.
        # But with d_embed != d_hidden, direct weight tying is not compatible.
        # So we'll not perform weight tying for this model.
        self.lm_head = nn.Linear(d_hidden, vocab_size, bias=False)
        # Initialize the weight matrix with small random values for stability
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        self.value_head = nn.Linear(d_hidden, 1)
        
        # Apply proper initialization (critical for numerical stability)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Small normal init for stability (std=0.02 is common)
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def gradient_checkpointing_enable(self):
        """Enables gradient checkpointing for the RWKV blocks."""
        self.gradient_checkpointing = True

    def forward(self, input_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.token_embedding(input_ids)
        x = self.drop_in(self.ln_in(self.embed_proj(x)))

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        x = self.ln_out(x)
        
        logits = self.lm_head(x)
        value = self.value_head(x)

        return logits, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)