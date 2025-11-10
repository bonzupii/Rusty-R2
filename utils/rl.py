# FILE: utils/rl.py
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
import numpy as np
from tokenizers import Tokenizer
from typing import Tuple, Optional
import torch.nn.functional as F

@torch.no_grad()
def generate_action(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    obs_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 50
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates an action sequence, calculates its log probability, and gets the value estimate.

    Returns:
        Tuple containing:
        - full_action_ids (Tensor): The full sequence of token IDs for the action.
        - action_log_probs (Tensor): The sum of log probabilities for the generated tokens.
        - value (Tensor): The value estimate for the initial state (observation).
        - full_sequence_ids (Tensor): The observation + action sequence.
    """
    model.eval()
    eos_token_id = tokenizer.token_to_id("<eos>")
    
    # Get initial value estimate from the observation
    initial_logits, initial_value = model(obs_ids)
    # Use the value of the last token of the observation as the state value
    value = initial_value[:, -1, :]

    input_ids = obs_ids
    generated_ids = []
    log_probs = []

    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        last_logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        
        if top_k > 0:
            v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
            last_logits[last_logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(last_logits, dim=-1)
        log_prob_dist = F.log_softmax(last_logits, dim=-1)
        
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Collect the log probability of the chosen token
        log_probs.append(log_prob_dist.gather(1, next_token))

        input_ids = torch.cat([input_ids, next_token], dim=1)
        generated_ids.append(next_token)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    full_action_ids = torch.cat(generated_ids, dim=1)
    # Sum the log probabilities for the entire generated sequence
    action_log_probs = torch.cat(log_probs, dim=1).sum(dim=1)
    
    return full_action_ids, action_log_probs, value, input_ids


class RolloutBuffer:
    """
    Stores trajectories for PPO and computes advantages using GAE.
    """
    is_stub = False # This is now a real implementation

    def __init__(self, num_steps: int, num_envs: int, obs_shape, action_shape, device: str, gamma: float, gae_lambda: float):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Storage for trajectories. We store token IDs, not raw text.
        self.obs = torch.zeros((self.num_steps, self.num_envs, *obs_shape), dtype=torch.long, device=device)
        self.actions = torch.zeros((self.num_steps, self.num_envs, *action_shape), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((self.num_steps, self.num_envs), device=device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs), device=device)
        self.dones = torch.zeros((self.num_steps, self.num_envs), device=device)
        self.values = torch.zeros((self.num_steps, self.num_envs), device=device)
        
        self.step = 0

    def add(self, obs, action, log_prob, value, reward, done):
        """Add a new experience tuple to the buffer."""
        self.obs[self.step] = obs
        
        # Handle variable-length actions with proper padding/truncation
        # Determine the maximum action length available in storage
        max_store_len = self.actions.shape[2]
        
        # Handle action shape depending on whether it's 1D (single sequence) or 2D (batched)
        if len(action.shape) == 1:
            # If action is 1D (single sequence), add a dimension
            action_seq = action.unsqueeze(0)  # Shape becomes (1, action_len)
            action_len = action.shape[0]
        else:
            # If action is already 2D (batched), use as-is
            action_seq = action
            action_len = action.shape[1]
        
        # Determine how much of the action to store (up to max allowed length)
        store_len = min(action_len, max_store_len)
        
        # Clear the action slot first (set to 0, which should be padding token)
        self.actions[self.step, :, :].zero_()
        
        # Store the action (truncated if needed)
        if store_len > 0:
            self.actions[self.step, :, :store_len] = action_seq[:, :store_len]
        
        # Store other data
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value.flatten()
        self.rewards[self.step] = torch.tensor([reward], device=self.device)
        self.dones[self.step] = torch.tensor([done], device=self.device)
        
        self.step = (self.step + 1) % self.num_steps

    def compute_returns_and_advantages(self, last_value: torch.Tensor, last_done: torch.Tensor):
        """
        Computes returns and Generalized Advantage Estimation (GAE) for the collected trajectory.
        """
        self.advantages = torch.zeros_like(self.rewards).to(self.device)
        last_gae_lam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        self.returns = self.advantages + self.values

    def get(self, batch_size: int):
        """
        Yields minibatches of experience from the buffer.
        """
        num_samples = self.num_steps * self.num_envs
        indices = np.random.permutation(num_samples)

        # Flatten the data
        flat_obs = self.obs.reshape(num_samples, *self.obs.shape[2:])
        flat_actions = self.actions.reshape(num_samples, *self.actions.shape[2:])
        flat_log_probs = self.log_probs.flatten()
        flat_advantages = self.advantages.flatten()
        flat_returns = self.returns.flatten()
        flat_values = self.values.flatten()

        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            mb_indices = indices[start:end]
            yield (
                flat_obs[mb_indices],
                flat_actions[mb_indices],
                flat_log_probs[mb_indices],
                flat_advantages[mb_indices],
                flat_returns[mb_indices],
                flat_values[mb_indices],
            )