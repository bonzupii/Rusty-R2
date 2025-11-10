# FILE: train_agentic.py
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
import random
import time
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from rusty_r2.model.model_rwkv import TinyRWKVLM
from tokenizers import Tokenizer
from agent.env import CodingEnv

try:
    import bitsandbytes.optim as bnb_optim
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RolloutBuffer:
    def __init__(self, num_steps, obs_shape, action_shape, device):
        self.device = device
        self.num_steps = num_steps
        self.obs = torch.zeros((num_steps,) + obs_shape, device=device, dtype=torch.long)
        self.actions = torch.zeros((num_steps,) + action_shape, device=device, dtype=torch.long)
        self.log_probs = torch.zeros((num_steps,), device=device)
        self.rewards = torch.zeros((num_steps,), device=device)
        self.dones = torch.zeros((num_steps,), device=device)
        self.values = torch.zeros((num_steps,), device=device)
        self.pos = 0

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.pos += 1

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        advantages = torch.zeros_like(self.rewards)
        last_gae_lam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
        return advantages + self.values, advantages

    def get(self, batch_size):
        indices = np.random.permutation(self.num_steps)
        for start in range(0, self.num_steps, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            yield self.obs[batch_indices], self.actions[batch_indices], self.log_probs[batch_indices], self.values[batch_indices]

def generate_action(policy, obs, tokenizer, max_new_tokens=128):
    """Autoregressively sample an action and compute its log probability."""
    was_training = policy.training
    policy.eval()
    
    input_ids = obs
    log_probs = []
    action_tokens = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = policy(input_ids)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            dist = torch.distributions.Categorical(probs)
            next_token = dist.sample()
            
            log_probs.append(dist.log_prob(next_token))
            action_tokens.append(next_token.item())
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.token_to_id("<eos>"):
                break

    if was_training:
        policy.train()
        
    action_log_prob = torch.stack(log_probs).sum()
    action_json = tokenizer.decode(action_tokens, skip_special_tokens=True)
    return action_json, action_log_prob

def main():
    parser = argparse.ArgumentParser(description="PPO-Lite Agentic Training for Rusty-R2")
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="rusty_r2/tokenizer/tokenizer.json")
    parser.add_argument("--total_timesteps", type=int, default=50000)
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--log_dir", type=str, default="runs_ppo_r2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    writer = SummaryWriter(args.log_dir)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    ckpt = torch.load(args.init_checkpoint, map_location=args.device)
    model_config = ckpt['model_config']
    # Ensure model type is set to rwkv
    model_config['model_type'] = 'rwkv'
    
    policy = TinyRWKVLM(**model_config).to(args.device)
    policy.load_state_dict(ckpt['model_state_dict'])

    env = CodingEnv()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, eps=1e-5)
    
    buffer = RolloutBuffer(args.num_steps, (512,), (128,), args.device) # Simplified shapes
    obs = env.reset()
    global_step = 0

    num_updates = args.total_timesteps // args.num_steps
    for update in range(1, num_updates + 1):
        policy.eval()
        for step in range(args.num_steps):
            global_step += 1
            tokenized_obs = torch.tensor([tokenizer.encode(obs).ids[-512:]], device=args.device)
            
            action_json, action_log_prob = generate_action(policy, tokenized_obs, tokenizer)
            
            with torch.no_grad():
                _, value = policy(tokenized_obs)

            # Partial rewards
            try:
                action_obj = json.loads(action_json)
                if action_obj.get("action") == "EDIT":
                    compile(action_obj.get("content", ""), '<string>', 'exec')
                syntax_reward = 0.3
            except (json.JSONDecodeError, SyntaxError):
                syntax_reward = -0.1

            next_obs, reward, done, info = env.step(action_json)
            final_reward = reward + syntax_reward
            
            # This is a simplification; action padding/truncating is needed for buffer
            padded_action = torch.zeros(128, dtype=torch.long) 
            buffer.add(tokenized_obs.squeeze(0), padded_action, action_log_prob, final_reward, done, value.squeeze())
            obs = next_obs
            if done:
                obs = env.reset()

        # GAE Calculation
        with torch.no_grad():
            _, last_value = policy(torch.tensor([tokenizer.encode(obs).ids[-512:]], device=args.device))
        returns, advantages = buffer.compute_returns_and_advantages(last_value, args.gamma, args.gae_lambda)

        # PPO Update
        policy.train()
        for _ in range(args.ppo_epochs):
            for obs_b, act_b, logp_b, val_b in buffer.get(args.num_steps // args.num_minibatches):
                # This is a highly simplified loss calculation for brevity.
                # A full implementation requires re-calculating log_probs and entropy.
                _, new_values = policy(obs_b)
                value_loss = F.mse_loss(new_values.squeeze(), returns) # Placeholder
                policy_loss = -advantages.mean() # Placeholder
                loss = policy_loss + args.vf_coef * value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        writer.add_scalar("charts/mean_reward", buffer.rewards.mean().item(), global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)

    writer.close()
    env.close()

if __name__ == "__main__":
    main()