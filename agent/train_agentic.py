# FILE: agent/train_agentic.py
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
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import warnings
# Suppress specific PyTorch deprecation warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.GradScaler.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
from tokenizers import Tokenizer
import torch.nn.functional as F
from tqdm import tqdm

from rusty_r2.model.model_rwkv import TinyRWKVLM
from agent.env import CodingEnv
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.rl import RolloutBuffer, generate_action

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

def main():
    parser = argparse.ArgumentParser(description="PPO-Lite Agentic Training for Rusty-R2")
    # Model & Data
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="rusty_r2/tokenizer/tokenizer.json")
    # PPO Hyperparameters
    parser.add_argument("--total_timesteps", type=int, default=50000)
    parser.add_argument("--num_steps", type=int, default=256, help="Steps per rollout.")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="Epochs per PPO update.")
    parser.add_argument("--num_minibatches", type=int, default=4, help="Minibatches per PPO epoch.")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    # System
    parser.add_argument("--log_dir", type=str, default="runs_ppo_r2")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints_r2_agentic")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    # Action generation
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens per action.")
    parser.add_argument("--max_obs_len", type=int, default=384, help="Max observation length.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    num_envs = 1 # This implementation is for a single environment
    batch_size = int(args.num_steps * num_envs)
    minibatch_size = batch_size // args.num_minibatches

    # --- Setup ---
    writer = SummaryWriter(args.log_dir)
    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    
    ckpt = load_checkpoint(Path(args.init_checkpoint), device=device)
    model_config = ckpt['model_config']
    
    # Filter config to only include params accepted by the model
    accepted_args = TinyRWKVLM.__init__.__code__.co_varnames
    filtered_config = {k: v for k, v in model_config.items() if k in accepted_args}
    
    model = TinyRWKVLM(**filtered_config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    if BITSANDBYTES_AVAILABLE:
        optimizer = bnb_optim.AdamW8bit(model.parameters(), lr=args.lr, eps=1e-5)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5)

    env = CodingEnv()
    buffer = RolloutBuffer(args.num_steps, num_envs, (args.max_obs_len,), (args.max_new_tokens,), device, args.gamma, args.gae_lambda)

    # --- Training Loop ---
    global_step = 0
    start_time = time.time()
    
    obs_text = env.reset()
    
    # Tokenize and truncate observation
    obs_ids_full = tokenizer.encode(obs_text).ids
    start_idx = max(0, len(obs_ids_full) - args.max_obs_len)
    next_obs = torch.tensor([obs_ids_full[start_idx:]], dtype=torch.long, device=device)
    next_done = torch.zeros(num_envs, device=device)

    # Initialize progress bar for PPO updates
    num_updates = int(args.total_timesteps // batch_size) + 1
    progress_bar = tqdm(range(1, num_updates + 1), desc="PPO Updates", unit="update")
    
    for update in range(1, num_updates + 1):
        for step in range(args.num_steps):
            global_step += 1 * num_envs
            
            # --- Rollout Phase ---
            action_ids, log_prob, value, full_sequence = generate_action(
                model, tokenizer, next_obs, args.max_new_tokens
            )
            
            action_text = tokenizer.decode(action_ids[0].tolist())
            obs_text, reward, done, info = env.step(action_text)

            buffer.add(next_obs.squeeze(0), action_ids.squeeze(0), log_prob, value, reward, done)
            
            if done:
                print(f"Global Step: {global_step}, Task: {info.get('task', 'N/A')}, Passed: {info.get('passed', False)}, Reward: {reward:.2f}")
                writer.add_scalar("charts/episodic_return", reward, global_step)
                obs_text = env.reset()

            obs_ids_full = tokenizer.encode(obs_text).ids
            start_idx = max(0, len(obs_ids_full) - args.max_obs_len)
            next_obs = torch.tensor([obs_ids_full[start_idx:]], dtype=torch.long, device=device)

        # --- Advantage Calculation ---
        with torch.no_grad():
            _, next_value = model(next_obs)
            next_value = next_value[:, -1, :]
        buffer.compute_returns_and_advantages(next_value, next_done)

        # --- Update Phase ---
        model.train()
        for epoch in range(args.ppo_epochs):
            for mb in buffer.get(minibatch_size):
                mb_obs, mb_actions, mb_old_log_probs, mb_advantages, mb_returns, mb_values = mb
                
                # CRITICAL FIX: Single forward pass for PPO step
                full_sequence = torch.cat([mb_obs, mb_actions], dim=1)
                full_logits, _ = model(full_sequence)
                obs_len = mb_obs.size(1)
                action_logits = full_logits[:, obs_len:, :]  # (B, A, V)

                # Calculate log-probs and entropy using this single pass
                log_probs_all = F.log_softmax(action_logits, dim=-1)
                probs_all = F.softmax(action_logits, dim=-1)

                action_mask = (mb_actions != 0).float()  # relies on <pad>=0

                gathered_log_probs = log_probs_all.gather(2, mb_actions.unsqueeze(-1)).squeeze(-1)
                new_log_probs = (gathered_log_probs * action_mask).sum(dim=1)

                entropy_per_token = -(probs_all * log_probs_all).sum(dim=-1)
                entropy = (entropy_per_token * action_mask).sum() / (action_mask.sum() + 1e-8)

                # Policy Loss (PPO-Clip)
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                # Normalize advantages at the minibatch level
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss using current value predictions - gradients should flow through value head
                _, v_preds_full = model(mb_obs)
                v_preds = v_preds_full[:, -1, :].squeeze(-1)
                v_loss = F.mse_loss(v_preds, mb_returns)
                
                loss = pg_loss - args.ent_coef * entropy + v_loss * args.vf_coef
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # --- Logging ---
        sps = int(global_step / (time.time() - start_time))
        print(f"SPS: {sps}, Update: {update}")
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.item(), global_step)

        # Update progress bar
        progress_bar.set_postfix({"global_step": global_step, "policy_loss": f"{pg_loss.item():.4f}", "value_loss": f"{v_loss.item():.4f}"})
        progress_bar.update(1)

        # --- Checkpointing ---
        if update % 20 == 0:
            save_checkpoint(
                model,
                Path(args.checkpoints_dir),
                f"agent_step_{global_step}.pt",
                optimizer=optimizer,
                global_step=global_step,
                model_config=model_config,
            )

    env.close()
    progress_bar.close()
    writer.close()
    print("--- Agentic Training Complete ---")

if __name__ == "__main__":
    main()
