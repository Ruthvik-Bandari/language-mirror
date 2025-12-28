#!/usr/bin/env python3
"""
🏋️ Language Mirror Pro - Training Script
==========================================
Train the custom RL language tutor model using PPO.

Usage:
    python scripts/train.py                      # Quick training
    python scripts/train.py --num_updates 2000   # Full training
    python scripts/train.py --resume checkpoint.pt  # Resume training
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm

from ai_core.models.transformer import LanguageMirrorPro, ModelConfig
from ai_core.training.environment import LanguageLearningEnvironment, ResponseType


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for Language Mirror Pro.
    """
    
    def __init__(
        self,
        model: LanguageMirrorPro,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        # Training stats
        self.training_stats = []
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32, device=self.device)
    
    def collect_rollout(self, env, num_steps: int):
        """Collect rollout data from environment"""
        self.model.eval()
        
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        
        obs = env.reset()
        
        for _ in range(num_steps):
            # Prepare inputs
            # For simplicity, we'll use dummy tokenized input
            input_ids = torch.randint(4, 1000, (1, 32)).to(self.device)
            language_idx = obs["language_idx"].unsqueeze(0).to(self.device)
            proficiency = obs["proficiency"].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, language_idx, proficiency, task="rl")
                state = outputs["state"]
                
                # Get action from policy
                action, log_prob, entropy = self.model.policy_head.get_action(state)
                value = outputs["value"]
            
            # Store
            states.append(state.squeeze(0))
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())
            
            # Step environment
            obs, reward, done, info = env.step(action.item())
            rewards.append(reward)
            dones.append(float(done))
            
            if done:
                obs = env.reset()
        
        return {
            "states": torch.stack(states),
            "actions": torch.tensor(actions, dtype=torch.long, device=self.device),
            "log_probs": torch.tensor(log_probs, device=self.device),
            "values": torch.tensor(values, device=self.device),
            "rewards": rewards,
            "dones": dones
        }
    
    def train_step(self, rollout, num_epochs: int = 4, batch_size: int = 64):
        """Perform PPO training step"""
        self.model.train()
        
        states = rollout["states"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"]
        values = rollout["values"]
        
        # Compute advantages and returns
        advantages = self.compute_gae(rollout["rewards"], values.tolist(), rollout["dones"])
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_batches = 0
        
        indices = np.arange(len(states))
        
        for _ in range(num_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Get new policy outputs
                policy_logits = self.model.policy_head(batch_states)
                new_values = self.model.value_head(batch_states)
                
                # Calculate new log probs
                probs = F.softmax(policy_logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss -
                    self.entropy_coef * entropy
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_batches += 1
        
        stats = {
            "loss": total_loss / num_batches,
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "entropy": total_entropy / num_batches
        }
        
        self.training_stats.append(stats)
        return stats
    
    def save_checkpoint(self, path: str, update: int, best_reward: float):
        """Save training checkpoint"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update": update,
            "best_reward": best_reward,
            "training_stats": self.training_stats
        }, path)
        print(f"✅ Checkpoint saved: {path}")


def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Train Language Mirror Pro")
    parser.add_argument("--num_updates", type=int, default=500, help="Number of training updates")
    parser.add_argument("--steps_per_update", type=int, default=128, help="Steps per rollout")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N updates")
    parser.add_argument("--save_interval", type=int, default=100, help="Save every N updates")
    parser.add_argument("--eval_interval", type=int, default=50, help="Evaluate every N updates")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🏋️  Language Mirror Pro - Training")
    print("=" * 70)
    
    # Device
    device = get_device() if args.device == "auto" else args.device
    print(f"🖥️  Device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print("\n🧠 Creating model...")
    config = ModelConfig()
    model = LanguageMirrorPro(config)
    
    n_params = model.count_parameters()
    print(f"   Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    # Create trainer
    trainer = PPOTrainer(model, lr=args.lr, device=device)
    
    # Resume if specified
    start_update = 0
    best_reward = float("-inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_update = checkpoint["update"]
        best_reward = checkpoint["best_reward"]
        trainer.training_stats = checkpoint.get("training_stats", [])
        print(f"   Resumed from update {start_update}")
    
    # Create environment
    print("\n🌍 Creating training environment...")
    env = LanguageLearningEnvironment(curriculum=True)
    
    # Training loop
    print(f"\n🚀 Starting training for {args.num_updates} updates...")
    print("-" * 70)
    
    progress_bar = tqdm(range(start_update, args.num_updates), desc="Training")
    recent_rewards = []
    
    for update in progress_bar:
        # Collect rollout
        rollout = trainer.collect_rollout(env, args.steps_per_update)
        
        # Train
        stats = trainer.train_step(rollout, batch_size=args.batch_size)
        
        # Track rewards
        episode_reward = sum(rollout["rewards"])
        recent_rewards.append(episode_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        
        # Update progress bar
        avg_reward = np.mean(recent_rewards)
        progress_bar.set_postfix({
            "reward": f"{avg_reward:.2f}",
            "policy_loss": f"{stats['policy_loss']:.4f}",
            "value_loss": f"{stats['value_loss']:.4f}"
        })
        
        # Logging
        if (update + 1) % args.log_interval == 0:
            tqdm.write(f"Update {update + 1}: reward={avg_reward:.2f}, "
                      f"policy_loss={stats['policy_loss']:.4f}, "
                      f"entropy={stats['entropy']:.4f}")
        
        # Save checkpoint
        if (update + 1) % args.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_{update + 1}.pt"
            trainer.save_checkpoint(str(checkpoint_path), update + 1, best_reward)
        
        # Save best model
        if avg_reward > best_reward and len(recent_rewards) >= 50:
            best_reward = avg_reward
            best_path = output_dir / "best_model.pt"
            model.save(str(best_path))
            tqdm.write(f"🏆 New best model! Reward: {best_reward:.2f}")
    
    # Final save
    final_path = output_dir / "final_model.pt"
    model.save(str(final_path))
    
    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print(f"   Best reward: {best_reward:.2f}")
    print(f"   Models saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
