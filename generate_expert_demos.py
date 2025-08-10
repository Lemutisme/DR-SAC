"""
Utility script to generate expert demonstrations for DR-AIL
This can either load pre-trained expert policies or use your trained SAC models
"""

import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from tqdm import tqdm
import argparse

# Import your SAC implementation
from sac import SAC_continuous, Actor


def collect_expert_demonstrations(env_name: str, 
                                 expert_model_path: str,
                                 num_episodes: int = 100,
                                 save_path: str = "./datasets",
                                 device: str = "cuda"):
    """
    Collect expert demonstrations using a pre-trained policy
    """
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Load expert policy
    expert_actor = Actor(
        state_dim, action_dim, 
        hid_shape=[256], hid_layers=2
    ).to(device)
    
    # Load pre-trained weights
    model_path = Path(expert_model_path)
    if model_path.exists():
        expert_actor.load_state_dict(
            torch.load(model_path / "actor.pth", map_location=device)
        )
        expert_actor.eval()
        print(f"Loaded expert policy from {model_path}")
    else:
        raise FileNotFoundError(f"Expert model not found at {model_path}")
    
    # Collect demonstrations
    all_states = []
    all_actions = []
    all_next_states = []
    all_rewards = []
    all_dones = []
    
    total_reward = 0
    
    for episode in tqdm(range(num_episodes), desc="Collecting demonstrations"):
        state, _ = env.reset()
        episode_states = []
        episode_actions = []
        episode_next_states = []
        episode_rewards = []
        episode_dones = []
        episode_reward = 0
        
        done = False
        while not done:
            # Get expert action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, _ = expert_actor(state_tensor, deterministic=True, with_logprob=False)
                action = action.cpu().numpy().flatten()
            
            # Step environment
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # Store transition
            episode_states.append(state)
            episode_actions.append(action)
            episode_next_states.append(next_state)
            episode_rewards.append(reward)
            episode_dones.append(float(done))
            
            state = next_state
            episode_reward += reward
        
        # Add episode data
        all_states.extend(episode_states)
        all_actions.extend(episode_actions)
        all_next_states.extend(episode_next_states)
        all_rewards.extend(episode_rewards)
        all_dones.extend(episode_dones)
        
        total_reward += episode_reward
    
    # Convert to numpy arrays
    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    next_states = np.array(all_next_states, dtype=np.float32)
    rewards = np.array(all_rewards, dtype=np.float32)
    dones = np.array(all_dones, dtype=np.float32)
    
    # Save demonstrations
    save_dir = Path(save_path) / f"{env_name.lower().replace('-', '_')}_expert"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(save_dir / "s.npy", states)
    np.save(save_dir / "a.npy", actions)
    np.save(save_dir / "s_next.npy", next_states)
    np.save(save_dir / "r.npy", rewards)
    np.save(save_dir / "dw.npy", dones)
    
    avg_reward = total_reward / num_episodes
    print(f"\nCollected {len(states)} transitions from {num_episodes} episodes")
    print(f"Average episode reward: {avg_reward:.2f}")
    print(f"Demonstrations saved to {save_dir}")
    
    env.close()
    return save_dir


def generate_mixed_quality_demonstrations(env_name: str,
                                         expert_model_path: str,
                                         num_expert_episodes: int = 50,
                                         num_noisy_episodes: int = 50,
                                         noise_std: float = 0.1,
                                         save_path: str = "./datasets",
                                         device: str = "cuda"):
    """
    Generate demonstrations with mixed quality (expert + noisy) for robustness testing
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Load expert policy
    expert_actor = Actor(
        state_dim, action_dim,
        hid_shape=[256], hid_layers=2
    ).to(device)
    
    model_path = Path(expert_model_path)
    expert_actor.load_state_dict(
        torch.load(model_path / "actor.pth", map_location=device)
    )
    expert_actor.eval()
    
    all_states = []
    all_actions = []
    all_next_states = []
    all_rewards = []
    all_dones = []
    
    # Collect expert demonstrations
    print("Collecting expert demonstrations...")
    for episode in tqdm(range(num_expert_episodes)):
        state, _ = env.reset()
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, _ = expert_actor(state_tensor, deterministic=True, with_logprob=False)
                action = action.cpu().numpy().flatten()
            
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            all_states.append(state)
            all_actions.append(action)
            all_next_states.append(next_state)
            all_rewards.append(reward)
            all_dones.append(float(done))
            
            state = next_state
    
    # Collect noisy demonstrations
    print(f"Collecting noisy demonstrations (noise_std={noise_std})...")
    for episode in tqdm(range(num_noisy_episodes)):
        state, _ = env.reset()
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, _ = expert_actor(state_tensor, deterministic=True, with_logprob=False)
                action = action.cpu().numpy().flatten()
                
                # Add noise to action
                noise = np.random.normal(0, noise_std, size=action.shape)
                action = action + noise
                action = np.clip(action, env.action_space.low, env.action_space.high)
            
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            all_states.append(state)
            all_actions.append(action)
            all_next_states.append(next_state)
            all_rewards.append(reward)
            all_dones.append(float(done))
            
            state = next_state
    
    # Convert and save
    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    next_states = np.array(all_next_states, dtype=np.float32)
    rewards = np.array(all_rewards, dtype=np.float32)
    dones = np.array(all_dones, dtype=np.float32)
    
    save_dir = Path(save_path) / f"{env_name.lower().replace('-', '_')}_mixed"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(save_dir / "s.npy", states)
    np.save(save_dir / "a.npy", actions)
    np.save(save_dir / "s_next.npy", next_states)
    np.save(save_dir / "r.npy", rewards)
    np.save(save_dir / "dw.npy", dones)
    
    print(f"\nCollected {len(states)} mixed-quality transitions")
    print(f"Demonstrations saved to {save_dir}")
    
    env.close()
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Generate expert demonstrations for DR-AIL")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Environment name")
    parser.add_argument("--expert_model", type=str, required=True, help="Path to expert model")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--mixed", action="store_true", help="Generate mixed quality demos")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Noise std for mixed demos")
    parser.add_argument("--save_path", type=str, default="./datasets", help="Save directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    if args.mixed:
        generate_mixed_quality_demonstrations(
            env_name=args.env,
            expert_model_path=args.expert_model,
            num_expert_episodes=args.num_episodes // 2,
            num_noisy_episodes=args.num_episodes // 2,
            noise_std=args.noise_std,
            save_path=args.save_path,
            device=args.device
        )
    else:
        collect_expert_demonstrations(
            env_name=args.env,
            expert_model_path=args.expert_model,
            num_episodes=args.num_episodes,
            save_path=args.save_path,
            device=args.device
        )


if __name__ == "__main__":
    main()