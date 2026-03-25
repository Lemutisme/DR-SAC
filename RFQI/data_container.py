import numpy as np
import torch
import sys
import os
from hydra.utils import get_original_cwd
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import Reward_adapter


class DATA(object):
    def __init__(self, state_dim, action_dim, max_action, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.max_action = max_action

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.device = device


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size, use_bootstrap=False):
        ind = np.random.randint(0, self.size, size=batch_size)

        if use_bootstrap:
            return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.bootstrap_mask[ind]).to(self.device),
        )

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.size])
        np.save(f"{save_folder}_action.npy", self.action[:self.size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)


    def load(self, save_folder, reward_adapt, reward_normalize, EnvIdex, size=-1, bootstrap_dim=None):
        save_folder = Path(get_original_cwd()) / save_folder / "dataset"
        # reward_buffer = np.load(f"{save_folder}_reward.npy")
        # # Adjust crt_size if we're using a custom size
        # size = min(int(size), self.max_size) if size > 0 else self.max_size
        # self.size = min(reward_buffer.shape[0], size)
        
        self.size = int(np.load(f"{save_folder}/size.npy")[0])
        print(f"{self.size} data loaded.")
        # resize data first!
        self.state.resize((self.size, self.state_dim))
        self.action.resize((self.size, self.action_dim))
        self.next_state.resize((self.size, self.state_dim))
        self.reward.resize((self.size, 1))
        self.not_done.resize((self.size, 1))
        
        # then load!
        self.state[:self.size] = np.load(f"{save_folder}/s.npy")[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}/a.npy")[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}/s_next.npy")[:self.size]
        self.reward[:self.size] = np.load(f"{save_folder}/r.npy")[:self.size]
        if reward_adapt:
            print(f"Before reward adaptation: Max: {self.reward.max():.4f}, Min: {self.reward.min():.4f}, Mean: {self.reward.mean():.4f}.")
            for i in range(self.size):
                self.reward[i] = Reward_adapter(self.reward[i], EnvIdex)
            print(f"After reward adaptation: Max: {self.reward[:self.size].max():.4f}, Min: {self.reward[:self.size].min():.4f}, Mean: {self.reward[:self.size].mean():.4f}.")
        elif reward_normalize:
            print("Normalize reward to [0,1]")
            r_max, r_min = np.max(self.reward), np.min(self.reward)
            for i in range(self.size):
                self.reward[i] = (self.reward[i] - r_min) / (r_max - r_min)
        self.not_done[:self.size] = 1. - np.load(f"{save_folder}/dw.npy")[:self.size]
        
        if bootstrap_dim is not None:
            self.bootstrap_dim = bootstrap_dim
            bootstrap_mask = np.random.binomial(n=1, size=(1, self.size, bootstrap_dim,), p=0.8)
            bootstrap_mask = np.squeeze(bootstrap_mask, axis=0)
            self.bootstrap_mask = bootstrap_mask[:self.size]
