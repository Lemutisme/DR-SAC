import numpy as np
import torch
from pathlib import Path
from utils import Reward_adapter

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size, device, EnvIndex):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # We normalize state for better Halfcheetah VAE training
        # Action is not normalized since it ranges from [-1,1].
        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.device)
        self.s_norm = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.device)
        self.a = torch.zeros((max_size, action_dim), dtype=torch.float, device=self.device)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.device)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.device)
        self.s_next_norm = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.device)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.device)
        
        # We keep the option to select dimensions to normalize, but in Halfcheetah, we normalize all dimensions.
        self.normalize_dim = []
        if EnvIndex == 3:
             self.normalize_dim = range(state_dim)        

    # Add data tuple (s,a,r,s',dw) to buffer.
    def add(self, s, a, r, s_next, dw):
        if self.size == self.max_size:
            print("Max size reached. The first tuple will be replaced.")
        self.s[self.ptr] = torch.from_numpy(s).to(self.device)
        self.a[self.ptr] = torch.from_numpy(a).to(self.device) 
        if not isinstance(r, np.ndarray):
            r = np.array([r])
        self.r[self.ptr] = torch.from_numpy(r).to(self.device) 
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.device)
        self.dw[self.ptr] = torch.tensor(dw, dtype=torch.bool)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # Sample from buffer
    def sample(self, batch_size, norm_only=False):
        ind = torch.randint(0, self.size, device=self.device, size=(batch_size,))
        # For VAE training, we return normalized data only.
        if norm_only:
             return self.s_norm[ind], self.a[ind], self.s_next_norm[ind]
        else: 
            return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind], self.s_norm[ind], self.s_next_norm[ind]
    
    def save(self, path=None): 
        if not path:
            path = Path(f"./dataset")
        path.mkdir(parents=True, exist_ok=True)
            
        np.save(f"{path}/size.npy", np.array([self.size]))
        np.save(f"{path}/s.npy", self.s[:self.size].cpu().numpy())
        np.save(f"{path}/a.npy", self.a[:self.size].cpu().numpy())
        np.save(f"{path}/r.npy", self.r[:self.size].cpu().numpy())
        np.save(f"{path}/s_next.npy", self.s_next[:self.size].cpu().numpy())
        np.save(f"{path}/dw.npy", self.dw[:self.size].cpu().numpy())
            
    def load(self, path, reward_adapt, reward_normalize, EnvIdex):
        path =  Path(path) / "dataset"
        
        self.size = int(np.load(f"{path}/size.npy")[0])
        print(f"{self.size} data loaded.")
        self.s[:self.size,] = torch.from_numpy(np.load(f"{path}/s.npy")).to(self.device)
        self.s_norm[:self.size,] = self.s[:self.size,].clone() 
        self.s_mean, self.s_std = self.s[:self.size,].mean(dim=0), self.s[:self.size,].std(dim=0)+1e-6
        self.s_norm[:self.size, self.normalize_dim] = (self.s[:self.size, self.normalize_dim] - self.s_mean[self.normalize_dim]) / self.s_std[self.normalize_dim]
        
        self.a[:self.size,] = torch.from_numpy(np.load(f"{path}/a.npy")).to(self.device)
        
        r_cpu = torch.from_numpy(np.load(f"{path}/r.npy")).reshape((-1,1))
        # We apply reward engineering or normalization when loading dataset.
        if reward_adapt:
            print(f"Before adaptation: Max: {r_cpu.max():.4f}, Min: {r_cpu.min():.4f}, Mean: {r_cpu.mean():.4f}.")
            r_cpu.apply_(lambda r: Reward_adapter(r, EnvIdex))
            print(f"After adaptation: Max: {r_cpu.max():.4f}, Min: {r_cpu.min():.4f}, Mean: {r_cpu.mean():.4f}.")
        elif reward_normalize:
            print("Normalize reward to [0,1]")
            r_max, r_min = np.max(r_cpu), np.min(r_cpu)
            r_cpu = (r_cpu - r_min) / (r_max - r_min)
        self.r[:self.size,] = r_cpu.to(self.device)
        
        self.s_next[:self.size,] = torch.from_numpy(np.load(f"{path}/s_next.npy")).to(self.device)
        self.s_next_norm[:self.size,] =  self.s_next[:self.size,:].clone() 
        self.s_next_norm[:self.size, self.normalize_dim] = (self.s_next[:self.size, self.normalize_dim] - self.s_mean[self.normalize_dim]) / self.s_std[self.normalize_dim]
        
        self.dw[:self.size,] = torch.from_numpy(np.load(f"{path}/dw.npy")).to(self.device)

    # State De-normalize function
    # Used to rescale state samples generated from VAE 
    def state_reverse(self, s):
        output = s.clone()
        output[:, :, self.normalize_dim] = s[:, :,  self.normalize_dim] * self.s_std[self.normalize_dim] + self.s_mean[self.normalize_dim]
        return output
            