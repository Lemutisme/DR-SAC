import numpy as np
import torch
from pathlib import Path
from utils import Reward_adapter
#from hydra.utils import get_original_cwd

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size, device):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.device)
        self.a = torch.zeros((max_size, action_dim), dtype=torch.float, device=self.device)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.device)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.device)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.device)

    def add(self, s, a, r, s_next, dw):
        self.s[self.ptr] = torch.from_numpy(s).to(self.device)
        self.a[self.ptr] = torch.from_numpy(a).to(self.device) 
        if not isinstance(r, np.ndarray):
            r = np.array([r])
        self.r[self.ptr] = torch.from_numpy(r).to(self.device) 
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.device)
        self.dw[self.ptr] = torch.tensor(dw, dtype=torch.bool)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.device, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
    
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
        self.a[:self.size,] = torch.from_numpy(np.load(f"{path}/a.npy")).to(self.device)
        r_cpu = torch.from_numpy(np.load(f"{path}/r.npy")).reshape((-1,1))
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
        self.dw[:self.size,] = torch.from_numpy(np.load(f"{path}/dw.npy")).to(self.device)
            