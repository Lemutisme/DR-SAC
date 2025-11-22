from utils import build_net
from ReplayBuffer import ReplayBuffer

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from pathlib import Path
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
from hydra.utils import get_original_cwd

# ----------------------------- Actor Network ------------------------------ #
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hid_dim: list[int], hid_layers: int, 
                 hidden_activation=nn.ReLU, output_activation=nn.ReLU):
        super(Actor, self).__init__()
        layers = [state_dim] + hid_dim * hid_layers

        self.a_net = build_net(layers, hidden_activation, output_activation)
        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.max_action = max_action
        
        # init as in the EDAC paper
        for layer in self.a_net[0:-1:2]:
            torch.nn.init.constant_(layer.bias, 0.1)
            
        torch.nn.init.uniform_(self.mu_layer.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu_layer.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_std_layer.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_std_layer.bias, -1e-3, 1e-3)

    def forward(self, state, deterministic, with_logprob):
        '''Network with Enforcing Action Bounds'''
        net_out = self.a_net(state)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)  
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        u = mu if deterministic else dist.rsample()

        '''Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf '''
        a = torch.tanh(u)
        if with_logprob:
            # Get probability density of logp_pi_a from probability density of u:
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
        else:
            logp_pi_a = None

        return a * self.max_action, logp_pi_a

# ----------------------------- V Critic Network ------------------------------ #
class V_Critic(nn.Module):
    def __init__(self, state_dim: int, hid_dim: list[int], hid_layers: int):
        super(V_Critic, self).__init__()
        self.state_dim = state_dim

        layers = [state_dim] + hid_dim * hid_layers + [1]
        self.V = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state):
        output = self.V(state)
        return output

# Reference: SAC-N https://arxiv.org/pdf/2110.01548
class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias

# Ensemble of vectorized critics
class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: list[int], hid_layers: int, num_critics: int,
    ):
        super().__init__()
        layers = [VectorizedLinear(state_dim + action_dim, hidden_dim[0], num_critics),
                  nn.ReLU()]
        for _ in range(hid_layers):
            layers.extend([VectorizedLinear(hidden_dim[0], hidden_dim[0], num_critics), 
                           nn.ReLU()])
        layers.append( VectorizedLinear(hidden_dim[0], 1, num_critics))
        self.critic = nn.Sequential(*layers)
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        # [num_critics, batch_size, state_dim + action_dim]
        state_action = state_action.unsqueeze(0).repeat_interleave(
            self.num_critics, dim=0
        )
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values

# Parallel implementation of two critics.
class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hid_dim: list[int], hid_layers: int):
        super(Double_Q_Critic, self).__init__()
        layers = [state_dim + action_dim] + hid_dim * hid_layers + [1]

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)   

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)            
        return q1, q2

# ----------------------------- Generative Transition Models ------------------------------ #
# VAE-based Transition Model
class MLPTransitionVAE(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: list[int], hidden_layer: int, latent_dim: int):
        super(MLPTransitionVAE, self).__init__()
        # Encoder layers
        e_layers = [state_dim * 2 + action_dim] + hidden_dim * hidden_layer
        self.encoder = build_net(e_layers, nn.ReLU, nn.Identity)
        self.e_mu = nn.Linear(e_layers[-1], latent_dim)
        self.e_logvar = nn.Linear(e_layers[-1], latent_dim)

        # Decoder layers: out_dim = state_dim
        d_layers = [state_dim + action_dim + latent_dim] + hidden_dim * hidden_layer + [state_dim]
        self.decoder = build_net(d_layers, nn.ReLU, nn.Identity)
        
        self.latent_dim = latent_dim

    def encode(self, s, a, s_next):
        x = torch.cat([s, a, s_next], dim=1)
        h = self.encoder(x)
        mu = self.e_mu(h)
        logvar = self.e_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, s, a, z):
        x = torch.cat([s, a, z], dim=-1)
        s_next_recon = self.decoder(x)
        return s_next_recon

    def forward(self, s, a, s_next):
        mu, logvar = self.encode(s, a, s_next)
        z = self.reparameterize(mu, logvar)
        s_next_recon = self.decode(s, a, z)
        return s_next_recon, mu, logvar  

    def sample(self, s, a, num_samples):
        batch_size = s.size(0)
        # Sample latent vectors from the prior with shape (batch, num_samples, latent_dim)
        z = torch.randn(batch_size, num_samples, self.latent_dim, device=s.device)
        # Expand s and a along a new sample dimension so that their shapes become (batch, num_samples, feature_dim)
        s_expanded = s.unsqueeze(1).expand(-1, num_samples, -1)
        a_expanded = a.unsqueeze(1).expand(-1, num_samples, -1)
        s_next_samples = self.decode(s_expanded, a_expanded, z)
        return s_next_samples   

# Helper MLP for time embedding
class TimeMLP(nn.Module):
    """Minimal timestep embedding: scalar t -> vector embedding."""
    def __init__(self, emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(1, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, t):
        # t: (B,) integer timesteps
        t = t.float().unsqueeze(-1)  # (B,1)
        x = F.relu(self.fc1(t))
        x = F.relu(self.fc2(x))
        return x  # (B, emb_dim)

# Diffusion-based Transition Model
class TransitionDiffusion(nn.Module):
    """
    Clean conditional diffusion model for p(s_next | s, a).

    - Train with:   loss = model.loss(s, a, s_next)
    - Sample with:  s_next_samples = model.sample(s, a, num_samples)
    """

    def __init__(self, state_dim, action_dim, hidden_dim, hidden_layer,
                 timesteps=20, time_embed_dim=32, beta_start=1e-5, beta_end=1e-3):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.timesteps = timesteps

        # ----- simple linear beta schedule -----
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).to())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # ----- tiny time embedding -----
        self.time_mlp = TimeMLP(time_embed_dim)

        # ----- noise prediction network ε_θ -----
        # Input: [noisy_s_next, s, a, t_emb]
        in_dim = state_dim + state_dim + action_dim + time_embed_dim
        layers = [in_dim] + hidden_dim * hidden_layer + [state_dim]
        self.eps_net = build_net(layers, nn.ReLU, nn.Identity)

    # ---------- utilities ----------
    def _q_sample(self, s_next_0, t, noise=None):
        """
        Forward diffusion:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(s_next_0)

        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1)            # (B,1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_alpha_bar_t * s_next_0 + sqrt_one_minus_alpha_bar_t * noise

    # ---------- core model ----------
    def forward(self, s, a, noisy_s_next, t):
        """
        Predict noise ε given noisy_s_next and condition (s, a, t).
        """
        t_emb = self.time_mlp(t)  # (B, time_embed_dim)
        x = torch.cat([noisy_s_next, s, a, t_emb], dim=-1)
        eps_pred = self.eps_net(x)
        return eps_pred

    def loss(self, s, a, s_next):
        """
        Standard DDPM denoising loss:
        E_{t, ε} || ε - ε_θ(x_t, s, a, t) ||^2
        """
        B = s_next.size(0)

        # sample random timesteps
        t = torch.randint(0, self.timesteps, (B,), device=s_next.device, dtype=torch.long)
        noise = torch.randn_like(s_next)

        noisy_s_next = self._q_sample(s_next, t, noise)
        eps_pred = self.forward(s, a, noisy_s_next, t)

        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, s, a, num_samples=1):
        """
        Generate s_next samples given (s, a).

        s: (B, state_dim)
        a: (B, action_dim)
        returns: (B, num_samples, state_dim)
        """
        B = s.size(0)

        # expand (s, a) to (B * num_samples, *)
        s_exp = s.unsqueeze(1).expand(B, num_samples, -1).reshape(B * num_samples, -1)
        a_exp = a.unsqueeze(1).expand(B, num_samples, -1).reshape(B * num_samples, -1)

        # start from pure noise
        x = torch.randn(B * num_samples, self.state_dim, device=s.device)

        # naive reverse loop
        for t_step in reversed(range(self.timesteps)):
            t = torch.full((x.size(0),), t_step, device=s.device, dtype=torch.long)
            eps = self.forward(s_exp, a_exp, x, t)

            beta_t = self.betas[t].unsqueeze(-1)
            alpha_t = self.alphas[t].unsqueeze(-1)
            alpha_bar_t = self.alphas_cumprod[t].unsqueeze(-1)

            # DDPM mean estimate
            x0_hat = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            mean = torch.sqrt(alpha_t) * x0_hat + torch.sqrt(1 - alpha_t) * eps

            if t_step > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

        return x.view(B, num_samples, self.state_dim)

# Flow-based Transition Model 
class ConditionalFlow(nn.Module):
    """
    Simplest conditional RealNVP-like flow:
    - Split state into two halves: x1, x2
    - x1 stays unchanged
    - x2 is affine-transformed using s(x1, s, a), t(x1, s, a)
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # split point
        self.split = state_dim // 2

        # MLP that predicts s(x1, s, a) and t(x1, s, a)
        input_dim = self.split + state_dim + action_dim   # x1 + s + a

        layers = [input_dim] + hidden_dim + [2 * (state_dim - self.split)]
        self.st_net = build_net(layers, nn.ReLU, nn.Identity)

    # ------------------------
    # FORWARD: x -> z  (for training)
    # ------------------------
    def forward_flow(self, x, s, a):
        """
        x : s_next_norm (B, state_dim)
        s : s_norm       (B, state_dim)
        a : action       (B, action_dim)
        returns: z, log_det_J
        """
        x1 = x[:, :self.split]
        x2 = x[:, self.split:]

        input = torch.cat([x1, s, a], dim=-1)
        st = self.st_net(input)
        log_s, t = torch.chunk(st, 2, dim=-1)

        s_val = torch.exp(log_s)
        y2 = x2 * s_val + t

        y = torch.cat([x1, y2], dim=-1)

        log_det = log_s.sum(dim=-1)  # diagonal Jacobian

        return y, log_det

    # ------------------------
    # INVERSE: z -> x (for sampling)
    # ------------------------
    def inverse_flow(self, z, s, a):
        """
        z: base Gaussian samples
        """
        z1 = z[:, :self.split]
        z2 = z[:, self.split:]

        inp = torch.cat([z1, s, a], dim=-1)
        st = self.st_net(inp)
        log_s, t = torch.chunk(st, 2, dim=-1)

        s_val = torch.exp(log_s)
        x2 = (z2 - t) / s_val

        x = torch.cat([z1, x2], dim=-1)
        return x

    def log_prob(self, x, s, a):
        z, log_det = self.forward_flow(x, s, a)
        # base Gaussian log prob
        base_log_prob = -0.5 * (z.pow(2).sum(dim=-1) + self.state_dim * math.log(2 * math.pi))
        return base_log_prob + log_det

    def loss(self, s_norm, a, s_next_norm):
        logp = self.log_prob(s_next_norm, s_norm, a)
        return -logp.mean()

    @torch.no_grad()
    def sample(self, s_norm, a, num_samples=1):
        B = s_norm.size(0)

        # expand conditioning
        s_exp = s_norm.unsqueeze(1).expand(B, num_samples, -1).reshape(-1, self.state_dim)
        a_exp = a.unsqueeze(1).expand(B, num_samples, -1).reshape(-1, self.action_dim)

        # base Gaussian
        z = 0.01 * torch.randn(B * num_samples, self.state_dim, device=s_norm.device)

        # inverse flow
        x = self.inverse_flow(z, s_exp, a_exp)

        # reshape
        return x.view(B, num_samples, self.state_dim)

# Score-based Transition Model
class SigmaMLP(nn.Module):
    """
    Tiny embedding network for the noise level sigma.
    Input: log(sigma) as a scalar
    Output: sigma embedding vector
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(1, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, log_sigma):
        # log_sigma: (B, 1)
        x = F.relu(self.fc1(log_sigma))
        x = F.relu(self.fc2(x))
        return x  # (B, emb_dim)


class ConditionalScoreModel(nn.Module):
    """
    Score-based transition model:
    s_theta(x_noisy, s, a, sigma) ≈ ∇_x log q_sigma(x | s, a)

    - x: noisy next-state (normalized)
    - s, a: conditioning (normalized state, raw/normalized action)
    - sigma: noise scale
    """

    def __init__(self, state_dim, action_dim, hidden_dim, hidden_layer,
                 num_sigmas=8, sigma_min=0.01, sigma_max=0.5, sigma_embed_dim=32):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_sigmas = num_sigmas

        # Discrete noise levels (log-spaced, standard NCSN-style)
        sigmas = torch.logspace(math.log10(sigma_max), math.log10(sigma_min), num_sigmas)
        self.register_buffer("sigmas", sigmas)  # (num_sigmas,)

        # Sigma embedding
        self.sigma_mlp = SigmaMLP(sigma_embed_dim)

        # Score network: input = [x_noisy, s, a, sigma_emb] -> output = score (same dim as state)
        in_dim = state_dim + state_dim + action_dim + sigma_embed_dim
        layers = [in_dim] + hidden_dim * hidden_layer + [state_dim]
        self.score_net = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, x_noisy, s_norm, a, sigma_idx):
        """
        x_noisy: (B, state_dim) - noisy next state
        s_norm: (B, state_dim)  - normalized current state
        a:      (B, action_dim)
        sigma_idx: (B,) long tensor of indices into self.sigmas
        """
        sigma = self.sigmas[sigma_idx].unsqueeze(-1)   # (B, 1)
        log_sigma = torch.log(sigma)                   # (B, 1)
        sigma_emb = self.sigma_mlp(log_sigma)          # (B, sigma_embed_dim)

        inp = torch.cat([x_noisy, s_norm, a, sigma_emb], dim=-1)
        score = self.score_net(inp)                    # (B, state_dim)
        return score

    def loss(self, s_norm, a, s_next_norm):
        """
        Denoising score matching loss:
        E_sigma E_{epsilon}[ || s_theta(x_noisy, s, a, sigma) + epsilon/sigma ||^2 ]

        s_norm:      (B, state_dim)
        a:           (B, action_dim)
        s_next_norm: (B, state_dim)  - clean next state
        """
        B = s_next_norm.size(0)

        # Sample noise levels (indices) uniformly
        sigma_idx = torch.randint(0, self.num_sigmas, (B,), device=s_next_norm.device, dtype=torch.long)
        sigma = self.sigmas[sigma_idx].unsqueeze(-1)  # (B, 1)

        # Add Gaussian noise to s_next_norm
        eps = torch.randn_like(s_next_norm)           # (B, state_dim)
        x_noisy = s_next_norm + sigma * eps           # x = x0 + sigma * eps

        # True score: -(x - x0)/sigma^2 = -eps / sigma
        target_score = -eps / sigma                   # (B, state_dim)

        # Predicted score
        score_pred = self.forward(x_noisy, s_norm, a, sigma_idx)

        # Score matching loss
        loss = F.mse_loss(score_pred, target_score)
        return loss

    @torch.no_grad()
    def sample(self, s_norm, a, num_samples=1, n_steps_each=5, step_scale=0.1):
        """
        Approximate sampling via annealed Langevin dynamics.
        This is a simple, not highly optimized version suitable for ablation.

        s_norm: (B, state_dim)
        a:      (B, action_dim)
        returns: (B, num_samples, state_dim) in normalized space
        """
        B = s_norm.size(0)

        # Expand conditioning
        s_exp = s_norm.unsqueeze(1).expand(B, num_samples, -1).reshape(-1, self.state_dim)
        a_exp = a.unsqueeze(1).expand(B, num_samples, -1).reshape(-1, self.action_dim)

        # Initialize x from standard normal
        x = torch.randn(B * num_samples, self.state_dim, device=s_norm.device)

        # Annealed Langevin dynamics over sigmas (from large to small)
        for k in reversed(range(self.num_sigmas)):
            sigma_k = self.sigmas[k]
            sigma_idx = torch.full((x.size(0),),k, device=s_norm.device, dtype=torch.long)
            # step size scaled with sigma^2
            alpha = step_scale * (sigma_k ** 2)

            for _ in range(n_steps_each):
                score = self.forward(x, s_exp, a_exp, sigma_idx)  # (N, state_dim)
                noise = torch.randn_like(x)

                # Langevin update
                x = x + alpha * score + math.sqrt(2.0 * alpha) * noise

        # Reshape back to (B, num_samples, state_dim)
        x = x.view(B, num_samples, self.state_dim)
        return x

class ExpActivation(nn.Module):
    def forward(self, x):
        return torch.exp(x)   

# To approximate functional set G
class dual(nn.Module):
    def __init__(self, state_dim, action_dim, hid_dim, hid_layers):
        super(dual, self).__init__()  
        layers = [state_dim + action_dim] + hid_dim * hid_layers + [1]
        self.G = build_net(layers, nn.ReLU, ExpActivation)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)          
        return self.G(sa)

# ----------------------------- Soft Actor-Critic (SAC) Agent ------------------------------ #
class SAC_continuous():
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.max_action = torch.tensor(self.max_action, device=self.device)
        self.max_state = torch.tensor(self.max_state, device=self.device)
        
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=self.data_size, device=self.device, EnvIndex=self.env_index)

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action, self.hid_dim, self.net_layer).to(self.device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.a_lr)

        # Option of using V-critic
        if self.use_v:
            self.v_critic = V_Critic(self.state_dim, self.hid_dim, self.net_layer).to(self.device)
            self.v_critic_optimizer = torch.optim.AdamW(self.v_critic.parameters(), lr=self.c_lr)
            self.v_critic_target = copy.deepcopy(self.v_critic)
            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            for p in self.v_critic_target.parameters():
                p.requires_grad = False

        # Option of vectorized / parallel(limit to 2) critics
        if self.critic_ensemble:
            self.q_critic = VectorizedCritic(self.state_dim, self.action_dim, self.hid_dim, self.net_layer, self.n_critic).to(self.device)
        else:
            self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, hid_dim=self.hid_dim, hid_layers=self.net_layer).to(self.device)
        self.q_critic_optimizer = torch.optim.AdamW(self.q_critic.parameters(), lr=self.c_lr)    
         
        self.q_critic_target = copy.deepcopy(self.q_critic)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False     
            
        if self.robust:    
            print('This is a robust policy.')
            # Generative model for transition dynamics
            if self.gen_type == 'vae':
                self.transition = MLPTransitionVAE(self.state_dim, self.action_dim, hidden_dim=self.hid_dim, hidden_layer=self.net_layer, latent_dim=self.vae_latent_dim).to(self.device)
            elif self.gen_type == 'diffusion':
                self.transition = TransitionDiffusion(self.state_dim, self.action_dim, hidden_dim=self.hid_dim, hidden_layer=self.net_layer, 
                                                      timesteps=self.diffusion_timesteps, time_embed_dim=self.time_embed_dim, beta_start=self.beta_start, beta_end=self.beta_end).to(self.device)
            elif self.gen_type == 'flow':
                self.transition = ConditionalFlow(self.state_dim, self.action_dim, hidden_dim=self.hid_dim).to(self.device)
            elif self.gen_type == 'score':
                self.transition = ConditionalScoreModel(self.state_dim, self.action_dim, hidden_dim=self.hid_dim, hidden_layer=self.net_layer).to(self.device)
            self.trans_optimizer = torch.optim.AdamW(self.transition.parameters(), lr=self.r_lr)
            # Robust optimization options
            if self.robust_optimizer == 'beta':
                self.log_beta = nn.Parameter(torch.ones((self.batch_size,1), requires_grad=True, device=self.device) * 1.0)
                self.beta_optimizer = torch.optim.AdamW([self.log_beta], lr=self.b_lr)
            elif self.robust_optimizer == 'functional':
                self.g = dual(self.state_dim, self.action_dim, self.hid_dim, self.net_layer).to(self.device)
                self.g_optimizer = torch.optim.AdamW(self.g.parameters(), lr=self.g_lr)

        # Option for auto-tune temperature alpha
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=self.device)
            # We learn log_alpha instead of alpha to ensure alpha>0
            self.log_alpha = torch.tensor(math.log(self.alpha), dtype=float, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.AdamW([self.log_alpha], lr=self.c_lr)      

    def select_action(self, state, deterministic):
        # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis,:]).to(self.device)
            a, _ = self.actor(state, deterministic, with_logprob=False)
        return a.cpu().numpy()[0]

    def dual_func_g(self, s, a, s_next):
        size = s_next.shape[1]
        dual_sa = self.g(s,a)
        return - dual_sa * (torch.logsumexp(-self.v_critic_target(s_next).squeeze(-1)/dual_sa, dim=1, keepdim=True) - math.log(size)) - dual_sa * self.delta     

    def dual_func_ind(self, s_next, beta):
        # Independently optimize, in np.array
        size = s_next.shape[-1]
        v_next = self.v_critic_target(s_next)
        v_next = v_next.cpu().numpy()
        return - beta * (logsumexp(-v_next/beta) - math.log(size)) - beta * self.delta

    def vae_loss(self, s_next, s_next_recon, mu, logvar, beta=1):
        recon_loss = F.mse_loss(s_next_recon, s_next)
        kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return recon_loss, beta * kl_div
    
    def vae_train(self, debug_print, writer, step, iterations, sample=True, s_norm=None, s_next_norm=None, a=None):
        for _ in range(iterations):
            if sample:
                s, a, r, s_next, dw, s_norm, s_next_norm = self.replay_buffer.sample(self.batch_size)
            s_next_recon, mu, logvar = self.transition(s_norm, a, s_next_norm)
            recon_loss, kl_div = self.vae_loss(s_next_norm, s_next_recon, mu, logvar)
            tr_loss = recon_loss + kl_div
            self.trans_optimizer.zero_grad()
            tr_loss.backward()
            self.trans_optimizer.step()
        if debug_print:
            print(f"VAE Train-- recon_loss: {recon_loss.item()}, kl_div = {kl_div}.")
        if writer:
            writer.add_scalar('tr_loss', tr_loss, global_step=step)
        return tr_loss.item()
    
    def diffusion_train(self, debug_print, writer, step, iterations, sample=True, s_norm=None, s_next_norm=None, a=None):
        """     
        Assumes self.transition is a SimpleTransitionDiffusion (or similar)
        and self.trans_optimizer is its optimizer.
        """
        for _ in range(iterations):
            if sample:
                # sample from replay, same as VAE
                s, a, r, s_next, dw, s_norm, s_next_norm = self.replay_buffer.sample(self.batch_size)

            # diffusion loss: E_{t,eps} || eps - eps_theta(x_t, s, a, t) ||^2
            tr_loss = self.transition.loss(s_norm, a, s_next_norm)

            self.trans_optimizer.zero_grad()
            tr_loss.backward()
            self.trans_optimizer.step()

        if debug_print:
            print(f"Diffusion Train --loss: {tr_loss.item()}")

        if writer:
            writer.add_scalar('tr_loss', tr_loss, global_step=step)

        return tr_loss.item()

    def flow_train(self, debug_print, writer, step, iterations, sample=True, s_norm=None, s_next_norm=None, a=None):
        for _ in range(iterations):
            if sample:
                s, a, r, s_next, dw, s_norm, s_next_norm = self.replay_buffer.sample(self.batch_size)
            tr_loss = self.transition.loss(s_norm, a, s_next_norm)

            self.trans_optimizer.zero_grad()
            tr_loss.backward()
            self.trans_optimizer.step()

        if debug_print:
            print(f"Flow Train --loss: {tr_loss.item()}")
            # original log prob
            logp_true = self.transition.log_prob(s_next_norm, s_norm, a).mean().item()

            # shuffled conditioning
            perm = torch.randperm(s_norm.size(0))
            logp_shuffled = self.transition.log_prob(s_next_norm, s_norm[perm], a[perm]).mean().item()

            print("logp_true:", logp_true, "logp_shuffled:", logp_shuffled)


        if writer:
            writer.add_scalar("flow_loss", tr_loss, global_step=step)

        return tr_loss.item()
    
    def score_train(self, debug_print, writer, step, iterations,
                sample=True, s_norm=None, s_next_norm=None, a=None):
        """
        Train the score-based transition model (ConditionalScoreModel)

        Assumes:
            - self.transition is a ConditionalScoreModel
            - self.trans_optimizer is its optimizer
            - replay_buffer.sample returns (..., s_norm, s_next_norm)
        """
        for _ in range(iterations):
            if sample:
                s, a, r, s_next, dw, s_norm, s_next_norm = self.replay_buffer.sample(self.batch_size)

            tr_loss = self.transition.loss(s_norm, a, s_next_norm)

            self.trans_optimizer.zero_grad()
            tr_loss.backward()
            self.trans_optimizer.step()

        if debug_print:
            print(f"score_tr_loss: {tr_loss.item()}")

        if writer:
            writer.add_scalar('tr_loss', tr_loss, global_step=step)

        return tr_loss.item()
    
    def bc_loss(self, debug_print, writer, step, iterations):
        scores = []
        for _ in range(iterations):
            s, a, r, s_next, dw, _, _ = self.replay_buffer.sample(self.batch_size)
            debug_print = self.debug_print and (step % 1000 == 0)
            
            policy_a , _ = self.actor(s, deterministic=True, with_logprob=False)
            bc_loss = F.mse_loss(policy_a, a)
            
            # self.actor_optimizer.zero_grad()
            # bc_loss.backward()
            # self.actor_optimizer.step()
            if debug_print:
                print(f"bc_loss: {bc_loss.item()}")
        if writer:
            writer.add_scalar('bc_loss', bc_loss, global_step=step)
        return bc_loss

    def behavior_clone_step(self, writer=None, global_step=0):
        """
        Single gradient step of pure behavior cloning on the offline dataset.
        """
        s, a, _, _, _, _, _ = self.replay_buffer.sample(self.batch_size)
        policy_a, _ = self.actor(s, deterministic=True, with_logprob=False)
        bc_loss = F.mse_loss(policy_a, a)

        self.actor_optimizer.zero_grad()
        bc_loss.backward()
        self.actor_optimizer.step()

        if writer is not None:
            writer.add_scalar('bc_pretrain_loss', bc_loss, global_step=global_step)
        return bc_loss.item()
        
    def train(self, writer, step):
        s, a, r, s_next, dw, s_norm, s_next_norm = self.replay_buffer.sample(self.batch_size)
        debug_print = self.debug_print and (step % 1000 == 0)
                
        #----------------------------- ↓↓↓↓↓ Update V Net ↓↓↓↓↓ ------------------------------#
        if self.use_v:
            for params in self.v_critic.parameters():
                params.requires_grad = True
                
            policy_a , log_pi_a = self.actor(s, deterministic=False, with_logprob=True)
            if self.critic_ensemble:
                Q_list = self.q_critic_target(s, policy_a)
                assert Q_list.shape[0] == self.n_critic
                Q_min = Q_list.min(0).values.unsqueeze(-1)
            else:
                current_Q1, current_Q2 = self.q_critic_target(s, policy_a)
                Q_min = torch.min(current_Q1, current_Q2)
            ### V(s) = E_pi(Q(s,a) - α * logπ(a|s)) ###
            target_V = (Q_min - self.alpha * log_pi_a).detach()

            current_V = self.v_critic(s)
            v_loss = F.mse_loss(current_V, target_V)

            self.v_critic_optimizer.zero_grad()
            v_loss.backward()
            self.v_critic_optimizer.step()
            if debug_print:
                print(f"v_loss: {v_loss.item()}")
            if writer:
                writer.add_scalar('v_loss', v_loss, global_step=step)        
                    
            for params in self.v_critic.parameters():
                params.requires_grad = False     
        
        #----------------------------- ↓↓↓↓↓ Update R Net ↓↓↓↓↓ ------------------------------#        
        if self.robust: 
            if self.gen_type == 'vae':
                self.vae_train(debug_print, writer, step, iterations=1, sample=False,
                               s_norm=s_norm, s_next_norm=s_next_norm, a=a)
            elif self.gen_type == 'diffusion':
                self.diffusion_train(debug_print, writer, step, iterations=1, sample=False,
                                     s_norm=s_norm, s_next_norm=s_next_norm, a=a)
            elif self.gen_type == 'flow':
                self.flow_train(debug_print, writer, step, iterations=1, sample=False,
                                s_norm=s_norm, s_next_norm=s_next_norm, a=a)
            elif self.gen_type == 'score':
                self.score_train(debug_print, writer, step, iterations=1, sample=False,
                                 s_norm=s_norm, s_next_norm=s_next_norm, a=a)

        #----------------------------- ↓↓↓↓↓ Robust Update ↓↓↓↓↓ ------------------------------#         
            with torch.no_grad():
                s_next_sample_norm = self.transition.sample(s_norm, a, 200)
                s_next_sample = self.replay_buffer.state_reverse(s_next_sample_norm)

            #############################################################		
            ### option1: optimize w.r.t functional g ###
            if self.robust_optimizer == 'functional':
                for i in range(5):
                    opt_loss = -self.dual_func_g(s, a, s_next_sample) 
                    self.g_optimizer.zero_grad()
                    opt_loss.mean().backward()
                    self.g_optimizer.step() 
        
                    if debug_print:
                        print(opt_loss.mean().item())

                V_next_opt = self.dual_func_g(s, a, s_next_sample) 
            #############################################################		

            #############################################################		
            # option2: Use scipy.optimize to separately optimize
            elif self.robust_optimizer == 'separate':
                V_next_opt = np.zeros((self.batch_size, 1))
                for i in range(s_next_sample.shape[0]):
                    opt = minimize_scalar(fun=lambda beta:-self.dual_func_ind(s_next_sample[i], beta), method='Bounded', bounds=(1e-4, 1.0))
                    V_next_opt[i] = -opt.fun
                V_next_opt = torch.from_numpy(V_next_opt).float()
                V_next_opt = V_next_opt.to('cuda' if torch.cuda.is_available() else 'cpu')
            ############################################################		

            else:
                raise NotImplementedError  

        #----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        for params in self.q_critic.parameters():
            params.requires_grad = True

        with torch.no_grad():
            if self.use_v:
                V_next = self.v_critic_target(s_next)
            else:
                a_next , log_pi_a_next = self.actor(s_next, deterministic=False, with_logprob=True)
                if self.critic_ensemble:
                    Q_list = self.q_critic_target(s_next, a_next)
                    assert Q_list.shape[0] == self.n_critic
                    Q_next = Q_list.min(0).values.unsqueeze(-1)
                else:
                    current_Q1, current_Q2 = self.q_critic_target(s_next, a_next)
                    Q_next = torch.min(current_Q1, current_Q2)
               
                V_next = (Q_next - self.alpha * log_pi_a_next)
            #############################################################		
            ### Q(s, a) = r + γ * (1 - done) * V(s') ###
            if self.robust:
                target_Q = r + (~dw) * self.gamma * V_next_opt
                if debug_print:
                    print(((V_next_opt - V_next) / V_next).norm().item()) # difference of robust update
            else:
                target_Q = r + (~dw) * self.gamma * V_next
            #############################################################

        # Get current Q estimates and JQ(θ)
        if self.critic_ensemble:        
            current_Q = self.q_critic(s, a)
            # [ensemble_size, batch_size] - [1, batch_size]
            q_loss = ((current_Q - target_Q.view(1, -1)) ** 2).mean(dim=1).sum(dim=0)
        else:
            current_Q1, current_Q2 = self.q_critic(s, a)
            q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()
        if debug_print:
            print(f"q_loss: {q_loss.item()}")
        if writer:
            writer.add_scalar('q_loss', q_loss, global_step=step)
            
        for params in self.q_critic.parameters():
            params.requires_grad = False

        #----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Entropy Regularization
        # Note that the entropy term is not included in the loss function
        if not self.use_v:
            policy_a , log_pi_a = self.actor(s, deterministic=False, with_logprob=True)
            if self.critic_ensemble:
                Q_list = self.q_critic(s, policy_a)
                assert Q_list.shape[0] == self.n_critic
                Q_min = Q_list.min(0).values.unsqueeze(-1)
            else:
                current_Q1, current_Q2 = self.q_critic(s, policy_a)
                Q_min = torch.min(current_Q1, current_Q2)
        #########################################
        ### Jπ(θ) = E[α * logπ(a|s) - Q(s,a)] ###
        a_loss = (self.alpha * log_pi_a - Q_min).mean()
        #########################################
        
        policy_a , _ = self.actor(s, deterministic=True, with_logprob=False)
        bc_loss = F.mse_loss(policy_a, a)
        
        a_loss = (1 - self.bc_weight) * a_loss + self.bc_weight * bc_loss
            
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()
        if debug_print:
            print(f"a_loss: {a_loss.item()}")
        if writer:
            writer.add_scalar('a_loss', a_loss, global_step=step)

        #----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha: # Adaptive alpha SAC
            # We learn log_alpha instead of alpha to ensure alpha>0
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp() 
            if debug_print:
                print(f"alpha = {self.alpha.item()}\n")

        #----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        if self.use_v:
            for param, target_param in zip(self.v_critic.parameters(), self.v_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, EnvName):
        model_dir = Path(f"./models/SAC_model/{EnvName}")
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.actor.state_dict(), model_dir / f"actor.pth")
        torch.save(self.q_critic.state_dict(), model_dir / f"q.pth")
        if self.use_v:
            torch.save(self.v_critic.state_dict(), model_dir / f"v.pth")
        if self.robust:
            torch.save(self.transition.state_dict(), model_dir / f"tran.pth")

    def load(self, EnvName, load_path):
        model_dir = Path(get_original_cwd())/f"{load_path}/models/SAC_model/{EnvName}"
        
        state_dict = torch.load(model_dir / f"q.pth", weights_only=True)

        self.actor.load_state_dict(torch.load(model_dir / f"actor.pth", map_location=self.device, weights_only=True))
        self.q_critic.load_state_dict(torch.load(model_dir / f"q.pth", map_location=self.device, weights_only=True))
        if self.use_v:
            self.v_critic.load_state_dict(torch.load(model_dir / f"v.pth", map_location=self.device, weights_only=True))
        if self.robust:
            self.transition.load_state_dict(torch.load(model_dir / f"tran.pth", map_location=self.device, weights_only=True))