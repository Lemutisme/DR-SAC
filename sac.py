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


######################################################
# NOTE: 
# Actor, Double_Q_Critic, V_Critic: MLP
# Transition learner: VAE
# Three robust dual optimization options:
# 1. beta, minimize loss.mean()
# 2. functional g, minimize scalar loss, equivalent to replace beta with g(s,a)
# 3. independent optimize, too slow
######################################################

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hid_dim, hid_layers, hidden_activation=nn.ReLU, output_activation=nn.ReLU):
        super(Actor, self).__init__()
        layers = [state_dim] + [hid_dim, hid_dim] * hid_layers

        self.a_net = build_net(layers, hidden_activation, output_activation)
        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
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

        '''↓↓↓ Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf ↓↓↓'''
        a = torch.tanh(u)
        if with_logprob:
            # Get probability density of logp_pi_a from probability density of u:
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
        else:
            logp_pi_a = None

        return a * self.max_action, logp_pi_a

class V_Critic(nn.Module):
    def __init__(self, state_dim, hid_dim, hid_layers):
        super(V_Critic, self).__init__()
        self.state_dim = state_dim

        layers = [state_dim] + [hid_dim, hid_dim] * hid_layers + [1]
        self.V = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state):
        output = self.V(state)
        return output
    
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

class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, hid_layers: int, num_critics: int,
    ):
        super().__init__()
        layers = [VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
                  nn.ReLU()]
        for _ in range(hid_layers):
            layers.extend([VectorizedLinear(hidden_dim, hidden_dim, num_critics), 
                           nn.ReLU()])
        layers.append( VectorizedLinear(hidden_dim, 1, num_critics))
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

class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hid_dim, hid_layers):
        super(Double_Q_Critic, self).__init__()
        layers = [state_dim + action_dim] + [hid_dim, hid_dim] * hid_layers + [1]

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)   

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)            
        return q1, q2

class TransitionVAE(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, hidden_layers, latent_dim):
        super(TransitionVAE, self).__init__()
        # Encoder layers
        e_layers = [state_dim * 2 + action_dim] + [hidden_dim, hidden_dim] * hidden_layers
        self.encoder = build_net(e_layers, nn.ReLU, nn.Identity)
        self.e_mu = nn.Linear(e_layers[-1], latent_dim)
        self.e_logvar = nn.Linear(e_layers[-1], latent_dim)

        # Decoder layers: out_dim = state_dim
        d_layers = [state_dim + action_dim + latent_dim] + [hidden_dim, hidden_dim] * hidden_layers + [state_dim]
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

class ExpActivation(nn.Module):
    def forward(self, x):
        return torch.exp(x)   

class dual(nn.Module):
    def __init__(self, state_dim, action_dim, hid_dim, hid_layers):
        super(dual, self).__init__()  
        layers = [state_dim + action_dim] + [hid_dim, hid_dim] * hid_layers + [1]

        self.G = build_net(layers, nn.ReLU, ExpActivation)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)          
        return self.G(sa)

class SAC_continuous():
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=self.data_size, device=self.device)

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action, self.hid_dim, self.net_layer).to(self.device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.a_lr)

        self.v_critic = V_Critic(self.state_dim, self.hid_dim, self.net_layer).to(self.device)
        self.v_critic_optimizer = torch.optim.AdamW(self.v_critic.parameters(), lr=self.c_lr)
        if self.soft_update_v:
            self.v_critic_target = copy.deepcopy(self.v_critic)
            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            for p in self.v_critic_target.parameters():
                p.requires_grad = False

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
            self.transition = TransitionVAE(self.state_dim, self.action_dim,
                                            hidden_dim=self.hid_dim, 
                                            hidden_layers=self.net_layer,
                                            latent_dim=self.latent_dim).to(self.device)
            self.trans_optimizer = torch.optim.AdamW(self.transition.parameters(), lr=self.r_lr)
            if self.robust_optimizer == 'beta':
                self.log_beta = nn.Parameter(torch.ones((self.batch_size,1), requires_grad=True, device=self.device) * 1.0)
                self.beta_optimizer = torch.optim.AdamW([self.log_beta], lr=self.b_lr)
            elif self.robust_optimizer == 'functional':
                self.g = dual(self.state_dim, self.action_dim, self.hid_dim, self.net_layer).to(self.device)
                self.g_optimizer = torch.optim.AdamW(self.g.parameters(), lr=self.g_lr)

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

    def vae_loss(self, s_next, s_next_recon, mu, logvar, beta=1):
        recon_loss = F.mse_loss(s_next_recon, s_next, reduction='sum')
        kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        return recon_loss, beta * kl_div

    def dual_func_g(self, s, a, s_next):
        size = s_next.shape[1]
        dual_sa = self.g(s,a)
        return - dual_sa * (torch.logsumexp(-self.v_critic_target(s_next).squeeze(-1)/dual_sa, dim=1, keepdim=True) - math.log(size)) - dual_sa * self.delta  

    def dual_func_beta(self, s_next, beta):
        # Jointly optimize, in tensor
        size = s_next.shape[1]
        return - beta * (torch.logsumexp(-self.v_critic_target(s_next).squeeze(-1)/beta, dim=1, keepdim=True) - math.log(size)) - beta * self.delta     

    def dual_func_ind(self, s_next, beta):
        # Independently optimize, in np.array
        size = s_next.shape[-1]
        v_next = self.v_critic_target(s_next)
        v_next = v_next.cpu().numpy()
        return - beta * (logsumexp(-v_next/beta) - math.log(size)) - beta * self.delta     
    
    def tran_vae_train(self, debug_print, writer, step, iterations, s=None, a=None, s_next=None):
        for _ in range(iterations):
            if s is None:
                s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)
            s_next_recon, mu, std = self.transition(s, a, s_next)
            recon_loss, kl_div = self.vae_loss(s_next, s_next_recon, mu, std)
            tr_loss = recon_loss + kl_div
            self.trans_optimizer.zero_grad()
            tr_loss.backward()
            self.trans_optimizer.step()
            if debug_print:
                print(f"recon_loss: {recon_loss.item()}, kl_div = {kl_div}.")
        if writer:
            writer.add_scalar('tr_loss', tr_loss, global_step=step)
        return tr_loss.item()
        
    def train(self, writer, step):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)
        debug_print = self.debug_print and (step % 1000 == 0)
                
        #----------------------------- ↓↓↓↓↓ Update V Net ↓↓↓↓↓ ------------------------------#
        for params in self.v_critic.parameters():
            params.requires_grad = True
            
        policy_a , log_pi_a = self.actor(s, deterministic=False, with_logprob=True)
        if self.critic_ensemble:
            Q_list = self.q_critic_target(s, policy_a)
            assert Q_list.shape[0] == self.n_critic
            Q_min = Q_list.min(0).values.unsqueeze(-1)
        else:
            current_Q1, current_Q2 = self.q_critic(s, policy_a)
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
            self.tran_vae_train(debug_print, writer, step, 1, s, a, s_next)
            
        #----------------------------- ↓↓↓↓↓ Robust Update ↓↓↓↓↓ ------------------------------#         
            with torch.no_grad():
                s_next_sample = self.transition.sample(s, a, 200)

            #############################################################		
            ### option1: optimize w.r.t beta ###
            # self.log_beta = nn.Parameter(torch.ones_like(self.log_beta, requires_grad=True, device=self.device) * 0.1)
            # self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=self.b_lr)
            if self.robust_optimizer == 'beta':
                for _ in range(5):
                    self.beta = torch.exp(self.log_beta)
                    opt_loss = -self.dual_func_beta(s_next_sample, self.beta)
                    self.beta_optimizer.zero_grad()
                    opt_loss.mean().backward()
                    if debug_print:
                        print(opt_loss.sum().item())
                    self.beta_optimizer.step() 

                V_next_opt = self.dual_func_beta(s_next_sample, torch.exp(self.log_beta)) 
            #############################################################		

            #############################################################		
            ### option2: optimize w.r.t functional g ###
            elif self.robust_optimizer == 'functional':
                for _ in range(10):
                    opt_loss = -self.dual_func_g(s, a, s_next_sample)
                    
                    self.g_optimizer.zero_grad()
                    opt_loss.mean().backward()
                    self.g_optimizer.step() 
        
                    if debug_print:
                        print(opt_loss.mean().item())

                V_next_opt = self.dual_func_g(s, a, s_next_sample) 
            #############################################################		

            #############################################################		
            # option3: Use scipy.optimize to separately optimize
            elif self.robust_optimizer == 'separate':
                s_next_sample = s_next_sample.cpu().numpy()
                V_next_opt_acc = np.zeros((self.batch_size, 1))
                for i in range(s_next_sample.shape[0]):
                    opt = minimize_scalar(fun=lambda beta:-self.dual_func_ind(s_next_sample[i], beta), method='Bounded', bounds=(1e-4, 1.0))
                    V_next_opt_acc[i] = -opt.fun
                V_next_opt_acc = torch.from_numpy(V_next_opt_acc).float()
                V_next_opt_acc = V_next_opt_acc.to('cuda' if torch.cuda.is_available() else 'cpu')
            ############################################################		

            else:
                raise NotImplementedError  

        #----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        for params in self.q_critic.parameters():
            params.requires_grad = True

        with torch.no_grad():
            V_next = self.v_critic_target(s_next) if self.soft_update_v else self.v_critic(s_next)
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
        #########################################
        ### Jπ(θ) = E[α * logπ(a|s) - Q(s,a)] ###
        a_loss = (self.alpha * log_pi_a - Q_min).mean()
        #########################################
            
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
        if self.soft_update_v:
            for param, target_param in zip(self.v_critic.parameters(), self.v_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, EnvName):
        model_dir = Path(f"./models/SAC_model/{EnvName}")
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.actor.state_dict(), model_dir / f"actor.pth")
        torch.save(self.q_critic.state_dict(), model_dir / f"q.pth")
        torch.save(self.v_critic.state_dict(), model_dir / f"v.pth")
        if self.robust:
            torch.save(self.transition.state_dict(), model_dir / f"tran.pth")

    def load(self, EnvName, load_path):
        model_dir = Path(get_original_cwd())/f"{load_path}/models/SAC_model/{EnvName}"

        self.actor.load_state_dict(torch.load(model_dir / f"actor.pth", map_location=self.device, weights_only=True))
        self.q_critic.load_state_dict(torch.load(model_dir / f"q.pth", map_location=self.device, weights_only=True))
        self.v_critic.load_state_dict(torch.load(model_dir / f"v.pth", map_location=self.device, weights_only=True))
        if self.robust:
            self.transition.load_state_dict(torch.load(model_dir / f"tran.pth", map_location=self.device, weights_only=True))