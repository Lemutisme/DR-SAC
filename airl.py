import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import gymnasium as gym
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from sac import Actor, V_Critic, Double_Q_Critic, MLPTransitionVAE, dual
from utils import build_net, evaluate_policy
from ReplayBuffer import ReplayBuffer


class Discriminator(nn.Module):
    """
    AIRL-style score with disentangled structure:
    f(s,a,s') = g_theta(s) + gamma * h_phi(s') - h_phi(s)
    NOTE: g_theta(s) is state-only for transferability.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, hidden_layers=2, gamma=0.99):
        super(Discriminator, self).__init__()
        self.gamma = gamma

        # Reward head g_theta(s)  <-- state-only (no action!)
        g_layers = [state_dim] + [hidden_dim] * hidden_layers + [1]
        self.g_theta = build_net(g_layers, nn.ReLU, nn.Identity)

        # Potential head h_phi(s)
        h_layers = [state_dim] + [hidden_dim] * hidden_layers + [1]
        self.h_phi = build_net(h_layers, nn.ReLU, nn.Identity)

    def forward(self, state, action, next_state):
        """
        Return AIRL score f(s,a,s') (NOT the reward):
        f = g(s) + gamma*h(s') - h(s)
        """
        g_val   = self.g_theta(state)
        h_s     = self.h_phi(state)
        h_s_next= self.h_phi(next_state)
        f_val   = g_val + self.gamma * h_s_next - h_s
        return f_val

    def get_reward(self, state, action, next_state):
        """
        Extract the reward signal for policy training
        """
        return self.forward(state, action, next_state)

class VAEEnsemble(nn.Module):
    """
    Ensemble of VAEs for uncertainty-aware nominal model estimation
    """
    def __init__(self, state_dim, action_dim, num_models=5, hidden_dim=256, 
                 hidden_layers=2, latent_dim=10):
        super(VAEEnsemble, self).__init__()
        self.num_models = num_models
        self.state_dim = state_dim

        # Create ensemble of VAEs
        self.models = nn.ModuleList([
            MLPTransitionVAE(state_dim, action_dim, 
                         hidden_dim=hidden_dim, hidden_layer=hidden_layers, 
                         latent_dim=latent_dim)
            for _ in range(num_models)
        ])

    def forward(self, s, a, s_next, model_idx=None):
        """Forward pass through specific model or all models"""
        if model_idx is not None:
            return self.models[model_idx](s, a, s_next)
        else:
            outputs = []
            for model in self.models:
                outputs.append(model(s, a, s_next))
            return outputs

    def sample_next_states(self, s, a, num_samples=10):
        """
        Sample next states from the ensemble
        Returns: (batch_size, num_samples, state_dim)
        """
        all_samples = []

        # Randomly choose which model to use for each of the num_samples needed
        model_indices = np.random.choice(self.num_models, size=num_samples, replace=True)

        # For each model in the ensemble, generate the number of samples it was chosen for
        for i in range(self.num_models):
            count = np.count_nonzero(model_indices == i)
            if count > 0:
                # Generate 'count' samples from the i-th model
                samples = self.models[i].sample(s, a, count)
                all_samples.append(samples)
        
        # Handle the edge case where num_samples is 0
        if not all_samples:
            return torch.empty(s.size(0), 0, self.state_dim, device=s.device)

        # Concatenate the samples from all chosen models
        combined_samples = torch.cat(all_samples, dim=1)
        return combined_samples

    def get_uncertainty(self, s, a, num_samples=50):
        """
        Compute predictive variance as measure of epistemic uncertainty
        """
        samples = self.sample_next_states(s, a, num_samples)
        # Compute variance across samples
        variance = torch.var(samples, dim=1)  # (batch_size, state_dim)
        return variance.mean(dim=-1, keepdim=True)  # (batch_size, 1)


class ExpertDataset:
    """
    Expert demonstration dataset handler
    """
    def __init__(self, path: str, device: str = 'cuda'):
        self.device = device
        self.load_demonstrations(path)

    def load_demonstrations(self, path: str):
        """Load expert demonstrations from file"""
        # Assuming demonstrations are saved as numpy arrays
        data_path = Path(path)

        if data_path.exists():
            self.states = torch.from_numpy(np.load(data_path / 's.npy')).to(self.device)
            self.actions = torch.from_numpy(np.load(data_path / 'a.npy')).to(self.device)
            self.next_states = torch.from_numpy(np.load(data_path / 's_next.npy')).to(self.device)

            # Optional: rewards and dones if available
            if (data_path / 'r.npy').exists():
                self.rewards = torch.from_numpy(np.load(data_path / 'r.npy')).to(self.device)
            if (data_path / 'dw.npy').exists():
                self.dones = torch.from_numpy(np.load(data_path / 'dw.npy')).to(self.device)

            self.size = len(self.states)
        else:
            raise FileNotFoundError(f"Expert dataset not found at {path}")

    def sample(self, batch_size: int):
        """Sample a batch of expert transitions"""
        indices = torch.randint(0, self.size, (batch_size,))
        return (
            self.states[indices],
            self.actions[indices],
            self.next_states[indices]
        )


class DR_AIL:
    """
    Distributionally Robust Adversarial Imitation Learning
    """
    def __init__(self, **kwargs):
        # Store all hyperparameters
        self.__dict__.update(kwargs)

        # Initialize discriminator (reward function)
        self.discriminator = Discriminator(
            self.state_dim, self.action_dim,
            hidden_dim=self.disc_hidden_dim,
            hidden_layers=self.disc_hidden_layers,
            gamma=self.gamma
        ).to(self.device)
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.disc_lr
        )

        if self.robust:
            print("Initializing DR-AIL with ROBUST components.")
            # Initialize VAE ensemble for uncertainty-aware nominal model
            self.vae_ensemble = VAEEnsemble(
                self.state_dim, self.action_dim,
                num_models=self.num_vae_models,
                hidden_dim=self.vae_hidden_dim,
                hidden_layers=self.vae_hidden_layers,
                latent_dim=self.latent_dim
            ).to(self.device)
            self.vae_optimizer = torch.optim.Adam(
                self.vae_ensemble.parameters(), lr=self.vae_lr
            )
            # Initialize Lagrange multiplier network for β(s,a)
            self.beta_network = dual(
                self.state_dim, self.action_dim,
                hid_dim=self.net_width,
                hid_layers=self.net_layers
            ).to(self.device)
            self.beta_optimizer = torch.optim.Adam(
                self.beta_network.parameters(), lr=self.beta_lr
            )
        else:
            print("Initializing standard AIRL baseline.")
            self.vae_ensemble = None
            self.beta_network = None

        # Initialize policy networks (actor)
        self.actor = Actor(
            self.state_dim, self.action_dim,
            max_action=self.max_action,
            hid_dim=self.net_width,
            hid_layers=self.net_layers
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr
        )

        # Initialize value networks (V and Q critics)
        self.v_critic = V_Critic(
            self.state_dim,
            hid_dim=self.net_width,  # <-- Use hid_dim (int)
            hid_layers=self.net_layers
        ).to(self.device)
        self.v_critic_optimizer = torch.optim.Adam(
            self.v_critic.parameters(), lr=self.critic_lr
        )
        self.v_critic_target = copy.deepcopy(self.v_critic)

        self.q_critic = Double_Q_Critic(
            self.state_dim, self.action_dim,
            hid_dim=self.net_width,  # <-- Use hid_dim (int)
            hid_layers=self.net_layers
        ).to(self.device)
        self.q_critic_optimizer = torch.optim.Adam(
            self.q_critic.parameters(), lr=self.critic_lr
        )

        # Initialize replay buffer for policy samples
        self.replay_buffer = ReplayBuffer(
            self.state_dim, self.action_dim,
            max_size=int(1e6), device=self.device
        )

        # Temperature parameter for SAC
        if self.adaptive_alpha:
            self.target_entropy = -self.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = self.initial_alpha

    def pretrain_vae_ensemble(self, expert_dataset: ExpertDataset, 
                             num_epochs: int = 100, batch_size: int = 256):
        """
        Phase 1: Pre-train VAE ensemble on bootstrap samples of expert data
        """
        print("Phase 1: Pre-training VAE ensemble...")

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 10  # Number of batches per epoch

            for _ in range(num_batches):
                # Sample batch from expert data
                s, a, s_next = expert_dataset.sample(batch_size)

                # Train each VAE on bootstrap sample
                for i, vae in enumerate(self.vae_ensemble.models):
                    # Bootstrap sampling: sample with replacement
                    bootstrap_idx = torch.randint(0, batch_size, (batch_size,))
                    s_boot = s[bootstrap_idx]
                    a_boot = a[bootstrap_idx]
                    s_next_boot = s_next[bootstrap_idx]

                    # VAE forward pass
                    s_next_recon, mu, logvar = vae(s_boot, a_boot, s_next_boot)

                    # VAE loss (reconstruction + KL divergence)
                    recon_loss = F.mse_loss(s_next_recon, s_next_boot)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    kl_loss = kl_loss / batch_size

                    vae_loss = recon_loss + self.vae_kl_weight * kl_loss

                    # Optimize this VAE
                    self.vae_optimizer.zero_grad()
                    vae_loss.backward()
                    self.vae_optimizer.step()

                    total_loss += vae_loss.item()

            if epoch % 10 == 0:
                avg_loss = total_loss / (num_batches * self.vae_ensemble.num_models)
                print(f"Epoch {epoch}: VAE Loss = {avg_loss:.4f}")

    def z_logits(self, states, actions, next_states):
        """
        z = f - log pi(a|s), where f comes from the discriminator and log pi from the actor.
        """
        with torch.no_grad():
            logp = self.actor.log_prob(states, actions)
        f = self.discriminator(states, actions, next_states)
        return f - logp  # this equals r = log D - log(1-D)

    def set_delta_E(self, m):
        """
        Expert-side KL radius: delta_E = max{ log(1/eta), log(1/rho)/m }.
        Provide self.demo_eta and self.demo_rho via config; defaults below.
        """
        eta = getattr(self, "demo_eta", 0.10)   # target contamination (e.g., 10%)
        rho = getattr(self, "demo_rho", 0.05)   # failure prob for coverage (e.g., 5%)
        delta_E = max(np.log(1.0/eta), np.log(1.0/rho) / float(m))
        self.delta_E = float(delta_E)
        return self.delta_E

    def compute_kl_radius(self, states, actions):
        """
        Compute state-action dependent KL radius based on ensemble uncertainty
        """
        with torch.no_grad():
            uncertainty = self.vae_ensemble.get_uncertainty(states, actions)
            # Scale uncertainty to get KL radius
            delta_kl = self.kl_radius_scale * uncertainty
            return delta_kl

    def update_discriminator(self, expert_batch, policy_batch, K_beta=2):
        """
        Expert-side KL-DRO for positives, standard BCE for negatives.
        L_D = (- L_E^DRO) + E[ell_pi], where
        L_E^DRO = - beta * delta_E - beta * logmeanexp(-ell_E / beta)  (stable)
        """
        s_exp, a_exp, s_next_exp = expert_batch
        s_pol, a_pol, s_next_pol = policy_batch

        # z = f - log pi(a|s)
        z_E = self.z_logits(s_exp, a_exp, s_next_exp)   # expert
        z_P = self.z_logits(s_pol, a_pol, s_next_pol)   # policy

        # Per-sample BCE losses
        with torch.no_grad():
            ell_E = F.softplus(-z_E)    # = -log sigmoid(z_E)

        ell_P = F.softplus( z_P)    # = -log(1 - sigmoid(z_P))
        ell_pi_mean = ell_P.mean()

        # Set / hold delta_E (fixed offline for demos)
        m = s_exp.shape[0]
        if not hasattr(self, "delta_E"):
            self.set_delta_E(m)
        delta_E = torch.as_tensor(self.delta_E, device=z_E.device, dtype=z_E.dtype)

        # Dual ascent on scalar beta_E (batch-wise) to maximize L_E^DRO
        beta_raw = torch.zeros((), device=z_E.device, requires_grad=True)
        beta_lr  = getattr(self, "beta_lr_dro", 1e-2)

        for _ in range(K_beta):
            beta = F.softplus(beta_raw) + 1e-8
            u = -ell_E / beta
            u_max = u.max()
            # log(mean(exp(u))) with stabilization
            log_mean_exp = u_max + torch.log(torch.mean(torch.exp(u - u_max)) + 1e-12)
            L_E_DRO = - beta * delta_E - beta * log_mean_exp
            # gradient ASCENT on beta_raw
            (L_E_DRO).backward(retain_graph=True)
            with torch.no_grad():
                beta_raw += beta_lr * beta_raw.grad
                beta_raw.grad = None

        # Recompute final DRO term (no grad for beta now)
        beta = (F.softplus(beta_raw) + 1e-8).detach()
        u = -ell_E / beta
        u_max = u.max()
        log_mean_exp = u_max + torch.log(torch.mean(torch.exp(u - u_max)) + 1e-12)
        L_E_DRO = - beta * delta_E - beta * log_mean_exp

        # Final discriminator objective
        L_D = (-L_E_DRO) + ell_pi_mean

        # Optional: gradient penalty (kept after the DRO objective)
        if self.use_grad_penalty:
            L_D = L_D + self.compute_gradient_penalty((s_exp, a_exp, s_next_exp),
                                                      (s_pol, a_pol, s_next_pol))

        self.disc_optimizer.zero_grad(set_to_none=True)
        L_D.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.disc_optimizer.step()

        # Logging helpers
        worst_weight_cap = float(np.exp(self.delta_E) / m)
        return {
            "disc_loss": L_D.item(),
            "ell_pi_mean": ell_pi_mean.item(),
            "delta_E": float(self.delta_E),
            "beta_E": float(beta.item()),
            "worst_weight_cap": worst_weight_cap,
        }

    def compute_gradient_penalty(self, expert_batch, policy_batch, lambda_gp=10):
        """
        WGAN-GP style penalty adapted for DR-AIRL:
        - penalize the gradient norm of f(s, a, s') w.r.t. (s, s') only
        because f no longer depends on action when g is state-only.
        - keep the actor out of this path (no log pi here).
        """
        s_exp, a_exp, s_next_exp = expert_batch
        s_pol, a_pol, s_next_pol = policy_batch

        batch_size = s_exp.size(0)
        device = s_exp.device

        # Interpolate ONLY states and next states (actions are irrelevant for f)
        alpha = torch.rand(batch_size, 1, device=device)
        s_interp     = (alpha * s_exp     + (1 - alpha) * s_pol    ).requires_grad_(True)
        s_next_interp= (alpha * s_next_exp+ (1 - alpha) * s_next_pol).requires_grad_(True)

        # Pass a dummy action tensor (zeros) with correct shape; f ignores it
        a_dummy = torch.zeros_like(a_exp)

        # Compute f(s,a,s') on the interpolates
        f_interp = self.discriminator(s_interp, a_dummy, s_next_interp)  # shape [B,1] or [B]

        # Make a scalar (sum over batch) for autograd
        f_sum = f_interp.sum()

        # Gradients w.r.t. s and s'
        grads = torch.autograd.grad(
            outputs=f_sum,
            inputs=[s_interp, s_next_interp],
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )

        # Concatenate and compute L2 norm per sample
        grad_s, grad_sp = grads
        grad_cat = torch.cat([grad_s.view(batch_size, -1), grad_sp.view(batch_size, -1)], dim=1)
        grad_norm = grad_cat.norm(2, dim=1)

        gp = lambda_gp * ((grad_norm - 1.0) ** 2).mean()
        return gp

    def update_critic(self, batch):
        """
        Standard Bellman using AIRL reward r := z = f - log pi.
        """
        states, actions, next_states = batch

        with torch.no_grad():
            rewards = self.z_logits(states, actions, next_states)
            v_next = self.v_critic_target(next_states)
            q_targets = rewards + self.gamma * v_next

        q1, q2 = self.q_critic(states, actions)
        q_loss = F.mse_loss(q1, q_targets) + F.mse_loss(q2, q_targets)

        self.q_critic_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_critic_optimizer.step()
        return q_loss.item()

    def update_robust_critic(self, batch):
        """
        Robust Bellman backup with KL-DRO (entropic risk, stabilized).
        """
        states, actions, next_states = batch

        # Reward r := z
        with torch.no_grad():
            rewards = self.z_logits(states, actions, next_states)

        # Soft value target from actor & Q (SAC-style)
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states, False, True)
            q1_next, q2_next = self.q_critic(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            v_target = q_next - self.alpha * next_log_probs  # shape [B,1] or [B]

        # KL radius per (s,a)
        delta_kl = self.compute_kl_radius(states, actions)  # shape [B,1]
        beta = F.softplus(self.beta_network(states, actions)) + 1e-8  # shape [B,1]

        # Sample next states from the ensemble
        num_samples = 20
        s_next_samples = self.vae_ensemble.sample_next_states(states, actions, num_samples)  # [B,S,D]

        # Compute entropic-risk expectation: -β δ - β[ u_max + log E exp(u-u_max) ], u = -V/β
        with torch.no_grad():
            v_list = []
            for i in range(num_samples):
                v_i = self.v_critic_target(s_next_samples[:, i, :])  # [B,1] or [B]
                v_list.append(v_i.squeeze(-1))
            V = torch.stack(v_list, dim=1)  # [B,S]

            beta_b = beta.squeeze(-1)  # [B]
            u = - V / (beta_b.unsqueeze(1))
            u_max, _ = u.max(dim=1, keepdim=True)           # [B,1]
            log_mean_exp = u_max.squeeze(1) + torch.log(torch.mean(torch.exp(u - u_max), dim=1) + 1e-12)  # [B]
            robust_v_next = - beta_b * delta_kl.squeeze(-1) - beta_b * log_mean_exp  # [B]

            q_targets = rewards.squeeze(-1) + self.gamma * robust_v_next  # [B]

        q1, q2 = self.q_critic(states, actions)           # [B,1]
        q1 = q1.squeeze(-1); q2 = q2.squeeze(-1)

        q_loss = F.mse_loss(q1, q_targets) + F.mse_loss(q2, q_targets)
        beta_loss = -(robust_v_next).mean()               # dual ascent (maximize robust term)

        total_critic_loss = q_loss + beta_loss

        self.q_critic_optimizer.zero_grad(set_to_none=True)
        self.beta_optimizer.zero_grad(set_to_none=True)
        total_critic_loss.backward()
        self.q_critic_optimizer.step()
        self.beta_optimizer.step()

        return q_loss.item()

    def update_actor(self, states):
        """
        Update policy using SAC objective with robust Q-values
        """
        actions, log_probs = self.actor(states, False, True)

        q1, q2 = self.q_critic(states, actions)
        q_values = torch.min(q1, q2)

        actor_loss = (self.alpha * log_probs - q_values).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Update temperature if adaptive
        if self.adaptive_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        return actor_loss.item()

    def update_value_network(self, states):
        """
        Update V-network for SAC
        """
        with torch.no_grad():
            actions, log_probs = self.actor(states, False, True)
            q1, q2 = self.q_critic(states, actions)
            q_values = torch.min(q1, q2)
            v_targets = q_values - self.alpha * log_probs

        v_values = self.v_critic(states)
        v_loss = F.mse_loss(v_values, v_targets)

        self.v_critic_optimizer.zero_grad()
        v_loss.backward()
        self.v_critic_optimizer.step()

        # Soft update target V-network
        for param, target_param in zip(
            self.v_critic.parameters(), 
            self.v_critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return v_loss.item()
    
    def select_action(self, state, deterministic=True):
        """Selects an action from the policy for evaluation."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor(state_tensor, deterministic=deterministic, with_logprob=False)
        return action.cpu().numpy().flatten()

    def train(self, env, expert_dataset: ExpertDataset, 
                num_iterations: int = 100000, batch_size: int = 256,
                eval_freq: int = 1000):
        """
        Main training loop for DR-AIL
        """
        # Phase 1: Pre-train VAE ensemble only if in robust mode
        if self.robust:
            self.pretrain_vae_ensemble(expert_dataset, num_epochs=50)

        # Phase 2: Integrated Adversarial Training
        print("\nPhase 2: Adversarial Training...")

        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        state, _ = env.reset()

        for t in tqdm(range(num_iterations), desc="Training DR-AIL"):
            # Collect experience with current policy
            if t < self.start_timesteps:
                # Random exploration
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action, _ = self.actor(state_tensor, deterministic=False, with_logprob=False)
                    action = action.cpu().numpy().flatten()

            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            # Store transition in replay buffer
            self.replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_timesteps += 1

            if done:
                print(f"Episode {episode_num}: Reward = {episode_reward:.2f}")
                state, _ = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # --- Start training after collecting enough samples
            if t >= self.start_timesteps:
                # 1) Expert batch
                expert_batch = expert_dataset.sample(batch_size)

                # 2) Policy batch from replay buffer
                rb_batch = self.replay_buffer.sample(batch_size)
                policy_states = rb_batch[0]
                policy_actions = rb_batch[1]
                # Next states:
                with torch.no_grad():
                    if self.robust:
                        # Use learned dynamics (ensemble) for DR-AIRL
                        policy_next_states = self.vae_ensemble.sample_next_states(
                            policy_states, policy_actions, num_samples=1
                        ).squeeze(1)
                    else:
                        policy_next_states = rb_batch[3]  # from buffer

                policy_batch = (policy_states, policy_actions, policy_next_states)

                # ---- Discriminator (returns dict)
                disc_stats = self.update_discriminator(expert_batch, policy_batch)
                disc_loss = float(disc_stats["disc_loss"])

                # ---- Critic (use POLICY transitions, not expert)
                critic_batch = policy_batch
                if self.robust:
                    q_loss = self.update_robust_critic(critic_batch)
                else:
                    q_loss = self.update_critic(critic_batch)

                # ---- Actor (use POLICY states)
                actor_states = policy_states
                actor_loss = self.update_actor(actor_states)

                # ---- Value network (use POLICY states)
                v_loss = self.update_value_network(actor_states)

                # ---- Logging
                if t % 100 == 0:
                    print(
                        f"Step {t}: D_loss={disc_loss:.4f}, "
                        f"Q_loss={q_loss:.4f}, A_loss={actor_loss:.4f}, V_loss={v_loss:.4f} | "
                        f"DRO(delta_E={disc_stats['delta_E']:.4f}, "
                        f"beta_E={disc_stats['beta_E']:.4f}, "
                        f"cap={disc_stats['worst_weight_cap']:.6f})"
                    )

            # Evaluation
            if t % eval_freq == 0:
                eval_reward = evaluate_policy(
                    env, self, turns=3
                )
                print(f"\nEvaluation at step {t}: Avg Reward = {eval_reward:.2f}\n")

    def save(self, path: str):
        """Save model checkpoints"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.discriminator.state_dict(), save_path / "discriminator.pth")
        if self.robust:
            torch.save(self.vae_ensemble.state_dict(), save_path / "vae_ensemble.pth")
            torch.save(self.beta_network.state_dict(), save_path / "beta_network.pth")
        torch.save(self.actor.state_dict(), save_path / "actor.pth")
        torch.save(self.q_critic.state_dict(), save_path / "q_critic.pth")
        torch.save(self.v_critic.state_dict(), save_path / "v_critic.pth")

        print(f"Models saved to {save_path}")

    def load(self, path: str):
        """Load model checkpoints"""
        load_path = Path(path)

        self.discriminator.load_state_dict(
            torch.load(load_path / "discriminator.pth", map_location=self.device)
        )
        self.vae_ensemble.load_state_dict(
            torch.load(load_path / "vae_ensemble.pth", map_location=self.device)
        )
        self.actor.load_state_dict(
            torch.load(load_path / "actor.pth", map_location=self.device)
        )
        self.q_critic.load_state_dict(
            torch.load(load_path / "q_critic.pth", map_location=self.device)
        )
        self.v_critic.load_state_dict(
            torch.load(load_path / "v_critic.pth", map_location=self.device)
        )
        self.beta_network.load_state_dict(
            torch.load(load_path / "beta_network.pth", map_location=self.device)
        )

        print(f"Models loaded from {load_path}")


@hydra.main(version_base=None, config_path="config", config_name="dr_ail_config")
def main(cfg: DictConfig):
    """
    Main function to train DR-AIL
    """
    # Set random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create environment
    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high.tolist()  # Action range [-max_action, max_action]
    # min_action = env.action_space.low.tolist() # Action range [-max_action, max_action]
    # max_e_steps = env._max_episode_steps

    # Load expert dataset
    expert_dataset = ExpertDataset(cfg.expert_data_path, device=cfg.device)

    # Initialize DR-AIL agent
    agent = DR_AIL(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        robust=cfg.robust,
        device=cfg.device,
        gamma=cfg.gamma,
        tau=cfg.tau,
        alpha=cfg.alpha,
        initial_alpha=cfg.initial_alpha,
        adaptive_alpha=cfg.adaptive_alpha,
        alpha_lr=cfg.alpha_lr,

        # Network architecture
        net_width=cfg.net_width,
        net_layers=cfg.net_layers,
        disc_hidden_dim=cfg.disc_hidden_dim,
        disc_hidden_layers=cfg.disc_hidden_layers,
        vae_hidden_dim=cfg.vae_hidden_dim,
        vae_hidden_layers=cfg.vae_hidden_layers,
        latent_dim=cfg.latent_dim,
        num_vae_models=cfg.num_vae_models,

        # Learning rates
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        disc_lr=cfg.disc_lr,
        vae_lr=cfg.vae_lr,
        beta_lr=cfg.beta_lr,

        # Training parameters
        start_timesteps=cfg.start_timesteps,
        kl_radius_scale=cfg.kl_radius_scale,
        vae_kl_weight=cfg.vae_kl_weight,
        use_grad_penalty=cfg.use_grad_penalty,
        seed=cfg.seed
    )

    # Train agent
    agent.train(
        env=env,
        expert_dataset=expert_dataset,
        num_iterations=cfg.num_iterations,
        batch_size=cfg.batch_size,
        eval_freq=cfg.eval_freq
    )

    # Save trained model
    agent.save(cfg.save_path)

    env.close()


if __name__ == "__main__":
    main()
