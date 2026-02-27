from utils import evaluate_policy_PPO as evaluate_policy
from utils import Action_adapter_pos as Action_adapter
from utils import Reward_adapter, str2bool, build_net
from ReplayBuffer import ReplayBuffer

import copy
import math
import hydra
import random
import logging


import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta

from tqdm import tqdm, trange
from pathlib import Path
from scipy.special import logsumexp
from omegaconf import DictConfig, OmegaConf

class BetaActor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(BetaActor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.alpha_head = nn.Linear(net_width, action_dim)
        self.beta_head = nn.Linear(net_width, action_dim)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))

        alpha = F.softplus(self.alpha_head(a)) + 1.0
        beta = F.softplus(self.beta_head(a)) + 1.0

        return alpha, beta

    def get_dist(self,state):
        alpha,beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist

    def deterministic_act(self, state):
        alpha, beta = self.forward(state)
        mode = (alpha) / (alpha + beta)
        return mode

class GaussianActor_musigma(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(GaussianActor_musigma, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.mu_head = nn.Linear(net_width, action_dim)
        self.sigma_head = nn.Linear(net_width, action_dim)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        mu = torch.sigmoid(self.mu_head(a))
        sigma = F.softplus(self.sigma_head(a))
        return mu, sigma

    def get_dist(self, state):
        mu, sigma = self.forward(state)
        dist = Normal(mu,sigma)
        return dist

    def deterministic_act(self, state):
        mu, sigma = self.forward(state)
        return mu

class GaussianActor_mu(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, log_std=0):
        super(GaussianActor_mu, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.mu_head = nn.Linear(net_width, action_dim)
        self.mu_head.weight.data.mul_(0.1)
        self.mu_head.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        mu = torch.sigmoid(self.mu_head(a))
        return mu

    def get_dist(self,state):
        mu = self.forward(state)
        action_log_std = self.action_log_std.expand_as(mu)
        action_std = torch.exp(action_log_std)

        dist = Normal(mu, action_std)
        return dist

    def deterministic_act(self, state):
        return self.forward(state)

class Critic(nn.Module):
    def __init__(self, state_dim, hid_shape, hid_layer):
        super(Critic, self).__init__()

        layers = [state_dim] + list(hid_shape) * hid_layer + [1]
        self.cnet = build_net(layers, nn.Tanh, nn.Identity)

    def forward(self, state):
        return self.cnet(state)

class TransitionVAE(nn.Module):
    def __init__(self, state_dim, action_dim, out_dim, hidden_dim=64, hidden_layers=1, latent_dim=5):
        super(TransitionVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder layers
        e_layers = [state_dim * 2 + action_dim] + [hidden_dim] * hidden_layers
        self.encoder = build_net(e_layers, nn.ReLU, nn.Identity)
        self.e_mu = nn.Linear(e_layers[-1], latent_dim)
        self.e_logvar = nn.Linear(e_layers[-1], latent_dim)

        # Decoder layers
        d_layers = [state_dim + action_dim + latent_dim] + [hidden_dim] * hidden_layers + [out_dim]
        self.decoder = build_net(d_layers, nn.ReLU, nn.Identity)

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

class dual(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, hid_layers):
        super(dual, self).__init__()  
        layers = [state_dim + action_dim] + list(hid_shape) * hid_layers + [1]

        self.G = build_net(layers, nn.ReLU, ExpActivation)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)          
        return self.G(sa)

class ExpActivation(nn.Module):
    def forward(self, x):
        return torch.exp(x)   

class PPO_agent(object):
    def __init__(self, **kwargs):
        # Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)

        # Choose distribution for the actor
        if self.distribution == 'Beta':
            self.actor = BetaActor(self.state_dim, self.action_dim, self.net_width).to(self.device)
        elif self.distribution == 'GS_ms':
            self.actor = GaussianActor_musigma(self.state_dim, self.action_dim, self.net_width).to(self.device)
        elif self.distribution == 'GS_m':
            self.actor = GaussianActor_mu(self.state_dim, self.action_dim, self.net_width).to(self.device)
        else: 
            print('Distribution Error')
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        # Build Critic
        self.critic = Critic(self.state_dim, (self.net_width, self.net_width), self.net_layer).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

        # Build Reward Net and Variable for OPT
        if self.robust:    
            print('This is a robust policy.\n')
            self.transition = TransitionVAE(self.state_dim, self.action_dim, self.state_dim, (self.net_width, self.net_width), self.net_layer).to(self.device)
            self.trans_optimizer = torch.optim.Adam(self.transition.parameters(), lr=self.r_lr)

            # self.log_beta = nn.Parameter(torch.ones((self.batch_size,1), requires_grad=True, device=self.device) * 1.0)
            # self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=self.b_lr)

            self.g = dual(self.state_dim, self.action_dim, (self.net_width, self.net_width), self.net_layer).to(self.device)
            self.g_optimizer = torch.optim.Adam(self.g.parameters(), lr=self.g_lr)

        # Build Trajectory holder
        self.s_holder = np.zeros((self.t_horizon, self.state_dim), dtype=np.float32)
        self.a_holder = np.zeros((self.t_horizon, self.action_dim), dtype=np.float32)
        self.r_holder = np.zeros((self.t_horizon, 1), dtype=np.float32)
        self.s_next_holder = np.zeros((self.t_horizon, self.state_dim), dtype=np.float32)
        self.logprob_a_holder = np.zeros((self.t_horizon, self.action_dim), dtype=np.float32)
        self.done_holder = np.zeros((self.t_horizon, 1), dtype=np.bool_)
        self.dw_holder = np.zeros((self.t_horizon, 1), dtype=np.bool_)
        
        # ReplayBuffer for data generation
        if self.mode == 'generate':
            self.buffer = ReplayBuffer(self.state_dim, self.action_dim, self.data_size, self.device)

    def select_action(self, state, deterministic):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            if deterministic:
                # only used when evaluate the policy. Making the performance more stable
                a = self.actor.deterministic_act(state)
                return a.cpu().numpy()[0], None  # action is in shape (adim, 0)
            else:
                # only used when interact with the env
                dist = self.actor.get_dist(state)
                a = dist.sample()
                a = torch.clamp(a, 0, 1)
                logprob_a = dist.log_prob(a).cpu().numpy().flatten()
                return a.cpu().numpy()[0], logprob_a # both are in shape (adim, 0)

    def vae_loss(self, s_next, s_next_recon, mu, logvar):
        recon_loss = F.mse_loss(s_next_recon, s_next, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def dual_func_g(self, s, a, s_next):
        size = s_next.shape[1]
        dual_sa = self.g(s,a)
        return - dual_sa * (torch.logsumexp(-self.critic(s_next).squeeze(-1)/dual_sa, dim=1, keepdim=True) - math.log(size)) - dual_sa * self.delta  

    def dual_func_beta(self, s_next, beta):
        # Jointly optimize, in tensor
        size = s_next.shape[1]
        return - beta * (torch.logsumexp(-self.critic(s_next).squeeze(-1)/beta, dim=1, keepdim=True) - math.log(size)) - beta * self.delta     

    def dual_func_ind(self, s_next, beta):
        # Independently optimize, in np.array
        size = s_next.shape[-1]
        v_next = self.critic(s_next)
        v_next = v_next.cpu().numpy()
        return - beta * (logsumexp(-v_next/beta) - math.log(size)) - beta * self.delta    

    def train(self, writer, step):
        self.entropy_coef *= self.entropy_coef_decay

        '''Prepare PyTorch data from Numpy data'''
        s = torch.from_numpy(self.s_holder).to(self.device)
        a = torch.from_numpy(self.a_holder).to(self.device)
        r = torch.from_numpy(self.r_holder).to(self.device)
        s_next = torch.from_numpy(self.s_next_holder).to(self.device)
        logprob_a = torch.from_numpy(self.logprob_a_holder).to(self.device)
        done = torch.from_numpy(self.done_holder).to(self.device)
        dw = torch.from_numpy(self.dw_holder).to(self.device)

        if self.robust:
            s_next_recon, mu, logvar = self.transition(s, a, s_next)
            tr_loss = self.vae_loss(s_next, s_next_recon, mu, logvar)
            self.trans_optimizer.zero_grad()
            tr_loss.backward()
            self.trans_optimizer.step()
            if self.debug_print:
                print(f"tr_loss: {tr_loss.item()}")
            if writer:
                writer.add_scalar('tr_loss', tr_loss, global_step=step)
            
            with torch.no_grad():
                s_next_sample = self.transition.sample(s, a, 200)

            if self.robust_optimizer == 'functional':
                for _ in range(5):
                    opt_loss = -self.dual_func_g(s, a, s_next_sample)
                    self.g_optimizer.zero_grad()
                    opt_loss.mean().backward()
                    if self.debug_print:
                        print(opt_loss.mean().item())    
                    self.g_optimizer.step()
            else:
                raise NotImplementedError

            vs_opt = self.dual_func_g(s, a, s_next_sample) 

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)

            '''dw for TD_target and Adv'''
            if self.robust:
                deltas = r + self.gamma * vs_opt * (~dw) - vs
            else:
                deltas = r + self.gamma * vs_ * (~dw) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~mask)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.device)
            td_target = adv + vs
            adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps

        """Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
        a_optim_iter_num = int(math.ceil(s.shape[0] / self.a_optim_batch_size))
        c_optim_iter_num = int(math.ceil(s.shape[0] / self.c_optim_batch_size))

        # Progress bars for training iterations
        k_epochs_bar = trange(self.k_epochs, desc="PPO Epochs", leave=False, position=0)

        for i in k_epochs_bar:
            # Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)
            s, a, td_target, adv, logprob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

            '''update the actor'''
            actor_bar = trange(a_optim_iter_num, desc="Actor Update", leave=False, position=1)
            for j in actor_bar:
                index = slice(j * self.a_optim_batch_size, min((j + 1) * self.a_optim_batch_size, s.shape[0]))
                distribution = self.actor.get_dist(s[index])
                dist_entropy = distribution.entropy().sum(1, keepdim=True)
                logprob_a_now = distribution.log_prob(a[index])
                ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

                actor_bar.set_postfix({'loss': a_loss.mean().item()})

                if writer:
                    writer.add_scalar('a_loss', a_loss.mean(), global_step=step)

            '''update the critic'''
            critic_bar = trange(c_optim_iter_num, desc="Critic Update", leave=False, position=1)
            for j in critic_bar:
                index = slice(j * self.c_optim_batch_size, min((j + 1) * self.c_optim_batch_size, s.shape[0]))
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name,param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()
                
                critic_bar.set_postfix({'loss': c_loss.item()})

                if writer:
                    writer.add_scalar('c_loss', c_loss, global_step=step)

    def put_data(self, s, a, r, s_next, logprob_a, done, dw, idx):
        self.s_holder[idx] = s
        self.a_holder[idx] = a
        self.r_holder[idx] = r
        self.s_next_holder[idx] = s_next
        self.logprob_a_holder[idx] = logprob_a
        self.done_holder[idx] = done
        self.dw_holder[idx] = dw

    def save(self, env_name):
        # params = f"{self.std}_{self.robust}"
        model_dir = Path(f'./models/PPO_model/{env_name}')
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.actor.state_dict(), model_dir / f"actor.pth")
        torch.save(self.critic.state_dict(), model_dir / f"critic.pth")

    def load(self, load_path, env_name):
        model_dir = Path(f'{load_path}/models/PPO_model/{env_name}')

        self.actor.load_state_dict(torch.load(model_dir / f"actor.pth", map_location=self.device, weights_only=True))
        self.critic.load_state_dict(torch.load(model_dir / f"critic.pth", map_location=self.device, weights_only=True))

@hydra.main(version_base=None, config_path="config", config_name="ppo_config")
def main(cfg: DictConfig):
    """
    Main function to train and evaluate a PPO agent on different environments.
    """
    # Set up logger
    log = logging.getLogger(__name__)
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create a summary log file for key information
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.log"

    # Configure file logging manually to ensure it works
    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'))
    log.addHandler(file_handler)

    # Create file handler for summary log
    summary_handler = logging.FileHandler(summary_path)
    summary_handler.setLevel(logging.INFO)
    summary_formatter = logging.Formatter('[%(asctime)s] %(message)s')
    summary_handler.setFormatter(summary_formatter)

    # Create a separate logger for summary information
    summary_logger = logging.getLogger("summary")
    summary_logger.setLevel(logging.INFO)
    summary_logger.addHandler(summary_handler)
    summary_logger.info(f"Starting PPO training with configuration: {cfg.env_name}")

    # Log system information
    import platform
    import torch.cuda
    system_info = {
        "Platform": platform.platform(),
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }

    log.info(f"System information:")
    for key, value in system_info.items():
        log.info(f"  {key}: {value}")
    summary_logger.info(f"System: {system_info['Platform']}, PyTorch: {system_info['PyTorch']}, GPU: {system_info['GPU']}")

    # 1. Define environment names and their abbreviations
    EnvName = [
        'Pendulum-v1',
        "ContinuousCartPole",
        'LunarLanderContinuous-v3',
        'HalfCheetah-v5',
        'Reacher-v5'
    ]
    BrifEnvName = [
        'PV1',
        'CPV0',
        'LLdV3',
        'HCV5',
        'RV5'
    ]

    # Create a config object for compatibility with rest of code
    opt = DictConfig({})
    for key, value in cfg.items():
        if key not in ['hydra']:  # Skip hydra config
            setattr(opt, key, value)

    # 2. Build Training and Evaluation Environments
    env = gym.make(EnvName[opt.env_index])

    if not opt.noise:
        eval_env = gym.make(EnvName[opt.env_index])
    else:
        if opt.env_index == 0:
            eval_env = gym.make("CustomPendulum-v1", std=opt.std) # Add noise when updating angle
        elif opt.env_index == 1:
            eval_env = gym.make("CustomCartPole", std=opt.std) # Add noise when updating angle
        else:
            eval_env = gym.make(EnvName[opt.env_index])


    # 3. Extract environment/state/action info
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])
    opt.max_steps = env._max_episode_steps

    log.info(
        f"Env: {EnvName[opt.env_index]}  "
        f"state_dim: {opt.state_dim}  "
        f"action_dim: {opt.action_dim}  "
        f"max_action: {opt.max_action}  "
        f"min_action: {env.action_space.low[0]}  "
        f"max_steps: {opt.max_steps}"
    )

    summary_logger.info(f"Environment: {EnvName[opt.env_index]}, Action dim: {opt.action_dim}, State dim: {opt.state_dim}")

    # 4. Seed everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info(f"Random Seed: {opt.seed}")

    # 5. Set up TensorBoard (optional)
    writer = None
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        writepath = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "tensorboard"
        writepath.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir=writepath)
        log.info(f"TensorBoard logs will be saved to {writepath}")

    # 7. Create the PPO agent
    agent = PPO_agent(**OmegaConf.to_container(opt, resolve=True))

    # 8. Load existing model if requested
    if opt.load_model:
        # params = f"{opt.std}_{opt.robust}"
        agent.load(opt.load_path, BrifEnvName[opt.env_index])
        log.info(f"Loaded pre-trained model")

    # 9. If rendering is enabled, just evaluate in a loop
    if opt.render:
        while True:
            ep_r = evaluate_policy(env, agent, opt.max_action, turns=1)
            log.info(f"Env: {EnvName[opt.env_index]}, Episode Reward: {ep_r}")

    # 10. If evaluating only, print result
    elif opt.eval_model:
        eval_num = 100
        log.info(f"Evaluating agent across {eval_num} episodes")
        seeds_list = [random.randint(0, 100000) for _ in range(eval_num)] if not hasattr(opt, 'seeds_list') else opt.seeds_list

        scores = []
        # Use tqdm for evaluation progress
        for i in tqdm(range(eval_num), desc="Evaluation Progress", ncols=100):
            score = evaluate_policy(eval_env, agent, opt.max_action, turns=1)
            scores.append(score)
            # Update progress bar with current mean score
            if i > 0 and i % 5 == 0:
                current_mean = np.mean(scores[:i])
                tqdm.write(f"Current mean score after {i} episodes: {current_mean:.2f}")
                # Log intermediate results to summary
                summary_logger.info(f"Intermediate evaluation ({i}/{eval_num}): Mean score = {current_mean:.2f}")

        # Calculate statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        p90_score = np.quantile(scores, 0.9)
        p10_score = np.quantile(scores, 0.1)

        # Save results to output directory
        results_path = output_dir / "results.txt"
        with open(results_path, 'a') as f:
            f.write(f"{[BrifEnvName[opt.env_index], opt.std, opt.robust, mean_score, std_score, p90_score, p10_score]}\n")

        log.info(f"Results: {BrifEnvName[opt.env_index]}, Mean: {mean_score:.2f}, Std: {std_score:.2f}")
        log.info(f"90th percentile: {p90_score:.2f}, 10th percentile: {p10_score:.2f}")
        log.info(f"Results saved to {results_path}")

        # Log final results to summary file
        summary_logger.info("-" * 50)
        summary_logger.info("EVALUATION COMPLETED")
        summary_logger.info(f"Environment: {EnvName[opt.env_index]}")
        summary_logger.info(f"Evaluation over {eval_num} episodes:")
        summary_logger.info(f"  Mean reward: {mean_score:.2f} ± {std_score:.2f}")
        summary_logger.info(f"  90th percentile: {p90_score:.2f}")
        summary_logger.info(f"  10th percentile: {p10_score:.2f}")
        summary_logger.info("-" * 50)

    # 11. Otherwise, proceed with training
    else:
        traj_length = 0
        total_steps = 0
        episode_count = 0
        episode_rewards = []

        if opt.mode == 'continual':
            # Create a progress bar for the total training steps
            with tqdm(total=opt.max_train_steps, desc="Training Progress", ncols=100) as pbar:
                while total_steps < opt.max_train_steps:
                    # Reset environment each episode with an incremented seed 
                    s, info = env.reset(seed=env_seed)
                    env_seed += 1
                    done = False
                    episode_count += 1
                    episode_reward = 0
                    episode_steps = 0

                    # Create a progress bar for steps within this episode
                    episode_pbar = tqdm(total=opt.max_steps, desc=f"Episode {episode_count}", 
                                        leave=False, ncols=100, position=1)

                    # 11-a. Interact & train for one episode
                    while not done:
                        # (i) Select action (stochastic when training)
                        a, logprob_a = agent.select_action(s, deterministic=False)

                        # (ii) Convert action range if needed
                        act = Action_adapter(a, opt.max_action)  # [0,1] -> [-max_action, max_action]

                        # (iii) Step environment
                        s_next, r, dw, tr, info = env.step(act)
                        r = Reward_adapter(r, opt.env_index)       # Custom reward adapter
                        done = (dw or tr)
                        episode_reward += r

                        # (iv) Store transition for PPO
                        agent.put_data(
                            s, a, r, s_next,
                            logprob_a, done, dw, idx=traj_length
                        )

                        # Move to next step
                        s = s_next
                        traj_length += 1
                        total_steps += 1
                        episode_steps += 1

                        # Update progress bars
                        pbar.update(1)
                        episode_pbar.update(1)

                        # Update progress bar description with more info
                        if total_steps % 10 == 0:
                            pbar.set_postfix({
                                'episode': episode_count,
                                'reward': f"{episode_reward:.2f}"
                            })

                        # (v) Update agent if horizon reached
                        if traj_length % opt.t_horizon == 0:
                            # Temporarily close episode progress bar for training updates
                            episode_pbar.set_description("Training agent...")

                            # Train the agent
                            agent.train(writer, total_steps)
                            traj_length = 0

                            # Reset episode progress bar description
                            episode_pbar.set_description(f"Episode {episode_count}")

                        # Learning rate decay for longer training runs
                        if total_steps >= 4e5 and total_steps % 5e3 == 0:
                            agent.a_lr *= 0.95
                            agent.c_lr *= 0.95
                            log.info(f"Decaying learning rates - Actor: {agent.a_lr:.6f}, Critic: {agent.c_lr:.6f}")

                        # (vi) Periodically evaluate and log
                        if total_steps % opt.eval_interval == 0:
                            # Temporarily close progress bars for evaluation
                            episode_pbar.close()
                            pbar.set_description("Evaluating...")

                            score = evaluate_policy(eval_env, agent, opt.max_action, turns=20)

                            if writer is not None:
                                writer.add_scalar('ep_r', score, global_step=total_steps)

                            log_message = (
                                f"EnvName: {EnvName[opt.env_index]}, "
                                f"Steps: {int(total_steps/1000)}k, "
                                f"Episodes: {episode_count}, "
                                f"Score: {score}"
                            )
                            log.info(log_message)

                            # Also log to summary file
                            summary_logger.info(f"Step {total_steps}: Score = {score:.2f}")

                            # Reset progress bar description
                            pbar.set_description("Training Progress")
                            episode_pbar = tqdm(total=opt.max_steps, initial=episode_steps,
                                                desc=f"Episode {episode_count}", 
                                                leave=False, ncols=100, position=1)

                        # (vii) Periodically save model
                        if opt.save_model and total_steps % opt.save_interval == 0:
                            agent.save(BrifEnvName[opt.env_index])
                            log.info(f"Model saved at step {total_steps}")

                    # Close episode progress bar when episode ends
                    episode_pbar.close()

                    # Log episode stats
                    episode_rewards.append(episode_reward)
                    recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards

                    log.info(f"Episode {episode_count} completed with reward {episode_reward:.2f} in {episode_steps} steps")
                    log.info(f"Recent average reward (last {len(recent_rewards)} episodes): {np.mean(recent_rewards):.2f}")
                    summary_logger.info(f"Episode {episode_count}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")
            
        elif opt.mode == 'generate':
            with tqdm(total=opt.max_train_steps, desc="Training Progress", ncols=100) as pbar:
                while total_steps < opt.max_train_steps:
                    # Reset environment each episode with an incremented seed 
                    s, info = env.reset(seed=env_seed)
                    env_seed += 1
                    done = False
                    episode_count += 1
                    episode_steps = 0

                    # Create a progress bar for steps within this episode
                    # episode_pbar = tqdm(total=opt.max_steps, desc=f"Episode {episode_count}", 
                    #                     leave=False, ncols=100, position=1)

                    # 11-a. Interact & train for one episode
                    while not done:
                        if np.random.random() < opt.epsilon:
                            # Sample action directly from environment's action space
                            act = env.action_space.sample()  
                        else:
                            # (i) Select action (stochastic when training)
                            a, logprob_a = agent.select_action(s, deterministic=False)

                            # (ii) Convert action range if needed
                            act = Action_adapter(a, opt.max_action)  # [0,1] -> [-max_action, max_action]

                        # (iii) Step environment
                        s_next, r, dw, tr, info = env.step(act)
                        # r = Reward_adapter(r, opt.env_index)       # Custom reward adapter
                        done = (dw or tr)

                        # (iv) Store transition for PPO
                        agent.buffer.add(s, act, r, s_next, done)
                        
                        # Move to next step
                        s = s_next
                        traj_length += 1
                        total_steps += 1
                        episode_steps += 1

                        # Update progress bars
                        pbar.update(1)
            
            agent.buffer.save()
                    

        # Final evaluation after training
        eval_num = 20
        log.info(f"Training completed. Evaluating across {eval_num} episodes")
        scores = []

        # Create a progress bar for evaluation
        for i in tqdm(range(eval_num), desc="Final Evaluation", ncols=100):
            score = evaluate_policy(eval_env, agent, opt.max_action, turns=1)
            scores.append(score)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        p90_score = np.quantile(scores, 0.9)
        p10_score = np.quantile(scores, 0.1)

        log.info(f"Final evaluation - Mean: {mean_score:.2f}, Std: {std_score:.2f}")
        log.info(f"90th percentile: {p90_score:.2f}, 10th percentile: {p10_score:.2f}")

        # Log final results to summary file
        summary_logger.info("-" * 50)
        summary_logger.info("TRAINING COMPLETED")
        summary_logger.info(f"Environment: {EnvName[opt.env_index]}")
        summary_logger.info(f"Total steps: {total_steps}")
        summary_logger.info(f"Total episodes: {episode_count}")
        summary_logger.info(f"Final evaluation over {eval_num} episodes:")
        summary_logger.info(f"  Mean reward: {mean_score:.2f} ± {std_score:.2f}")
        summary_logger.info(f"  90th percentile: {p90_score:.2f}")
        summary_logger.info(f"  10th percentile: {p10_score:.2f}")
        summary_logger.info("-" * 50)

        # Save final model
        if opt.save_model:
            agent.save(BrifEnvName[opt.env_index])
            log.info(f"Final model saved to models/PPO_model/{BrifEnvName[opt.env_index]}")

    # Close environments
    env.close()
    eval_env.close()

    if writer is not None:
        writer.close()

    return agent

if __name__ == '__main__':
    main()
