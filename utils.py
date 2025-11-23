import argparse
import random
import numpy as np
import torch.nn as nn

_REWARD_SCALE_OVERRIDES: dict[int, float] = {}

def set_reward_scale_override(env_index: int, scale: float | None):
    """
    Configure a per-environment reward scaling factor.
    Pass None to fall back to the default adapter behavior.
    """
    if scale is None:
        _REWARD_SCALE_OVERRIDES.pop(env_index, None)
    else:
        _REWARD_SCALE_OVERRIDES[env_index] = scale

def build_net(layer_shape, hidden_activation, output_activation):
    '''Build net with for loop'''
    layers = []
    for j in range(len(layer_shape)-1):
        # network shape
        layers += [nn.Linear(layer_shape[j], layer_shape[j+1])]
        # activation type
        act = hidden_activation if j < len(layer_shape)-2 else output_activation
        layers+=[act()]
    return nn.Sequential(*layers)

def Reward_adapter(r, EnvIndex):
    '''Reward engineering for better training'''
    scale_override = _REWARD_SCALE_OVERRIDES.get(EnvIndex)
    # For Pendulum-v0
    if EnvIndex == 0:
        r = (r + 8) / 8
    # For LunarLander-v3
    elif EnvIndex == 2:
        if r <= -100: r = -10
    # For Ant-v5
    elif EnvIndex == 5:
        scale = scale_override if scale_override is not None else 0.01
        r *= scale
        return r
    if scale_override is not None:
        r *= scale_override
    return r

def evaluate_policy(env, agent, turns = 1, seeds_list = [], random_action_prob=0):
    '''Evaluate SAC policy'''
    total_scores = 0
    for j in range(turns):
        # Use given seeds, otherwise reset randomly
        if len(seeds_list) > 0:
            s, _ = env.reset(seed=seeds_list[j])
        else:
            s, _ = env.reset()
            
        done = False
        while not done:
            # Actuator takes random action in some test cases
            if random.random() < random_action_prob:
                a = env.action_space.sample()
            else:
                # Take deterministic actions at test time
                a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)
            total_scores += r
            s = s_next
    return round(total_scores/turns, 2)

def Action_adapter_pos(action, max_action):
    """
    Map actions in [0,1] (Beta outputs) to the environment range [-max_action, max_action].
    """
    action = np.asarray(action)
    scaled = (action * 2.0 - 1.0) * max_action
    return scaled

def evaluate_policy_PPO(env, agent, max_action, turns=5):
    """
    Evaluate PPO agent with deterministic actions.
    """
    scores = []
    for _ in range(turns):
        s, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            a, _ = agent.select_action(s, deterministic=True)
            act = Action_adapter_pos(a, max_action)
            s, r, dw, tr, _ = env.step(act)
            done = (dw or tr)
            ep_r += r
        scores.append(ep_r)
    return float(np.mean(scores))

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
