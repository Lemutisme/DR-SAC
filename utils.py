import argparse
import random
import torch.nn as nn

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
    # For Pendulum-v0
    if EnvIndex == 0:
        r = (r + 8) / 8
    # For LunarLander-v3
    elif EnvIndex == 2:
        if r <= -100: r = -10
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
