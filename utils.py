import argparse
import random
import torch.nn as nn

def build_net(layer_shape, hidden_activation, output_activation):
    '''Build net with for loop'''
    layers = []
    for j in range(len(layer_shape)-1):
        act = hidden_activation if j < len(layer_shape)-2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
    return nn.Sequential(*layers)

# Reward engineering for better training
def Reward_adapter(r, EnvIndex):
    # For Pendulum-v0
    if EnvIndex == 0:
        r = (r + 8) / 8
    # For LunarLander
    elif EnvIndex == 2:
        if r <= -100: r = -10
    # For Halfcheetah
    # elif EnvIdex == 4:
    #     r = (r + 10) / 20
    # For Hooper
    # elif EnvIdex == 5:
    #     r = (r + 1) / 5
    return r

# def Action_adapter_symm(act, max_action):
#     #from [-1,1] to [-max,max]
#     return  act * max_action

# def Action_adapter_symm_reverse(act, max_action):
#     #from [-max,max] to [-1,1]
#     return  act / max_action

def Action_adapter_pos(a, max_action):
    #from [0,1] to [-max,max]
    return  2 * (a - 0.5) * max_action

def evaluate_policy_SAC(env, agent, turns = 1, seeds_list = [], random_action_prob=0):
    total_scores = 0
    for j in range(turns):
        if len(seeds_list) > 0:
            s, _ = env.reset(seed=seeds_list[j])
        else:
            s, _ = env.reset()
        done = False
        while not done:
            if random.random() < random_action_prob:
                a = env.action_space.sample()
            else:
                # Take deterministic actions at test time
                a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return round(total_scores/turns, 1)

def evaluate_policy_PPO(env, agent, max_action, turns=3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            a, logprob_a = agent.select_action(s, deterministic=True) # Take deterministic actions when evaluation
            act = Action_adapter_pos(a, max_action)  # [0,1] to [-max,max]
            s_next, r, dw, tr, info = env.step(act)
            done = (dw or tr)

            total_scores += r
            s = s_next

    return total_scores/turns

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
