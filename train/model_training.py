#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:01:29 2023

modified based on: 
https://github.com/kzl/decision-transformer

"""

import numpy as np
import torch
import wandb
import torch.nn as nn
import argparse
import random
from bc_transformer import BC_transformer
from trainer import SequenceTrainer
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import similaritymeasures
import scipy.stats as stats
act_dim = 4
state_dim = 8
# load data
expert_observations = np.load('../data/new_states.npy', allow_pickle=True)
expert_actions = np.load('../data/new_actions.npy', allow_pickle=True)
trajs_actions = []
for traj in expert_actions:
    traj_action = []
    for a in traj:
        action = np.zeros(act_dim)
        action[a] = 1
        traj_action.append(action)
    trajs_actions.append(traj_action)
trajs_state = [torch.tensor(i) for i in expert_observations]
trajs_actions = [torch.tensor(i) for i in trajs_actions]

indices = np.arange(len(trajs_state))
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(trajs_state, trajs_actions, indices, test_size=0.2, random_state=42)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='gym-lunarlander')
    parser.add_argument('--dataset', type=str, default='expert')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    #parser.add_argument('--max_learning_rate', '-lr', type=float, default=1e-2)
    #parser.add_argument('--min_learning_rate', type=float, default=1e-4)
    parser.add_argument('--action_learning_rate', type=float, default=1e-3*5)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5)
    parser.add_argument('--monotonicity_lr', '-m_lr', type=float, default=1e-4)
    #parser.add_argument('--warmup_steps', type=int, default=1000)
    #parser.add_argument('--num_eval_episodes', type=int, default=200)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=300)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    parser.add_argument('--step_size_up', type=int, default=400)
    
    args = parser.parse_args()

variant=vars(args)
device = variant.get('device')
log_to_wandb = variant.get('log_to_wandb', True)
env_name, dataset = variant['env'], variant['dataset']
model_type = variant['model_type']
group_name = f'{env_name}-{dataset}'
exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'
K = variant['K']
batch_size = variant['batch_size']
#num_eval_episodes = variant['num_eval_episodes']  
num_trajectories = len(X_train)
max_ep_len = max([len(i) for i in X_train])
# used for input normalization
states = np.concatenate(X_train, axis=0)
state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
def get_batch(batch_size=250, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
        )

        #s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        s, a, timesteps, mask = [], [], [], []
        for i in range(batch_size):
            traj_s_batch = X_train[int(batch_inds[i])]
            traj_a_batch = y_train[int(batch_inds[i])]
            si = random.randint(0, traj_a_batch.shape[0] - 1)

            # get sequences from dataset
            s.append(traj_s_batch[si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj_a_batch[si:si + max_len].reshape(1, -1, act_dim))
            # r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            # if 'terminals' in traj:
            #     d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            # else:
            #     d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff

            # if sequence length is smaller than the window_size(max_len), then padding 
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            # r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            # d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            # rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        # r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        # d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        # rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, timesteps, mask

model = BC_transformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
model = model.to(device=device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=variant['action_learning_rate']
)

criterion = nn.CrossEntropyLoss()
trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            optimizer_v=torch.optim.Adam(model.parameters(), lr=variant['monotonicity_lr']),
            batch_size=batch_size,
            K=K,
            alpha=0.1,
            get_batch=get_batch,
            scheduler=None,
            loss_fn=criterion,
            eval_fns=None,
        )
wandb.login(key='d7894b5940d8e57fc27f403885dfc2337e043c9b')
wandb.init(
    name=exp_prefix,
    group=group_name,
    project='gym-transformer-LunarLander',
    config=variant
)
for iter in range(variant['max_iters']): 
    outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], monoton_train=False, iter_num=iter+1, print_logs=True)
    if log_to_wandb:
        wandb.log(outputs)



