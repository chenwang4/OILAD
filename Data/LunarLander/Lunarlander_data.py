#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 22:41:01 2023

@author: cww3
"""

import gym
import numpy as np

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.cluster import KMeans

#%%
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
# Download checkpoint
checkpoint = load_from_hub("araffin/ppo-LunarLander-v2", "ppo-LunarLander-v2.zip")
# Load the model
model = PPO.load(checkpoint)

env = make_vec_env("LunarLander-v2", n_envs=1)

# Evaluate
print("Evaluating model")
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=1000,
    deterministic=True,
)
print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

# Start a new episode
obs = env.reset()

try:
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
except KeyboardInterrupt:
    pass

#%%
max_episode_len = 10000
actions_rec = []
obs_rec = []
reward_rec = []
actions_per_episode = []
obs_per_episode = []
reward_per_episode = []

obs = env.reset()
while True:
    # get action
    action, _states = model.predict(obs, deterministic=True)
    actions_per_episode.append(action)
    obs_per_episode.append(obs)
    # environment step
    obs, rewards, dones, info = env.step(action)
    reward_per_episode.append(rewards)
    if dones:
        actions_rec.append(actions_per_episode)
        obs_rec.append(obs_per_episode)
        reward_rec.append(reward_per_episode)
        actions_per_episode = []
        obs_per_episode = []
        reward_per_episode = []
        obs = env.reset()
    if len(actions_rec) >= max_episode_len:
        break

np.save('new_states.npy',obs_rec)
np.save('new_actions.npy',actions_rec)
np.save('new_rewards.npy', np.array(rewards))
