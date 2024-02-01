#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:44:48 2023

modified based on: 
https://github.com/kzl/decision-transformer

"""

import numpy as np
import torch
from fast_soft_sort.pytorch_ops import soft_rank
import time
from torch.distributions import Categorical

class Trainer:

    def __init__(self, model, optimizer, optimizer_v, batch_size, K, get_batch, alpha, loss_fn, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.optimizer_v = optimizer_v
        self.batch_size = batch_size
        self.K = K
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.alpha = alpha

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, monoton_train=False, print_logs=False):

        train_losses = []
        mon_train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step() #+ self.monoton_train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()
        if iter_num > 6 and monoton_train:
            for _ in range(num_steps):
                mon_train_loss = self.monoton_train_step()
                mon_train_losses.append(mon_train_loss)
                # if self.scheduler is not None:
                #     self.scheduler.step()
        
                
        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/monotonicity_loss_mean'] = np.mean(mon_train_losses)
        logs['training/monotonicity_train_loss_std'] = np.std(mon_train_losses)
        self.model.eval()
        for eval_fn in self.eval_fns:
            v_anomalous_score, q_anomalous_score = eval_fn(self.model)
            logs['evaluation/v_anomalous_score_mean'] = np.mean(v_anomalous_score)
            logs['evaluation/v_anomalous_score_std'] = np.std(v_anomalous_score)
            logs['evaluation/q_anomalous_score_mean'] = np.mean(q_anomalous_score)
            logs['evaluation/q_anomalous_score_std'] = np.std(q_anomalous_score)
        

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size, self.K)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
    def monoton_train_step(self):
        return 
    


class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, timesteps, attention_mask = self.get_batch(self.batch_size, self.K)
        action_target = torch.clone(actions)

        state_preds, pi_prob, q_opt, action_preds = self.model.forward(
            states, actions, timesteps, attention_mask=attention_mask,
        )

        act_dim = pi_prob.shape[2]
        pi_prob = pi_prob.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            pi_prob,
            action_target
        ) - self.alpha*torch.mean(Categorical(probs = pi_prob).entropy())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        # with torch.no_grad():
        #     self.diagnostics['training/action_error'] = torch.mean((pi_prob-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
    def monoton_train_step(self):#compute on the whole trajectory
        states, actions, timesteps, attention_mask = self.get_batch(self.batch_size, self.K)

        state_preds, pi_prob, v_preds, action_preds = self.model.forward(
            states, actions, timesteps, attention_mask=attention_mask,
        )
        
        loss_mo = -spearman(timesteps.float(), v_preds, regularization_strength=1e-2)
        self.optimizer_v.zero_grad()
        loss_mo.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer_v.step()
        return loss_mo.detach().cpu().item()

        
def corrcoef(target, pred):
    # np.corrcoef in torch from @mdo
    # https://forum.numer.ai/t/custom-loss-functions-for-xgboost-using-pytorch/960
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()

def spearman(
    target,
    pred,
    regularization="l2",
    regularization_strength=1.0,
):
    # fast_soft_sort uses 1-based indexing, divide by len to compute percentage of rank
    pred = soft_rank(
        pred,
        regularization=regularization,
        regularization_strength=regularization_strength,
    )
    return corrcoef(target, pred / pred.shape[-1])       
    


    
    
