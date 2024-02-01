#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:26:13 2023

"""
import numpy as np
import torch
import similaritymeasures
import scipy.stats as stats

class Decision_Sequence_AD:
    def __init__(self, model, action_dim, state_dim, win_len_q, win_len_v, step_size, MAX_LEN=1000):
        self.model = model
        self.act_dim = action_dim
        self.state_dim = state_dim
        self.win_len_q = win_len_q
        self.win_len_v = win_len_v
        self.step_size = step_size
        self.MAX_LEN = MAX_LEN
    
    def traj2q(self, traj_s, traj_a):
        traj_q = []
        q_opt = []      
        obs = traj_s.reshape(1, -1, self.state_dim).float()
        act_tgt = traj_a.reshape(1, -1, self.act_dim).float()
        timesteps = torch.from_numpy(np.arange(len(traj_s))).reshape(1,-1)
        _, actions, _, qs = self.model(obs, act_tgt, timesteps, None)
        i = 0
        for s, a in zip(traj_s, traj_a):
            try:
                a_ind = np.where(a==1)[0][0]
            except IndexError:
                print(a)
            traj_q.append(qs[0][i][a_ind].item())
            q_opt.append(max(qs[0][i]).item())
            i += 1
        return traj_q, q_opt, qs, actions
    
    def convert_trajs2qvalues(self, trajs_s, trajs_a):
        trajs_q = []
        trajs_qopt = []
        for i in range(len(trajs_s)):      
            traj_s, traj_a = trajs_s[i].reshape(-1, self.state_dim), trajs_a[i]
            if len(traj_s) > 300:
                continue
            q,q_opt,_,_ = self.traj2q(traj_s, traj_a)
            trajs_q.append(q)
            trajs_qopt.append(q_opt)
            #print(i)
        return trajs_q, trajs_qopt

    def traj2v(self, traj_s, traj_a):
        obs = traj_s.reshape(1, -1, self.state_dim).float()  
        act_tgt = traj_a.reshape(1, -1, self.act_dim).float()
        timesteps = torch.from_numpy(np.arange(len(traj_s))).reshape(1,-1)
        _, _, vs, _ = self.model(obs, act_tgt, timesteps, None)
        return vs.detach().numpy()

    def convert_trajs2state_values(self, trajs_s, trajs_a):
        state_value = []
        for traj_s, traj_a in zip(trajs_s,trajs_a):
            if len(traj_s) > 300:
                continue
            v = self.traj2v(traj_s, traj_a)
            state_value.append(v)
        return state_value
    
    def area_between_2trajs(self, traj_opt, traj):
        area_measure = []
        qopt = np.zeros((len(traj_opt),2))
        qopt[:,0] = np.arange(len(traj_opt))
        qopt[:,1] = traj_opt
        normal_q = np.zeros((len(traj_opt),2))
        normal_q[:,0] =  np.arange(len(traj_opt))
        normal_q[:,1] = traj
        area_measure = similaritymeasures.area_between_two_curves(qopt, normal_q)     
        return area_measure

    
    def slide_window_metric_measure(self, metric, traj_1, traj_2, win_len, step_size, metric_name=''):
        traj_len = len(traj_1)
        num_section = int((traj_len - win_len)/step_size + 1)
        dist = []
        for k in range(num_section):
            w = np.array(traj_1[step_size*k:win_len + step_size*k]).reshape(-1)
            w2 = np.array(traj_2[step_size*k:win_len + step_size*k]).reshape(-1)
            if metric_name == 'correlation':
                dist.append(metric(w, w2).correlation)
            else:
                dist.append(metric(w, w2))
        return dist 
    
    def downsample_dist(self, dist):
        win_diff = abs(self.win_len_v - self.win_len_q)+1
        traj_len = len(dist)
        num_section = int((traj_len - win_diff)/self.step_size + 1)
        sampled_dist = []
        for k in range(num_section):
            w = np.array(dist[self.step_size*k:win_diff + self.step_size*k]).reshape(-1)
            sampled_dist.append(max(w))
        return sampled_dist
    
    def online_score(self, trajs_s, trajs_a):        
        q, qopt = self.convert_trajs2qvalues(trajs_s, trajs_a)
        state_values = self.convert_trajs2state_values(trajs_s, trajs_a)
        #trajs_online_score = []
        q_online_score = []
        v_online_score = []
        grads_test_lst = [-i for i in range(0, self.MAX_LEN)]
        for i in range(len(q)):
            if len(q[i]) < self.win_len_q or len(state_values[i][0]) < self.win_len_v:
                continue
            dist_q = self.slide_window_metric_measure(self.area_between_2trajs, q[i], qopt[i], self.win_len_q, self.step_size)
            dist_v = np.array(self.slide_window_metric_measure(stats.spearmanr, state_values[i][0], grads_test_lst, 
                                           self.win_len_v, self.step_size, metric_name='correlation'))
            if self.win_len_q < self.win_len_v:
                dist_q_sample = np.array(self.downsample_dist(dist_q))
                q_online_score.append(dist_q_sample)
                v_online_score.append(dist_v)
            else:
                dist_v_sample = np.array(self.downsample_dist(dist_v))
                q_online_score.append(dist_q)
                v_online_score.append(dist_v_sample)
        return q_online_score, v_online_score


    
    
    
    
    
    
    
    
    
    
    
    
    
    