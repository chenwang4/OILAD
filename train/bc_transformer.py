#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adjust decision transfomer as BC model
modified based on: 
https://github.com/kzl/decision-transformer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from trajectory_gpt2 import GPT2Model

class TrajectoryModel(nn.Module):
    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        
    def forward(self, states, actions, masks=None, attention_mask=None):
        return None, None
    
    def get_action(self, states, actions, **kwargs):
        return torch.zeros_like(actions[-1])
    
class BC_transformer(TrajectoryModel):
    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.hidden_size = hidden_size
        config = GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT2Model(config)
        
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        #self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim) 
        # self.predict_action = nn.Sequential(
        #     *([nn.Linear(hidden_size, self.act_dim)] + ([nn.ReLU()] if action_tanh else []))
        #     )
        self.softmax = nn.Softmax(dim=2)
        self.linear = nn.Linear(hidden_size, 64)
        self.fc = nn.Linear(64, self.act_dim) 
        
    def forward(self, states, actions, timesteps, attention_mask=None):
        
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)
        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs,
                                             attention_mask=stacked_attention_mask)
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)
        # get predictions
        state_preds = self.predict_state(x[:,1])    # predict next state given state and action
        # action_preds = self.predict_action(x[:,0])  # predict next action given state
        out = F.relu(self.linear(x[:,0]))
        out = self.fc(out) # here can be seen as q(s,a)
        pi_prob = self.softmax(out) #pi(s,a)
        v_preds = (pi_prob * out).sum(dim=2) #state-values
        return state_preds, pi_prob, v_preds, out
        #return state_preds, action_preds
    
    # def get_action(self, states, actions, timesteps, **kwargs):
    #     # we don't care about the past rewards in this model
    #     states = states.reshape(1, -1, self.state_dim)
    #     actions = actions.reshape(1, -1, self.act_dim)
    #     timesteps = timesteps.reshape(1, -1)

    #     if self.max_length is not None:
    #         states = states[:,-self.max_length:]
    #         actions = actions[:,-self.max_length:]
    #         timesteps = timesteps[:,-self.max_length:]

    #         # pad all tokens to sequence length
    #         attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
    #         attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
    #         states = torch.cat(
    #             [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
    #             dim=1).to(dtype=torch.float32)
    #         actions = torch.cat(
    #             [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
    #                          device=actions.device), actions],
    #             dim=1).to(dtype=torch.float32)
    #         timesteps = torch.cat(
    #             [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
    #             dim=1
    #         ).to(dtype=torch.long)
    #     else:
    #         attention_mask = None

    #     _, action_preds = self.forward(
    #         states, actions, timesteps, attention_mask=attention_mask, **kwargs)
    #     return action_preds[0,-1]

        

            
        
        
        
        
        
        
        
        
        
        
        
        
        
        