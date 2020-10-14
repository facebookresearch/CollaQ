# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F
import torch as th

class RNNAgent(nn.Module):
    def __init__(self, input_shape, input_shape_alone, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
    
    def update_n_agents(self, n_agents):
        pass

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = th.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = th.matmul(attn, v)
        return output, attn

class Multi_Head_Attention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dout, dropout=0., bias=True):
        super(Multi_Head_Attention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=bias)
        self.fc = nn.Sequential(nn.Linear(n_head * d_v, n_head * d_v, bias=bias), nn.ReLU(), nn.Linear(n_head * d_v, dout, bias=bias))

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.layer_norm_q = nn.LayerNorm(n_head * d_k, eps=1e-6)
        self.layer_norm_k = nn.LayerNorm(n_head * d_k, eps=1e-6)
        self.layer_norm_v = nn.LayerNorm(n_head * d_v, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        residual = q

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = self.layer_norm_q(q).transpose(1, 2), self.layer_norm_k(k).transpose(1, 2), self.layer_norm_v(v).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        return q, residual, attn.squeeze()

class Multi_Head_Attention_2layer(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dout, dropout=0., bias=True):
        super(Multi_Head_Attention_2layer, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs_1 = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_ks_1 = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_vs_1 = nn.Linear(d_model, n_head * d_v, bias=bias)
        self.fc_1 = nn.Linear(n_head * d_v, dout, bias=bias)

        self.attention_1 = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.layer_norm_q_1 = nn.LayerNorm(n_head * d_k, eps=1e-6)
        self.layer_norm_k_1 = nn.LayerNorm(n_head * d_k, eps=1e-6)
        self.layer_norm_v_1 = nn.LayerNorm(n_head * d_v, eps=1e-6)

        # 2nd layer of attention
        self.w_qs_2 = nn.Linear(n_head * d_k, n_head * d_k, bias=bias)
        self.w_ks_2 = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_vs_2 = nn.Linear(d_model, n_head * d_v, bias=bias)
        self.fc_2 = nn.Linear(n_head * d_v, dout, bias=bias)

        self.attention_2 = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.layer_norm_q_2 = nn.LayerNorm(n_head * d_k, eps=1e-6)
        self.layer_norm_k_2 = nn.LayerNorm(n_head * d_k, eps=1e-6)
        self.layer_norm_v_2 = nn.LayerNorm(n_head * d_v, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        #In this layer, we perform self attention
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q_ = self.w_qs_1(q).view(sz_b, len_q, n_head, d_k)
        k_ = self.w_ks_1(k).view(sz_b, len_k, n_head, d_k)
        v_ = self.w_vs_1(v).view(sz_b, len_v, n_head, d_v)
        residual1 = q_

        # Transpose for attention dot product: b x n x lq x dv
        q_, k_, v_ = self.layer_norm_q_1(q_).transpose(1, 2), self.layer_norm_k_1(k_).transpose(1, 2), self.layer_norm_v_1(v_).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        q_, attn1 = self.attention_1(q_, k_, v_, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q_ = q_.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q_ = self.fc_1(q_)

        # In second layer we use attention
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q_ = self.w_qs_2(q_).view(sz_b, len_q, n_head, d_k)
        k_ = self.w_ks_2(k).view(sz_b, len_k, n_head, d_k)
        v_ = self.w_vs_2(v).view(sz_b, len_v, n_head, d_v)
        residual2 = q_

        # Transpose for attention dot product: b x n x lq x dv
        q_, k_, v_ = self.layer_norm_q_2(q_).transpose(1, 2), self.layer_norm_k_2(k_).transpose(1, 2), self.layer_norm_v_2(v_).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        q_, attn2 = self.attention_2(q_, k_, v_, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q_ = q_.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q_ = self.fc_2(q_)
        return q_, th.cat((residual1, residual2), dim=-1), attn2.squeeze()


class SelfAttn(nn.Module):
    def __init__(self, input_shape, input_shape_alone, args):
        super(SelfAttn, self).__init__()
        self.args = args
        if args.obs_agent_id:
            self.individual_feats_size = (input_shape-args.n_agents-input_shape_alone+1) // (args.n_agents - 1) - 4
        else:
            self.individual_feats_size = (input_shape-input_shape_alone) // (args.n_agents - 1) - 4
        self.all_feats_size = self.individual_feats_size + 4
        self.n_enemies = (input_shape-self.individual_feats_size-self.args.n_agents-self.args.n_actions-self.all_feats_size*(self.args.n_agents-1)) // self.all_feats_size
        self.self_relative = th.tensor([1, 0, 0, 0], device=self.args.device).float().reshape(1, 1, -1)
        if args.attn_layers == 1:
            self.a_self_attn = Multi_Head_Attention(1, self.all_feats_size, args.attn_embed_dim, args.attn_embed_dim, args.attn_embed_dim)
        elif args.attn_layers == 2:
            self.a_self_attn = Multi_Head_Attention_2layer(1, self.all_feats_size, args.attn_embed_dim, args.attn_embed_dim, args.attn_embed_dim)
        self.n_agents = args.n_agents

    def forward(self, inputs):
        if self.args.obs_agent_id:
            bs = inputs.shape[0]
            # World features
            world_feats = inputs[:, :-self.individual_feats_size-self.n_agents-self.args.n_actions-self.all_feats_size*(self.n_agents-1+self.n_enemies)]
            action_id_feats = inputs[:, -self.n_agents-self.args.n_actions:]
            self_feats = inputs[:, -self.n_agents-self.args.n_actions-self.individual_feats_size:-self.n_agents-self.args.n_actions].reshape(bs, 1, -1)
            self_feats = th.cat((self.self_relative.expand((bs, 1, 4)), self_feats), dim=-1)
            #Ally features
            ally_feats = inputs[:, -self.individual_feats_size-self.n_agents-self.args.n_actions-self.all_feats_size*(self.n_agents-1):-self.n_agents-self.args.n_actions-self.individual_feats_size].reshape(bs, self.n_agents-1, -1)
            ally_feats, self_feats_a, _ = self.a_self_attn(self_feats, ally_feats, ally_feats)
            ally_self_feats = th.cat((ally_feats.reshape(bs, -1), self_feats_a.reshape(bs, -1)), dim=-1)
            #Enemy features
            enemy_feats = inputs[:, -self.individual_feats_size-self.n_agents-self.args.n_actions-self.all_feats_size*(self.n_agents-1+self.n_enemies):-self.individual_feats_size-self.n_agents-self.args.n_actions-self.all_feats_size*(self.n_agents-1)].reshape(bs, self.n_enemies, -1)
            enemy_self_feats = enemy_feats.reshape(bs, -1)
            #Concat everything
            inputs = th.cat((world_feats, enemy_self_feats, ally_self_feats, action_id_feats), dim=-1)
        else:
            bs = inputs.shape[0]
            # World features
            world_feats = inputs[:, :-self.individual_feats_size-self.args.n_actions-self.all_feats_size*(self.n_agents-1+self.n_enemies)]
            action_id_feats = inputs[:, -self.args.n_actions:]
            self_feats = inputs[:, -self.args.n_actions-self.individual_feats_size:-self.args.n_actions].reshape(bs, 1, -1)
            self_feats = th.cat((self.self_relative.expand((bs, 1, 4)), self_feats), dim=-1)
            #Ally features
            ally_feats = inputs[:, -self.individual_feats_size-self.args.n_actions-self.all_feats_size*(self.n_agents-1):-self.args.n_actions-self.individual_feats_size].reshape(bs, self.n_agents-1, -1)
            ally_feats, self_feats_a, _ = self.a_self_attn(self_feats, ally_feats, ally_feats)
            ally_self_feats = th.cat((ally_feats.reshape(bs, -1), self_feats_a.reshape(bs, -1)), dim=-1)
            #Enemy features
            enemy_feats = inputs[:, -self.individual_feats_size-self.args.n_actions-self.all_feats_size*(self.n_agents-1+self.n_enemies):-self.individual_feats_size-self.args.n_actions-self.all_feats_size*(self.n_agents-1)].reshape(bs, self.n_enemies, -1)
            enemy_self_feats = enemy_feats.reshape(bs, -1)
            #Concat everything
            inputs = th.cat((world_feats, enemy_self_feats, ally_self_feats, action_id_feats), dim=-1)
        return inputs

class RNNAttnAgent(nn.Module):
    def __init__(self, input_shape, input_shape_alone, args):
        super(RNNAttnAgent, self).__init__()
        if args.obs_agent_id:
            self.individual_feats_size = (input_shape-args.n_agents-input_shape_alone+1) // (args.n_agents - 1) - 4
        else:
            self.individual_feats_size = (input_shape-input_shape_alone) // (args.n_agents - 1) - 4
        self.all_feats_size = self.individual_feats_size + 4
        self.self_attn = SelfAttn(input_shape, input_shape_alone, args)
        if args.attn_layers == 1:
            self.agent = RNNAgent(input_shape+args.attn_embed_dim*2-self.individual_feats_size-(args.n_agents-1)*self.all_feats_size, input_shape_alone, args)
        elif args.attn_layers == 2:
            self.agent = RNNAgent(input_shape+args.attn_embed_dim*3-self.individual_feats_size-(args.n_agents-1)*self.all_feats_size, input_shape_alone, args)
        self.args = args
        self.n_agents = args.n_agents

    def init_hidden(self):
        # make hidden states on same device as model
        return self.agent.init_hidden()

    def forward(self, inputs, hidden_state):
        inputs = self.self_attn(inputs)
        q, h = self.agent(inputs, hidden_state)
        return q, h

    def update_n_agents(self, n_agents):
        self.n_agents = n_agents
        self.self_attn.n_agents = n_agents
