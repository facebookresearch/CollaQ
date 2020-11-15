# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F
import torch as th


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



class SelfAttnInteractive(nn.Module):
    def __init__(self, input_shape, input_shape_alone, args):
        super(SelfAttnInteractive, self).__init__()
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
            self.e_self_attn = Multi_Head_Attention(1, self.all_feats_size, args.attn_embed_dim, args.attn_embed_dim, args.attn_embed_dim)
        elif args.attn_layers == 2:
            self.a_self_attn = Multi_Head_Attention_2layer(1, self.all_feats_size, args.attn_embed_dim, args.attn_embed_dim, args.attn_embed_dim)
            self.e_self_attn = Multi_Head_Attention_2layer(1, self.all_feats_size, args.attn_embed_dim, args.attn_embed_dim, args.attn_embed_dim)
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
            enemy_feats, self_feats_e, _ = self.e_self_attn(self_feats, enemy_feats, enemy_feats)
            enemy_self_feats = th.cat((enemy_feats.reshape(bs, -1), self_feats_e.reshape(bs, -1)), dim=-1)
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
            enemy_feats, self_feats_e, _ = self.e_self_attn(self_feats, enemy_feats, enemy_feats)
            enemy_self_feats = th.cat((enemy_feats.reshape(bs, -1), self_feats_e.reshape(bs, -1)), dim=-1)
            #Concat everything
            inputs = th.cat((world_feats, enemy_self_feats, ally_self_feats, action_id_feats), dim=-1)
        return inputs

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

class SelfAttnAlone(nn.Module):
    def __init__(self, input_shape, input_shape_alone, args):
        super(SelfAttnAlone, self).__init__()
        self.args = args
        if args.obs_agent_id:
            self.individual_feats_size = (input_shape-args.n_agents-input_shape_alone+1) // (args.n_agents - 1) - 4
        else:
            self.individual_feats_size = (input_shape-input_shape_alone) // (args.n_agents - 1) - 4
        self.all_feats_size = self.individual_feats_size + 4
        self.n_enemies = (input_shape-self.individual_feats_size-self.args.n_agents-self.args.n_actions-self.all_feats_size*(self.args.n_agents-1)) // self.all_feats_size
        self.self_relative = th.tensor([1, 0, 0, 0], device=self.args.device).float().reshape(1, 1, -1)
        if args.attn_layers == 1:
            self.e_self_attn = Multi_Head_Attention(1, self.all_feats_size, args.attn_embed_dim, args.attn_embed_dim, args.attn_embed_dim)
        elif args.attn_layers == 2:
            self.e_self_attn = Multi_Head_Attention_2layer(1, self.all_feats_size, args.attn_embed_dim, args.attn_embed_dim, args.attn_embed_dim)
        self.n_agents = args.n_agents

    def forward(self, inputs):
        bs = inputs.shape[0]
        # World features
        world_feats = inputs[:, :-self.individual_feats_size-1-self.args.n_actions-self.all_feats_size*self.n_enemies]
        action_id_feats = inputs[:, -1-self.args.n_actions:]
        self_feats = inputs[:, -1-self.args.n_actions-self.individual_feats_size:-1-self.args.n_actions].reshape(bs, 1, -1)
        self_feats = th.cat((self.self_relative.expand((bs, 1, 4)), self_feats), dim=-1)
        #Enemy features
        enemy_feats = inputs[:, -self.individual_feats_size-1-self.args.n_actions-self.all_feats_size*self.n_enemies:-self.individual_feats_size-1-self.args.n_actions].reshape(bs, self.n_enemies, -1)
        enemy_feats, self_feats_e, _ = self.e_self_attn(self_feats, enemy_feats, enemy_feats)
        enemy_self_feats = th.cat((enemy_feats.reshape(bs, -1), self_feats_e.reshape(bs, -1)), dim=-1)
        #Concat everything
        inputs = th.cat((world_feats, enemy_self_feats, action_id_feats), dim=-1)
        return inputs

class RNNRegAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNRegAgent, self).__init__()
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
        return h, q


class RNNInteractiveRegAgent(nn.Module):
    def __init__(self, input_shape, input_shape_alone, args):
        super(RNNInteractiveRegAgent, self).__init__()
        self.agent_alone = RNNRegAgent(input_shape_alone, args)
        self.agent_interactive = RNNRegAgent(input_shape, args)
        self.individual_feats_size = (input_shape-args.n_agents-input_shape_alone+1) // (args.n_agents - 1) - 4
        self.all_feats_size = self.individual_feats_size + 4
        self.args = args

    def init_hidden(self):
        # make hidden states on same device as model
        hidden_alone = self.agent_alone.init_hidden()
        hidden_interactive = self.agent_interactive.init_hidden()
        return hidden_interactive, hidden_alone, hidden_interactive

    def forward(self, inputs, inputs_alone, hidden_state, hidden_state_alone, hidden_state_):
        _inputs = inputs.clone()
        bs = _inputs.shape[0]
        # World features
        world_feats = _inputs[:, :-self.individual_feats_size-self.args.n_agents-self.args.n_actions-self.all_feats_size*(self.args.n_agents-1)]
        action_id_feats = _inputs[:, -self.args.n_agents-self.args.n_actions:]
        self_feats = _inputs[:, -self.args.n_agents-self.args.n_actions-self.individual_feats_size:-self.args.n_agents-self.args.n_actions]
        ally_feats = _inputs[:, -self.individual_feats_size-self.args.n_agents-self.args.n_actions-self.all_feats_size*(self.args.n_agents-1):-self.args.n_agents-self.args.n_actions-self.individual_feats_size].reshape(bs, self.args.n_agents-1, -1)
        _inputs = th.cat((world_feats, th.zeros(ally_feats.reshape(bs, -1).shape, device=self.args.device), self_feats, action_id_feats), dim=-1)

        h_alone, q_alone = self.agent_alone(inputs_alone, hidden_state_alone)
        h_interactive_, q_interactive_ = self.agent_interactive(_inputs, hidden_state_)
        h_interactive, q_interactive = self.agent_interactive(inputs, hidden_state)

        q = q_alone + q_interactive
        return q, h_interactive, h_alone, h_interactive_, q_interactive_

    def get_individual_q(self, inputs, inputs_alone, hidden_state, hidden_state_alone, hidden_state_):
        _inputs = inputs.clone()
        bs = _inputs.shape[0]
        # World features
        world_feats = _inputs[:, :-self.individual_feats_size-self.args.n_agents-self.args.n_actions-self.all_feats_size*(self.args.n_agents-1)]
        action_id_feats = _inputs[:, -self.args.n_agents-self.args.n_actions:]
        self_feats = _inputs[:, -self.args.n_agents-self.args.n_actions-self.individual_feats_size:-self.args.n_agents-self.args.n_actions]
        ally_feats = _inputs[:, -self.individual_feats_size-self.args.n_agents-self.args.n_actions-self.all_feats_size*(self.args.n_agents-1):-self.args.n_agents-self.args.n_actions-self.individual_feats_size].reshape(bs, self.args.n_agents-1, -1)
        _inputs = th.cat((world_feats, th.zeros(ally_feats.reshape(bs, -1).shape, device=self.args.device), self_feats, action_id_feats), dim=-1)

        h_alone, q_alone = self.agent_alone(inputs_alone, hidden_state_alone)
        h_interactive_, q_interactive_ = self.agent_interactive(_inputs, hidden_state_)
        h_interactive, q_interactive = self.agent_interactive(inputs, hidden_state)
        
        q = q_alone + q_interactive
        return q, q_interactive, q_interactive_, q_alone, h_interactive, h_alone, h_interactive_

    def get_parameters(self):
        return self.parameters()

    def update_n_agents(self, n_agents):
        pass

class RNNInteractiveAttnAgentV1(nn.Module):
    def __init__(self, input_shape, input_shape_alone, args):
        super(RNNInteractiveAttnAgentV1, self).__init__()
        if args.obs_agent_id:
            self.individual_feats_size = (input_shape-args.n_agents-input_shape_alone+1) // (args.n_agents - 1) - 4
        else:
            self.individual_feats_size = (input_shape-input_shape_alone) // (args.n_agents - 1) - 4
        self.all_feats_size = self.individual_feats_size + 4
        self.self_attn_i = SelfAttn(input_shape, input_shape_alone, args)
        self.agent_alone = RNNRegAgent(input_shape_alone, args)
        if args.attn_layers == 1:
            self.agent_interactive = RNNRegAgent(input_shape+args.attn_embed_dim*2-self.individual_feats_size-(args.n_agents-1)*self.all_feats_size, args)
        elif args.attn_layers == 2:
            self.agent_interactive = RNNRegAgent(input_shape+args.attn_embed_dim*3-self.individual_feats_size-(args.n_agents-1)*self.all_feats_size, args)
        self.args = args
        self.n_agents = args.n_agents

    def init_hidden(self):
        # make hidden states on same device as model
        hidden_alone = self.agent_alone.init_hidden()
        hidden_interactive = self.agent_interactive.init_hidden()
        return hidden_interactive, hidden_alone, hidden_interactive

    def forward(self, inputs, inputs_alone, hidden_state, hidden_state_alone, hidden_state_):
        _inputs = inputs.clone()
        if self.args.obs_agent_id:
            bs = _inputs.shape[0]
            # World features
            world_feats = _inputs[:, :-self.individual_feats_size-self.n_agents-self.args.n_actions-self.all_feats_size*(self.n_agents-1)]
            action_id_feats = _inputs[:, -self.n_agents-self.args.n_actions:]
            self_feats = _inputs[:, -self.n_agents-self.args.n_actions-self.individual_feats_size:-self.n_agents-self.args.n_actions]
            ally_feats = _inputs[:, -self.individual_feats_size-self.n_agents-self.args.n_actions-self.all_feats_size*(self.n_agents-1):-self.n_agents-self.args.n_actions-self.individual_feats_size].reshape(bs, self.n_agents-1, -1)
            _inputs = th.cat((world_feats, th.zeros(ally_feats.reshape(bs, -1).shape, device=self.args.device), self_feats, action_id_feats), dim=-1)
        else:
            bs = _inputs.shape[0]
            # World features
            world_feats = _inputs[:, :-self.individual_feats_size-self.args.n_actions-self.all_feats_size*(self.n_agents-1)]
            action_id_feats = _inputs[:, -self.args.n_actions:]
            self_feats = _inputs[:, -self.args.n_actions-self.individual_feats_size:-self.args.n_actions]
            ally_feats = _inputs[:, -self.individual_feats_size-self.args.n_actions-self.all_feats_size*(self.n_agents-1):-self.args.n_actions-self.individual_feats_size].reshape(bs, self.n_agents-1, -1)
            _inputs = th.cat((world_feats, th.zeros(ally_feats.reshape(bs, -1).shape, device=self.args.device), self_feats, action_id_feats), dim=-1)

        inputs = self.self_attn_i(inputs)
        _inputs = self.self_attn_i(_inputs)
        h_alone, q_alone = self.agent_alone(inputs_alone, hidden_state_alone)
        h_interactive_, q_interactive_ = self.agent_interactive(_inputs, hidden_state_)
        h_interactive, q_interactive = self.agent_interactive(inputs, hidden_state)

        q = q_alone + q_interactive
        return q, h_interactive, h_alone, h_interactive_, q_interactive_

    def get_individual_q(self, inputs, inputs_alone, hidden_state, hidden_state_alone, hidden_state_):
        _inputs = inputs.clone()
        if self.args.obs_agent_id:
            bs = _inputs.shape[0]
            # World features
            world_feats = _inputs[:, :-self.individual_feats_size-self.n_agents-self.args.n_actions-self.all_feats_size*(self.n_agents-1)]
            action_id_feats = _inputs[:, -self.n_agents-self.args.n_actions:]
            self_feats = _inputs[:, -self.n_agents-self.args.n_actions-self.individual_feats_size:-self.n_agents-self.args.n_actions]
            ally_feats = _inputs[:, -self.individual_feats_size-self.n_agents-self.args.n_actions-self.all_feats_size*(self.n_agents-1):-self.n_agents-self.args.n_actions-self.individual_feats_size].reshape(bs, self.n_agents-1, -1)
            _inputs = th.cat((world_feats, th.zeros(ally_feats.reshape(bs, -1).shape, device=self.args.device), self_feats, action_id_feats), dim=-1)
        else:
            bs = _inputs.shape[0]
            # World features
            world_feats = _inputs[:, :-self.individual_feats_size-self.args.n_actions-self.all_feats_size*(self.n_agents-1)]
            action_id_feats = _inputs[:, -self.args.n_actions:]
            self_feats = _inputs[:, -self.args.n_actions-self.individual_feats_size:-self.args.n_actions]
            ally_feats = _inputs[:, -self.individual_feats_size-self.args.n_actions-self.all_feats_size*(self.n_agents-1):-self.args.n_actions-self.individual_feats_size].reshape(bs, self.n_agents-1, -1)
            _inputs = th.cat((world_feats, th.zeros(ally_feats.reshape(bs, -1).shape, device=self.args.device), self_feats, action_id_feats), dim=-1)

        inputs = self.self_attn_i(inputs)
        _inputs = self.self_attn_i(_inputs)
        h_alone, q_alone = self.agent_alone(inputs_alone, hidden_state_alone)
        h_interactive_, q_interactive_ = self.agent_interactive(_inputs, hidden_state_)
        h_interactive, q_interactive = self.agent_interactive(inputs, hidden_state)
        
        q = q_alone + q_interactive
        return q, q_interactive, q_interactive_, q_alone, h_interactive, h_alone, h_interactive_

    def get_parameters(self):
        return self.parameters()

    def update_n_agents(self, n_agents):
        self.n_agents = n_agents
        self.self_attn_i.n_agents = n_agents


class RNNInteractiveAttnAgentV2(nn.Module):
    def __init__(self, input_shape, input_shape_alone, args):
        super(RNNInteractiveAttnAgentV2, self).__init__()
        if args.obs_agent_id:
            self.individual_feats_size = (input_shape-args.n_agents-input_shape_alone+1) // (args.n_agents - 1) - 4
        else:
            self.individual_feats_size = (input_shape-input_shape_alone) // (args.n_agents - 1) - 4
        self.all_feats_size = self.individual_feats_size + 4
        self.self_attn_i = SelfAttnInteractive(input_shape, input_shape_alone, args)
        if args.attn_layers == 1:
            self.agent_interactive = RNNRegAgent(input_shape+args.attn_embed_dim*4-self.individual_feats_size-(args.n_agents-1+args.n_actions - 6)*self.all_feats_size, args)
        elif args.attn_layers == 2:
            self.agent_interactive = RNNRegAgent(input_shape+args.attn_embed_dim*6-self.individual_feats_size-(args.n_agents-1+args.n_actions - 6)*self.all_feats_size, args)
        # self.self_attn_a = SelfAttnAlone(input_shape, input_shape_alone, args)
        # self.agent_alone = RNNRegAgent(input_shape_alone+args.attn_embed_dim*2-self.individual_feats_size-(args.n_actions - 6)*self.all_feats_size, args)
        self.agent_alone = RNNRegAgent(input_shape_alone, args)
        self.args = args
        self.n_agents = args.n_agents

    def init_hidden(self):
        # make hidden states on same device as model
        hidden_alone = self.agent_alone.init_hidden()
        hidden_interactive = self.agent_interactive.init_hidden()
        return hidden_interactive, hidden_alone, hidden_interactive

    def forward(self, inputs, inputs_alone, hidden_state, hidden_state_alone, hidden_state_):
        _inputs = inputs.clone()
        if self.args.obs_agent_id:
            bs = _inputs.shape[0]
            # World features
            world_feats = _inputs[:, :-self.individual_feats_size-self.n_agents-self.args.n_actions-self.all_feats_size*(self.n_agents-1)]
            action_id_feats = _inputs[:, -self.n_agents-self.args.n_actions:]
            self_feats = _inputs[:, -self.n_agents-self.args.n_actions-self.individual_feats_size:-self.n_agents-self.args.n_actions]
            ally_feats = _inputs[:, -self.individual_feats_size-self.n_agents-self.args.n_actions-self.all_feats_size*(self.n_agents-1):-self.n_agents-self.args.n_actions-self.individual_feats_size].reshape(bs, self.n_agents-1, -1)
            _inputs = th.cat((world_feats, th.zeros(ally_feats.reshape(bs, -1).shape, device=self.args.device), self_feats, action_id_feats), dim=-1)
        else:
            bs = _inputs.shape[0]
            # World features
            world_feats = _inputs[:, :-self.individual_feats_size-self.args.n_actions-self.all_feats_size*(self.n_agents-1)]
            action_id_feats = _inputs[:, -self.args.n_actions:]
            self_feats = _inputs[:, -self.args.n_actions-self.individual_feats_size:-self.args.n_actions]
            ally_feats = _inputs[:, -self.individual_feats_size-self.args.n_actions-self.all_feats_size*(self.n_agents-1):-self.args.n_actions-self.individual_feats_size].reshape(bs, self.n_agents-1, -1)
            _inputs = th.cat((world_feats, th.zeros(ally_feats.reshape(bs, -1).shape, device=self.args.device), self_feats, action_id_feats), dim=-1)

        inputs = self.self_attn_i(inputs)
        _inputs = self.self_attn_i(_inputs)
        # inputs_alone = self.self_attn_a(inputs_alone)
        h_alone, q_alone = self.agent_alone(inputs_alone, hidden_state_alone)
        h_interactive_, q_interactive_ = self.agent_interactive(_inputs, hidden_state_)
        h_interactive, q_interactive = self.agent_interactive(inputs, hidden_state)

        q = q_alone + q_interactive
        return q, h_interactive, h_alone, h_interactive_, q_interactive_

    def get_individual_q(self, inputs, inputs_alone, hidden_state, hidden_state_alone, hidden_state_):
        _inputs = inputs.clone()
        if self.args.obs_agent_id:
            bs = _inputs.shape[0]
            # World features
            world_feats = _inputs[:, :-self.individual_feats_size-self.n_agents-self.args.n_actions-self.all_feats_size*(self.n_agents-1)]
            action_id_feats = _inputs[:, -self.n_agents-self.args.n_actions:]
            self_feats = _inputs[:, -self.n_agents-self.args.n_actions-self.individual_feats_size:-self.n_agents-self.args.n_actions]
            ally_feats = _inputs[:, -self.individual_feats_size-self.n_agents-self.args.n_actions-self.all_feats_size*(self.n_agents-1):-self.n_agents-self.args.n_actions-self.individual_feats_size].reshape(bs, self.n_agents-1, -1)
            _inputs = th.cat((world_feats, th.zeros(ally_feats.reshape(bs, -1).shape, device=self.args.device), self_feats, action_id_feats), dim=-1)
        else:
            bs = _inputs.shape[0]
            # World features
            world_feats = _inputs[:, :-self.individual_feats_size-self.args.n_actions-self.all_feats_size*(self.n_agents-1)]
            action_id_feats = _inputs[:, -self.args.n_actions:]
            self_feats = _inputs[:, -self.args.n_actions-self.individual_feats_size:-self.args.n_actions]
            ally_feats = _inputs[:, -self.individual_feats_size-self.args.n_actions-self.all_feats_size*(self.n_agents-1):-self.args.n_actions-self.individual_feats_size].reshape(bs, self.n_agents-1, -1)
            _inputs = th.cat((world_feats, th.zeros(ally_feats.reshape(bs, -1).shape, device=self.args.device), self_feats, action_id_feats), dim=-1)

        inputs = self.self_attn_i(inputs)
        _inputs = self.self_attn_i(_inputs)
        # inputs_alone = self.self_attn_a(inputs_alone)
        h_alone, q_alone = self.agent_alone(inputs_alone, hidden_state_alone)
        h_interactive_, q_interactive_ = self.agent_interactive(_inputs, hidden_state_)
        h_interactive, q_interactive = self.agent_interactive(inputs, hidden_state)
        
        q = q_alone + q_interactive
        return q, q_interactive, q_interactive_, q_alone, h_interactive, h_alone, h_interactive_

    def get_parameters(self):
        return self.parameters()

    def update_n_agents(self, n_agents):
        self.n_agents = n_agents
        self.self_attn_i.n_agents = n_agents
        # self.self_attn_a.n_agents = n_agents

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
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

class RNNInteractiveAgent(nn.Module):
    def __init__(self, input_shape, input_shape_alone, args):
        super(RNNInteractiveAgent, self).__init__()
        self.agent_alone = RNNAgent(input_shape_alone, args)
        self.agent_interactive = RNNAgent(input_shape, args)

    def init_hidden(self):
        # make hidden states on same device as model
        hidden_alone = self.agent_alone.init_hidden()
        hidden_interactive = self.agent_interactive.init_hidden()
        return hidden_interactive, hidden_alone

    def forward(self, inputs, inputs_alone, hidden_state, hidden_state_alone):
        q_alone, h_alone = self.agent_alone(inputs_alone, hidden_state_alone)
        q_interactive, h_interactive = self.agent_interactive(inputs, hidden_state)
        return q_interactive+q_alone-self.agent_interactive.fc2.bias, h_interactive, h_alone

    def get_parameters(self):
        return self.parameters()

    def update_n_agents(self, n_agents):
        pass

    
class RNNInteractiveAttnAgent(nn.Module):
    def __init__(self, input_shape, input_shape_alone, args):
        super(RNNInteractiveAttnAgent, self).__init__()
        if args.obs_agent_id:
            self.individual_feats_size = (input_shape-args.n_agents-input_shape_alone+1) // (args.n_agents - 1) - 4
        else:
            self.individual_feats_size = (input_shape-input_shape_alone) // (args.n_agents - 1) - 4
        self.all_feats_size = self.individual_feats_size + 4
        self.self_attn_i = SelfAttn(input_shape, input_shape_alone, args)
        self.agent_alone = RNNAgent(input_shape_alone, args)
        if args.attn_layers == 1:
            self.agent_interactive = RNNAgent(input_shape+args.attn_embed_dim*2-self.individual_feats_size-(args.n_agents-1)*self.all_feats_size, args)
        elif args.attn_layers == 2:
            self.agent_interactive = RNNAgent(input_shape+args.attn_embed_dim*3-self.individual_feats_size-(args.n_agents-1)*self.all_feats_size, args)
        self.args = args
        self.n_agents = args.n_agents

    def init_hidden(self):
        # make hidden states on same device as model
        hidden_alone = self.agent_alone.init_hidden()
        hidden_interactive = self.agent_interactive.init_hidden()
        return hidden_interactive, hidden_alone

    def forward(self, inputs, inputs_alone, hidden_state, hidden_state_alone):
        inputs = self.self_attn_i(inputs)
        q_alone, h_alone = self.agent_alone(inputs_alone, hidden_state_alone)
        q_interactive, h_interactive = self.agent_interactive(inputs, hidden_state)
        return q_interactive+q_alone-self.agent_interactive.fc2.bias, h_interactive, h_alone

    def get_parameters(self):
        return self.parameters()

    def update_n_agents(self, n_agents):
        self.n_agents = n_agents
        self.self_attn_i.n_agents = n_agents

# #TODO: CHANGED THIS
# class RNNAgent(nn.Module):
#     def __init__(self, input_shape, args):
#         super(RNNAgent, self).__init__()
#         self.args = args

#         self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
#         self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         # self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

#     def forward(self, inputs, hidden_state):
#         x = F.relu(self.fc1(inputs))
#         h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
#         h = self.rnn(x, h_in)
#         return h

# class RNNInteractiveAgent(nn.Module):
#     def __init__(self, input_shape, input_shape_alone, args):
#         super(RNNInteractiveAgent, self).__init__()
#         self.agent_alone = RNNAgent(input_shape_alone, args)
#         self.agent_interactive = RNNAgent(input_shape, args)
#         self.fc = nn.Linear(2 * args.rnn_hidden_dim, args.n_actions, bias=False)
#         self.alone_bias = nn.Parameter(th.zeros(args.n_actions))
#         self.interactive_bias = nn.Parameter(th.zeros(args.n_actions))
#         self.args = args

#     def init_hidden(self):
#         # make hidden states on same device as model
#         hidden_alone = self.agent_alone.init_hidden()
#         hidden_interactive = self.agent_interactive.init_hidden()
#         return hidden_interactive, hidden_alone

#     def forward(self, inputs, inputs_alone, hidden_state, hidden_state_alone):
#         h_alone = self.agent_alone(inputs_alone, hidden_state_alone)
#         h_interactive = self.agent_interactive(inputs, hidden_state)
#         q = self.fc(th.cat((h_interactive, h_alone), dim=-1))+self.interactive_bias+self.alone_bias
#         return q, h_interactive, h_alone

#     def get_interactive_q(self, inputs, hidden_state):
#         h_interactive = self.agent_interactive(inputs, hidden_state)
#         q_interactive = self.fc(th.cat((th.zeros_like(h_interactive, device=self.args.device), h_interactive), dim=-1))
#         return q_interactive, h_interactive

#     def get_alone_q(self, inputs_alone, hidden_state_alone):
#         h_alone = self.agent_alone(inputs_alone, hidden_state_alone)
#         q_alone = self.fc(th.cat((h_alone, th.zeros_like(h_alone, device=self.args.device)), dim=-1))
#         return q_alone, h_alone

#     def get_individual_q(self, inputs, inputs_alone, hidden_state, hidden_state_alone):
#         h_alone = self.agent_alone(inputs_alone, hidden_state_alone)
#         h_interactive = self.agent_interactive(inputs, hidden_state)
#         q = self.fc(th.cat((h_interactive, h_alone), dim=-1))+self.interactive_bias+self.alone_bias
#         q_interactive = self.fc(th.cat((th.zeros_like(h_interactive, device=self.args.device), h_interactive), dim=-1))
#         q_alone = self.fc(th.cat((h_alone, th.zeros_like(h_alone, device=self.args.device)), dim=-1))
#         return q, q_interactive, q_alone, h_interactive, h_alone, th.zeros(1, device=self.args.device).sum()

#     def get_parameters(self):
#         return self.parameters()

#     def update_n_agents(self, n_agents):
#         pass
