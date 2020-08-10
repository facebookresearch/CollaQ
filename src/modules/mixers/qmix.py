import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        self.fc = nn.Linear(n_head * d_v, dout, bias=bias)

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


class SelfAttnV1(nn.Module):
    def __init__(self, state_dim, input_shape, input_shape_alone, args):
        super(SelfAttnV1, self).__init__()
        self.args = args
        self.n_agents = self.args.n_agents
        self.n_enemies = self.args.n_actions - 6 # 6 is the non-attack actions
        self.action_size = self.n_agents * self.args.n_actions
        self.a_individual_feats_size = (input_shape-args.n_agents-input_shape_alone+1) // (args.n_agents - 1) - 1
        self.e_individual_feats_size = (state_dim - self.action_size - self.a_individual_feats_size * self.n_agents) // self.n_enemies
        self.a_self_attn = Multi_Head_Attention(1, self.a_individual_feats_size, args.attn_embed_dim, args.attn_embed_dim, args.attn_embed_dim)
        self.e_cooldown = th.zeros((1, 1, 1), device=self.args.device).float()

    def forward(self, inputs):
        bs = inputs.shape[0]
        #Actions
        actions = inputs[:, -self.action_size:]
        #Ally feats
        ally_feats = inputs[:, :-self.action_size-self.n_enemies*self.e_individual_feats_size]
        ally_feats = ally_feats.reshape(bs, self.n_agents, -1)
        self_feats = ally_feats[:, [0], :]
        ally_feats = ally_feats[:, 1:, :]
        ally_feats, self_feats_a, _ = self.a_self_attn(self_feats, ally_feats, ally_feats)
        ally_self_feats = th.cat((ally_feats.reshape(bs, -1), self_feats_a.reshape(bs, -1)), dim=-1)
        #Enemy feats
        enemy_feats = inputs[:, -self.action_size-self.n_enemies*self.e_individual_feats_size:-self.action_size]
        enemy_feats = enemy_feats.reshape(bs, self.n_enemies, -1)
        enemy_self_feats = enemy_feats.reshape(bs, -1)
        x = th.cat((ally_self_feats, enemy_self_feats, actions), dim=-1)
        return x

class SelfAttnV2(nn.Module):
    def __init__(self, state_dim, input_shape, input_shape_alone, args):
        super(SelfAttnV2, self).__init__()
        self.args = args
        self.n_agents = self.args.n_agents
        self.n_enemies = self.args.n_actions - 6 # 6 is the non-attack actions
        self.action_size = self.n_agents * self.args.n_actions
        self.a_individual_feats_size = (input_shape-args.n_agents-input_shape_alone+1) // (args.n_agents - 1) - 1
        self.e_individual_feats_size = (state_dim - self.action_size - self.a_individual_feats_size * self.n_agents) // self.n_enemies
        self.a_self_attn = Multi_Head_Attention(1, self.a_individual_feats_size, args.attn_embed_dim, args.attn_embed_dim, args.attn_embed_dim)
        self.e_self_attn = Multi_Head_Attention(1, self.a_individual_feats_size, args.attn_embed_dim, args.attn_embed_dim, args.attn_embed_dim)
        self.e_cooldown = th.zeros((1, 1, 1), device=self.args.device).float()

    def forward(self, inputs):
        bs = inputs.shape[0]
        #Actions
        actions = inputs[:, -self.action_size:]
        #Ally feats
        ally_feats = inputs[:, :-self.action_size-self.n_enemies*self.e_individual_feats_size]
        ally_feats = ally_feats.reshape(bs, self.n_agents, -1)
        self_feats = ally_feats[:, [0], :]
        ally_feats = ally_feats[:, 1:, :]
        ally_feats, self_feats_a, _ = self.a_self_attn(self_feats, ally_feats, ally_feats)
        ally_self_feats = th.cat((ally_feats.reshape(bs, -1), self_feats_a.reshape(bs, -1)), dim=-1)
        #Enemy feats
        enemy_feats = inputs[:, -self.action_size-self.n_enemies*self.e_individual_feats_size:-self.action_size]
        enemy_feats = enemy_feats.reshape(bs, self.n_enemies, -1)
        enemy_feats = th.cat((enemy_feats[:, :, [0]], self.e_cooldown.expand(bs, self.n_enemies, 1), enemy_feats[:, :, 1:]), dim=-1)
        enemy_feats, self_feats_e, _ = self.e_self_attn(self_feats, enemy_feats, enemy_feats)
        enemy_self_feats = th.cat((enemy_feats.reshape(bs, -1), self_feats_e.reshape(bs, -1)), dim=-1)
        x = th.cat((ally_self_feats, enemy_self_feats, actions), dim=-1)
        return x

class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot


class QAttnMixerV1(nn.Module):
    def __init__(self, args, input_shape=0, input_shape_alone=0):
        super(QAttnMixerV1, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.self_attn = SelfAttnV1(self.state_dim, input_shape, input_shape_alone, args)
        self.embed_dim = args.mixing_embed_dim
        self.n_enemies = self.args.n_actions - 6 # 6 is the non-attack actions
        self.action_size = self.n_agents * self.args.n_actions
        self.a_individual_feats_size = (input_shape-args.n_agents-input_shape_alone+1) // (args.n_agents - 1) - 1
        self.e_individual_feats_size = (self.state_dim - self.action_size - self.a_individual_feats_size * self.n_agents) // self.n_enemies

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim-self.n_agents*self.a_individual_feats_size+args.attn_embed_dim*2, self.embed_dim)
            self.hyper_w_final = nn.Linear(self.state_dim-self.n_agents*self.a_individual_feats_size+args.attn_embed_dim*2, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim-self.n_agents*self.a_individual_feats_size+args.attn_embed_dim*2, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim-self.n_agents*self.a_individual_feats_size+args.attn_embed_dim*2, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim-self.n_agents*self.a_individual_feats_size+args.attn_embed_dim*2, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim-self.n_agents*self.a_individual_feats_size+args.attn_embed_dim*2, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim).unsqueeze(1)
        states = states.expand(-1, self.n_agents, self.state_dim)
        for agent_i in range(self.n_agents):
            states[:, agent_i, :-self.action_size-self.n_enemies*self.e_individual_feats_size] = th.cat((states[:, agent_i, -self.action_size-self.n_enemies*self.e_individual_feats_size-(self.n_agents-agent_i)*self.a_individual_feats_size:\
                -self.action_size-self.n_enemies*self.e_individual_feats_size-(self.n_agents-agent_i-1)*self.a_individual_feats_size], states[:, agent_i, -self.action_size-self.n_enemies*self.e_individual_feats_size-self.n_agents*self.a_individual_feats_size:\
                -self.action_size-self.n_enemies*self.e_individual_feats_size-(self.n_agents-agent_i)*self.a_individual_feats_size], states[:, agent_i, -self.action_size-self.n_enemies*self.e_individual_feats_size-(self.n_agents-agent_i-1)*self.a_individual_feats_size:\
                -self.action_size-self.n_enemies*self.e_individual_feats_size]), dim=-1)
        states = states.reshape(-1, self.state_dim)
        states = self.self_attn(states)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states)) # bs*n_agents, -1
        b1 = self.hyper_b_1(states) # bs*n_agents, -1
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, self.n_agents, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1.mean(1, keepdim=True))
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.n_agents, self.embed_dim, 1).mean(1)
        # State-dependent bias
        v = self.V(states).view(-1, self.n_agents, 1, 1).mean(1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

class QAttnMixerV2(nn.Module):
    def __init__(self, args, input_shape=0, input_shape_alone=0):
        super(QAttnMixerV2, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.self_attn = SelfAttnV2(self.state_dim, input_shape, input_shape_alone, args)
        self.embed_dim = args.mixing_embed_dim
        self.n_enemies = self.args.n_actions - 6 # 6 is the non-attack actions
        self.action_size = self.n_agents * self.args.n_actions
        self.a_individual_feats_size = (input_shape-args.n_agents-input_shape_alone+1) // (args.n_agents - 1) - 1
        self.e_individual_feats_size = (self.state_dim - self.action_size - self.a_individual_feats_size * self.n_agents) // self.n_enemies

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim-self.n_agents*self.a_individual_feats_size-self.n_enemies*self.e_individual_feats_size+args.attn_embed_dim*4, self.embed_dim)
            self.hyper_w_final = nn.Linear(self.state_dim-self.n_agents*self.a_individual_feats_size-self.n_enemies*self.e_individual_feats_size+args.attn_embed_dim*4, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim-self.n_agents*self.a_individual_feats_size-self.n_enemies*self.e_individual_feats_size+args.attn_embed_dim*4, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim-self.n_agents*self.a_individual_feats_size-self.n_enemies*self.e_individual_feats_size+args.attn_embed_dim*4, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim-self.n_agents*self.a_individual_feats_size-self.n_enemies*self.e_individual_feats_size+args.attn_embed_dim*4, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim-self.n_agents*self.a_individual_feats_size-self.n_enemies*self.e_individual_feats_size+args.attn_embed_dim*4, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim).unsqueeze(1)
        states = states.expand(-1, self.n_agents, self.state_dim)
        for agent_i in range(self.n_agents):
            states[:, agent_i, :-self.action_size-self.n_enemies*self.e_individual_feats_size] = th.cat((states[:, agent_i, -self.action_size-self.n_enemies*self.e_individual_feats_size-(self.n_agents-agent_i)*self.a_individual_feats_size:\
                -self.action_size-self.n_enemies*self.e_individual_feats_size-(self.n_agents-agent_i-1)*self.a_individual_feats_size], states[:, agent_i, -self.action_size-self.n_enemies*self.e_individual_feats_size-self.n_agents*self.a_individual_feats_size:\
                -self.action_size-self.n_enemies*self.e_individual_feats_size-(self.n_agents-agent_i)*self.a_individual_feats_size], states[:, agent_i, -self.action_size-self.n_enemies*self.e_individual_feats_size-(self.n_agents-agent_i-1)*self.a_individual_feats_size:\
                -self.action_size-self.n_enemies*self.e_individual_feats_size]), dim=-1)
        states = states.reshape(-1, self.state_dim)
        states = self.self_attn(states)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states)) # bs*n_agents, -1
        b1 = self.hyper_b_1(states) # bs*n_agents, -1
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, self.n_agents, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1.mean(1, keepdim=True))
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.n_agents, self.embed_dim, 1).mean(1)
        # State-dependent bias
        v = self.V(states).view(-1, self.n_agents, 1, 1).mean(1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
