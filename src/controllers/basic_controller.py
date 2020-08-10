from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape = self._get_input_shape(scheme)
        if args.obs_agent_id:
            self.input_alone_shape = scheme["obs_alone"]["vshape"] + scheme["actions_onehot"]["vshape"][0] + 1
        else:
            self.input_alone_shape = scheme["obs_alone"]["vshape"] + scheme["actions_onehot"]["vshape"][0]
        self._build_agents(self.input_shape, self.input_alone_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

        self.test_total = 0
        self.avg_num_agents_attack = th.zeros(self.n_agents + 1)
        self.avg_ally_distance = 0

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, env=None):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        # if test_mode:
        #     self.focus_fire_rate(chosen_actions, ep_batch, t_ep)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape, input_alone_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, input_alone_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def focus_fire_rate(self, chosen_actions, batch, t):
        self.test_total += 1
        n_actions_no_attack = 6
        #Compute focus fire rate
        target_id = th.clamp(chosen_actions - n_actions_no_attack, min=-1)
        max_id = self.args.n_actions - n_actions_no_attack
        num_agents_attack = []
        for i in range(max_id):
            num_agents_attack.append(th.sum(target_id == i).item())
        #Compute average distance
        inputs = batch["obs"][:, t]
        bs = batch.batch_size
        individual_feats_size = (self.input_shape-self.n_agents-self.input_alone_shape+1) // (self.n_agents - 1) - 4
        all_feats_size = individual_feats_size + 4
        n_enemies = (self.input_shape-individual_feats_size-self.n_agents-self.args.n_actions-all_feats_size*(self.n_agents-1)) // all_feats_size
        enemy_ally_feats = inputs[:, :, -individual_feats_size-all_feats_size*(self.n_agents-1+n_enemies):-individual_feats_size]\
                            .reshape(inputs.shape[0], inputs.shape[1], self.n_agents-1+n_enemies, -1)
        #Compute enemy
        e_shootable = (enemy_ally_feats[:, :, :n_enemies, 0] > 0).long()
        e_visible = (enemy_ally_feats[:, :, :n_enemies, 1] > 0).long()
        e_distance = enemy_ally_feats[:, :, :n_enemies, 1]
        e_average_distance = th.sum(e_distance, dim=1)/(th.sum(e_visible, dim=1) + 1e-6)
        #Compute ally
        #Compute enemy
        a_visible = (enemy_ally_feats[:, :, :n_enemies, 0] > 0).long()
        a_distance = enemy_ally_feats[:, :, :n_enemies, 1] * a_visible
        a_average_distance = th.sum(a_distance, dim=1)/(th.sum(a_visible, dim=1) + 1e-6)

        for num_attack in num_agents_attack:
            self.avg_num_agents_attack[num_attack] += 1
        self.avg_ally_distance += a_average_distance.mean().item()

        th.set_printoptions(precision=2)
        print("focus fire rate: ", self.avg_num_agents_attack/self.test_total)
        print("focus fire rate mean: ", self.avg_num_agents_attack[2:].sum()/self.test_total)
        print("average distance between agents: ", "%.2f" % (self.avg_ally_distance/self.test_total))
