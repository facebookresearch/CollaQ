# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# from modules.agents import REGISTRY as agent_REGISTRY
# from components.action_selectors import REGISTRY as action_REGISTRY
# import torch as th
# import copy


# # This multi-agent controller shares parameters between agents
# class BasicMACInfluence:
#     def __init__(self, scheme, groups, args):
#         self.n_agents = args.n_agents
#         self.args = args
#         input_shape = self._get_input_shape(scheme)
#         input_alone_shape = self._get_input_alone_shape(scheme)
#         self._build_agents(input_shape, input_alone_shape)
#         self.agent_output_type = args.agent_output_type

#         self.action_selector = action_REGISTRY[args.action_selector](args)

#         self.hidden_states = None
#         self.hidden_states_alone = None
#         self.target_hidden_states = None

#     def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, env=None):
#         # Only select actions for the selected batch elements in bs
#         avail_actions = ep_batch["avail_actions"][:, t_ep]
#         agent_outputs_interactive, agent_outputs = self.get_interactive_q(ep_batch, t_ep, test_mode=test_mode, env=env)
#         chosen_actions = self.action_selector.select_action(agent_outputs_interactive[bs], agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
#         return chosen_actions

#     def forward(self, ep_batch, t, test_mode=False):
#         agent_inputs = self._build_inputs(ep_batch, t)
#         agent_alone_inputs = self._build_alone_inputs(ep_batch, t)
#         avail_actions = ep_batch["avail_actions"][:, t]
#         agent_outs, self.hidden_states, self.hidden_states_alone = self.agent(agent_inputs, agent_alone_inputs, self.hidden_states, self.hidden_states_alone)

#         # Softmax the agent outputs if they're policy logits
#         if self.agent_output_type == "pi_logits":

#             if getattr(self.args, "mask_before_softmax", True):
#                 # Make the logits for unavailable actions very negative to minimise their affect on the softmax
#                 reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
#                 agent_outs[reshaped_avail_actions == 0] = -1e10

#             agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
#             if not test_mode:
#                 # Epsilon floor
#                 epsilon_action_num = agent_outs.size(-1)
#                 if getattr(self.args, "mask_before_softmax", True):
#                     # With probability epsilon, we will pick an available action uniformly
#                     epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

#                 agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
#                                + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

#                 if getattr(self.args, "mask_before_softmax", True):
#                     # Zero out the unavailable actions
#                     agent_outs[reshaped_avail_actions == 0] = 0.0

#         return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

#     def get_interactive_q(self, ep_batch, t, test_mode=False, env=None):
#         agent_inputs = self._build_inputs(ep_batch, t)
#         agent_alone_inputs = self._build_alone_inputs(ep_batch, t)
#         avail_actions = ep_batch["avail_actions"][:, t]
#         agent_outs, self.hidden_states, self.hidden_states_alone = self.agent.forward(agent_inputs, agent_alone_inputs, self.hidden_states, self.hidden_states_alone)
#         #TODO: need support from sc2
#         # _, self.target_hidden_states = self.target_agent.get_interactive_q(agent_inputs, self.target_hidden_states)
#         # agent_inputs_simulated, num_actions = self._build_simulated_states_inputs(ep_batch, t, env)
#         # agent_outs_interactive, _ = self.target_agent.get_interactive_q(agent_inputs_simulated, self.target_hidden_states.repeat_interleave((self.n_agents-1)*num_actions, 0))
#         # agent_outs_interactive = th.sum(agent_outs_interactive.reshape(self.n_agents, num_actions, -1), dim=-1)
        
        
#         agent_outs_interactive, self.target_hidden_states = self.target_agent.get_interactive_q(agent_inputs, self.target_hidden_states)

#         # Softmax the agent outputs if they're policy logits
#         if self.agent_output_type == "pi_logits":

#             if getattr(self.args, "mask_before_softmax", True):
#                 # Make the logits for unavailable actions very negative to minimise their affect on the softmax
#                 reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
#                 agent_outs_interactive[reshaped_avail_actions == 0] = -1e10
#                 agent_outs[reshaped_avail_actions == 0] = -1e10

#             agent_outs_interactive = th.nn.functional.softmax(agent_outs_interactive, dim=-1)
#             agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
#             if not test_mode:
#                 # Epsilon floor
#                 epsilon_action_num = agent_outs.size(-1)
#                 if getattr(self.args, "mask_before_softmax", True):
#                     # With probability epsilon, we will pick an available action uniformly
#                     epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

#                 agent_outs_interactive = ((1 - self.action_selector.epsilon) * agent_outs_interactive
#                                + th.ones_like(agent_outs_interactive) * self.action_selector.epsilon/epsilon_action_num)
#                 agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
#                                + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

#                 if getattr(self.args, "mask_before_softmax", True):
#                     # Zero out the unavailable actions
#                     agent_outs_interactive[reshaped_avail_actions == 0] = 0.0
#                     agent_outs[reshaped_avail_actions == 0] = 0.0

#         return agent_outs_interactive.view(ep_batch.batch_size, self.n_agents, -1), agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

#     def init_hidden(self, batch_size):
#         hidden_states, hidden_states_alone = self.agent.init_hidden()
#         self.hidden_states = hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1) # bav
#         self.hidden_states_alone = hidden_states_alone.unsqueeze(0).expand(batch_size, self.n_agents, -1)

#         target_hidden_states, _ = self.target_agent.init_hidden()
#         self.target_hidden_states = target_hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1) # bav

#     def parameters(self):
#         return self.agent.get_parameters()

#     def load_state(self, other_mac):
#         self.agent.load_state_dict(other_mac.agent.state_dict())

#     def cuda(self):
#         self.agent.cuda()
#         self.target_agent.cuda()

#     def save_models(self, path):
#         th.save(self.agent.state_dict(), "{}/agent.th".format(path))

#     def load_models(self, path):
#         self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

#     def _build_agents(self, input_shape, input_alone_shape):
#         self.agent = agent_REGISTRY[self.args.agent](input_shape, input_alone_shape, self.args)
#         # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
#         self.target_agent = copy.deepcopy(self.agent)

#     def _update_targets(self):
#         self.target_agent.load_state_dict(self.agent.state_dict())

#     def _build_inputs(self, batch, t):
#         # Assumes homogenous agents with flat observations.
#         # Other MACs might want to e.g. delegate building inputs to each agent
#         bs = batch.batch_size
#         inputs = []
#         inputs.append(batch["obs"][:, t])  # b1av
#         if self.args.obs_last_action:
#             if t == 0:
#                 inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
#             else:
#                 inputs.append(batch["actions_onehot"][:, t-1])
#         if self.args.obs_agent_id:
#             inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

#         inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
#         return inputs

#     def _build_simulated_states_inputs(self, batch, t, env=None):
#         # Assumes homogenous agents with flat observations.
#         # Other MACs might want to e.g. delegate building inputs to each agent
#         assert(batch.batch_size == 1)
#         inputs, num_actions = env.simulate_next_state(self.args.obs_last_action, self.args.obs_agent_id)
#         inputs = th.tensor(inputs, device=batch.device) # b1av
#         inputs = inputs.reshape(-1, inputs.shape[-1]).float()
#         return inputs, num_actions

#     def _build_alone_inputs(self, batch, t):
#         # Assumes homogenous agents with flat observations.
#         # Other MACs might want to e.g. delegate building inputs to each agent
#         bs = batch.batch_size
#         inputs_alone = []
#         inputs_alone.append(batch["obs_alone"][:, t])  # b1av
#         if self.args.obs_last_action:
#             if t == 0:
#                 inputs_alone.append(th.zeros_like(batch["actions_onehot"][:, t]))
#             else:
#                 inputs_alone.append(batch["actions_onehot"][:, t-1])
#         if self.args.obs_agent_id:
#             inputs_alone.append(th.eye(1, device=batch.device).expand(self.n_agents, -1).unsqueeze(0).expand(bs, -1, -1))

#         inputs_alone = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs_alone], dim=1)
#         return inputs_alone

#     def _get_input_shape(self, scheme):
#         input_shape = scheme["obs"]["vshape"]
#         if self.args.obs_last_action:
#             input_shape += scheme["actions_onehot"]["vshape"][0]
#         if self.args.obs_agent_id:
#             input_shape += self.n_agents

#         return input_shape

#     def _get_input_alone_shape(self, scheme):
#         input_alone_shape = scheme["obs_alone"]["vshape"]
#         if self.args.obs_last_action:
#             input_alone_shape += scheme["actions_onehot"]["vshape"][0]
#         if self.args.obs_agent_id:
#             input_alone_shape += 1

#         return input_alone_shape


from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import copy


# This multi-agent controller shares parameters between agents
class BasicMACInfluence:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        input_alone_shape = self._get_input_alone_shape(scheme)
        self._build_agents(input_shape, input_alone_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        self.hidden_states_alone = None
        self.target_hidden_states_alone = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, env=None):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs_alone, agent_outputs = self.get_alone_q(ep_batch, t_ep, test_mode=test_mode, env=env)
        chosen_actions = self.action_selector.select_action(agent_outputs_alone[bs], agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_alone_inputs = self._build_alone_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states, self.hidden_states_alone = self.agent(agent_inputs, agent_alone_inputs, self.hidden_states, self.hidden_states_alone)

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

    def get_individual_q(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_alone_inputs = self._build_alone_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, agent_outs_interactive, agent_outs_alone, self.hidden_states, self.hidden_states_alone = self.agent.get_individual_q(agent_inputs, agent_alone_inputs, self.hidden_states, self.hidden_states_alone)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
                agent_outs_interactive[reshaped_avail_actions == 0] = -1e10
                agent_outs_alone[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            agent_outs_interactive = th.nn.functional.softmax(agent_outs_interactive, dim=-1)
            agenagent_outs_alonet_outs = th.nn.functional.softmax(agent_outs_alone, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)
                agent_outs_interactive = ((1 - self.action_selector.epsilon) * agent_outs_interactive
                               + th.ones_like(agent_outs_interactive) * self.action_selector.epsilon/epsilon_action_num)
                agent_outs_alone = ((1 - self.action_selector.epsilon) * agent_outs_alone
                               + th.ones_like(agent_outs_alone) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0
                    agent_outs_interactive[reshaped_avail_actions == 0] = 0.0
                    agent_outs_alone[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), agent_outs_interactive.view(ep_batch.batch_size, self.n_agents, -1), agent_outs_alone.view(ep_batch.batch_size, self.n_agents, -1)

    def get_alone_q(self, ep_batch, t, test_mode=False, env=None):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_alone_inputs = self._build_alone_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states, self.hidden_states_alone = self.agent.forward(agent_inputs, agent_alone_inputs, self.hidden_states, self.hidden_states_alone)
        #TODO: need support from sc2
        # _, self.target_hidden_states = self.target_agent.get_interactive_q(agent_inputs, self.target_hidden_states)
        # agent_inputs_simulated, num_actions = self._build_simulated_states_inputs(ep_batch, t, env)
        # agent_outs_interactive, _ = self.target_agent.get_interactive_q(agent_inputs_simulated, self.target_hidden_states.repeat_interleave((self.n_agents-1)*num_actions, 0))
        # agent_outs_interactive = th.sum(agent_outs_interactive.reshape(self.n_agents, num_actions, -1), dim=-1)
        
        
        agent_outs_alone, self.target_hidden_states_alone = self.target_agent.get_alone_q(agent_alone_inputs, self.target_hidden_states_alone)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs_alone[reshaped_avail_actions == 0] = -1e10
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs_alone = th.nn.functional.softmax(agent_outs_alone, dim=-1)
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs_alone = ((1 - self.action_selector.epsilon) * agent_outs_alone
                               + th.ones_like(agent_outs_alone) * self.action_selector.epsilon/epsilon_action_num)
                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs_alone[reshaped_avail_actions == 0] = 0.0
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs_alone.view(ep_batch.batch_size, self.n_agents, -1), agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        hidden_states, hidden_states_alone = self.agent.init_hidden()
        self.hidden_states = hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1) # bav
        self.hidden_states_alone = hidden_states_alone.unsqueeze(0).expand(batch_size, self.n_agents, -1)

        _, target_hidden_states_alone = self.target_agent.init_hidden()
        self.target_hidden_states_alone = target_hidden_states_alone.unsqueeze(0).expand(batch_size, self.n_agents, -1) # bav

    def parameters(self):
        return self.agent.get_parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.target_agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape, input_alone_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, input_alone_shape, self.args)
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_agent = copy.deepcopy(self.agent)

    def _update_targets(self):
        self.target_agent.load_state_dict(self.agent.state_dict())

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

    def _build_simulated_states_inputs(self, batch, t, env=None):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        assert(batch.batch_size == 1)
        inputs, num_actions = env.simulate_next_state(self.args.obs_last_action, self.args.obs_agent_id)
        inputs = th.tensor(inputs, device=batch.device) # b1av
        inputs = inputs.reshape(-1, inputs.shape[-1]).float()
        return inputs, num_actions

    def _build_alone_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs_alone = []
        inputs_alone.append(batch["obs_alone"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs_alone.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs_alone.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs_alone.append(th.eye(1, device=batch.device).expand(self.n_agents, -1).unsqueeze(0).expand(bs, -1, -1))

        inputs_alone = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs_alone], dim=1)
        return inputs_alone

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def _get_input_alone_shape(self, scheme):
        input_alone_shape = scheme["obs_alone"]["vshape"]
        if self.args.obs_last_action:
            input_alone_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_alone_shape += 1

        return input_alone_shape
