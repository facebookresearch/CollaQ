import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector

class InfluenceBasedActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.e_mode = args.e_mode

    def select_action(self, agents_inputs_alone, agents_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        if self.e_mode == "negative_sample":
            # mask actions that are excluded from selection
            masked_q_values_alone = -agents_inputs_alone.clone()
            masked_q_values_alone[avail_actions == 0.0] = -float("inf")  # should never be selected!
            # mask actions that are excluded from selection
            masked_q_values = agents_inputs.clone()
            masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!
        elif self.e_mode == "exclude_max":
            # mask actions that are excluded from selection
            masked_q_values_alone = agents_inputs_alone.clone()
            masked_q_values_alone[avail_actions == 0.0] = -float("inf")  # should never be selected!
            # mask actions that are excluded from selection
            masked_q_values = agents_inputs.clone()
            masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

            # Get rid off the top value
            masked_q_values_alone_max = th.argmax(masked_q_values_alone, dim=-1, keepdim=True)
            masked_q_values_alone_max_oh = th.zeros(masked_q_values_alone.shape).cuda()
            masked_q_values_alone_max_oh.scatter_(-1, masked_q_values_alone_max, 1)
            masked_q_values_alone[masked_q_values_alone_max_oh == 1] = -1
            masked_q_values_alone[masked_q_values_alone_max_oh == 0] = 0
            masked_q_values_alone = masked_q_values_alone * (th.sum(avail_actions, dim=-1, keepdim=True) != 1)
            masked_q_values_alone[masked_q_values_alone == -1] = -float("inf")
            masked_q_values_alone[avail_actions == 0.0] = -float("inf")


        random_numbers = th.rand_like(agents_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()
        #TODO: these numbers are fixed now
        if t_env > 1000000:
            random_numbers = th.rand_like(agents_inputs[:, :, 0])
            pick_alone = (random_numbers < self.args.e_prob).long()
            alone_actions = Categorical(logits=masked_q_values_alone.float()).sample().long()
            final_random_actions = pick_alone * alone_actions + (1 - pick_alone) * random_actions
        else:
            final_random_actions = random_actions

        picked_actions = pick_random * final_random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["influence"] = InfluenceBasedActionSelector
