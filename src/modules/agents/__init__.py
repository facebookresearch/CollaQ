# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

REGISTRY = {}

from .rnn_agent import RNNAgent, RNNAttnAgent
from .rnn_interactive_agent import RNNInteractiveAgent, RNNInteractiveAttnAgentV1, RNNInteractiveAttnAgentV2, RNNInteractiveRegAgent, RNNInteractiveAttnAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_attn"] = RNNAttnAgent
REGISTRY["rnn_interactive"] = RNNInteractiveAgent
REGISTRY["rnn_interactive_reg"] = RNNInteractiveRegAgent
REGISTRY["rnn_interactive_attnv1"] = RNNInteractiveAttnAgentV1
REGISTRY["rnn_interactive_attnv2"] = RNNInteractiveAttnAgentV2
REGISTRY["rnn_interactive_attn"] = RNNInteractiveAttnAgent
