# Copyright (c) Facebook, Inc. and its affiliates
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import torch.nn as nn


class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, batch):
        return th.sum(agent_qs, dim=2, keepdim=True)
