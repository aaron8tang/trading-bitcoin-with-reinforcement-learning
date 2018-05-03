import torch.nn.functional as F
import torch.nn as nn
import torch

import math


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(28, 32),
            nn.ReLU(),
            nn.Dropout(p=.2)
        )
        self.out = nn.Linear(32, 5)
        self._init_param()

        self.optim = torch.optim.Adam(self.parameters())

    def _init_param(self):
        # He init. for hidden layer
        nn.init.normal(self.hidden[0].weight.data,
                       mean=0., std=math.sqrt(2 / self.hidden[0].in_features))
        nn.init.constant(self.hidden[0].bias.data, 0.)

        # zeroing output layer
        nn.init.constant(self.out.weight.data, 0.)
        nn.init.constant(self.out.bias.data, 1.)

    def forward(self, x):
        x = self.hidden(x)
        a = F.softmax(self.out(x), dim=1)
        return a
