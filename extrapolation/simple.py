import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from copy import deepcopy
import math
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


class simpleELU(nn.Module):
    def __init__(self, hidden=8):
        super(simpleELU, self).__init__()
        # [500, 100], []100, 150]

        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        return self.fc2(h)


class simpleReLU(nn.Module):
    def __init__(self, hidden=8):
        super(simpleReLU, self).__init__()
        # [500, 100], []100, 150]

        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)


