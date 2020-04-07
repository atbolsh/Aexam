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


class bVAE(nn.Module):
    def __init__(self, latent, beta=1e-2):
        super(bVAE, self).__init__()
        # [500, 100], []100, 150]
        self.beta = beta
        self.fc1a = nn.Linear(2, 500)
        self.fc1b = nn.Linear(500, 100)
        self.fc21 = nn.Linear(100, latent)
        self.fc22 = nn.Linear(100, latent)
        self.fc3b = nn.Linear(latent, 100)
        self.fc3a = nn.Linear(100, 150)
        self.fc4 = nn.Linear(150, 2)

    def encode(self, x):
        h1a = F.elu(self.fc1a(x))
        h1b = F.elu(self.fc1b(h1a))
        return self.fc21(h1b), self.fc22(h1b)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = Variable(torch.randn(std.size()).cuda())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3b = F.elu(self.fc3b(z))
        h3a = F.elu(self.fc3a(h3b))
        return self.fc4(h3a)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 2))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, y, output, mu, logvar):
        # S4, Equation 1, pg 15 of supplement
        recon = (y - output)**2
        var = torch.exp(logvar)
        KL = mu**2 + var - logvar
        return torch.sum(recon + (self.beta / 2)*KL)
 




