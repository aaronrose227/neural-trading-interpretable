"""
paper_exact_network_meta.py
Purpose: Paper-exact interpretable trading network + training loop used by specialists.
This mirrors the clean, interpretable architecture with softargmax.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from evaluation_meta import prepare_data
import numpy as np


class PaperExactTradingNetwork(nn.Module):
    def __init__(self, lookback_long=200, lookback_short=50, beta=1.0):
        super().__init__()
        self.lookback_long = lookback_long
        self.lookback_short = lookback_short
        self.beta = beta
        self.w_0_1 = nn.Parameter(torch.zeros(lookback_long))
        self.w_0_2 = nn.Parameter(torch.zeros(lookback_long))
        self.feature_layer = nn.Linear(2, 2, bias=True)
        self.logic_layer = nn.Linear(2, 3, bias=True)
        self._initialize_paper_momentum()

    def _initialize_paper_momentum(self):
        with torch.no_grad():
            self.w_0_1.zero_()
            self.w_0_1[-self.lookback_short:] = 1.0 / self.lookback_short
            self.w_0_2.fill_(1.0 / self.lookback_long)
            self.feature_layer.weight[0, 0] = 1.0
            self.feature_layer.weight[0, 1] = -1.0
            self.feature_layer.bias[0] = 0.0
            self.feature_layer.weight[1, 0] = -1.0
            self.feature_layer.weight[1, 1] = 1.0
            self.feature_layer.bias[1] = 0.0
            self.logic_layer.weight[0, 0] = 1.0
            self.logic_layer.weight[0, 1] = -1.0
            self.logic_layer.bias[0] = 0.0
            self.logic_layer.weight[1, 0] = -1.0
            self.logic_layer.weight[1, 1] = 1.0
            self.logic_layer.bias[1] = 0.0
            self.logic_layer.weight[2, 0] = -1.0
            self.logic_layer.weight[2, 1] = -1.0
            self.logic_layer.bias[2] = 1.0

    def _initialize_paper_reversion(self):
        with torch.no_grad():
            self._initialize_paper_momentum()
            self.logic_layer.weight[0, 0] = -1.0
            self.logic_layer.weight[0, 1] = 1.0
            self.logic_layer.weight[1, 0] = 1.0
            self.logic_layer.weight[1, 1] = -1.0

    def _initialize_paper_buy_and_hold(self):
        with torch.no_grad():
            self.w_0_1.zero_()
            self.w_0_1[-self.lookback_short:] = 1.0 / self.lookback_short
            self.w_0_2.fill_(1.0 / self.lookback_long)
            self.feature_layer.weight.zero_()
            self.feature_layer.bias.zero_()
            self.logic_layer.weight[0, :] = 0.0
            self.logic_layer.bias[0] = 5.0
            self.logic_layer.weight[1, :] = 0.0
            self.logic_layer.bias[1] = -5.0
            self.logic_layer.weight[2, :] = 0.0
            self.logic_layer.bias[2] = -5.0

    def forward(self, x):
        x_norm = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
        ma_s = torch.sum(x_norm * self.w_0_1, dim=1)
        ma_l = torch.sum(x_norm * self.w_0_2, dim=1)
        feats = torch.stack([ma_s, ma_l], dim=1)
        f_out = torch.sigmoid(self.feature_layer(feats))
        l_out = torch.sigmoid(self.logic_layer(f_out))
        return torch.softmax(self.beta * l_out, dim=1)


def paper_exact_train_network(network: PaperExactTradingNetwork,
                              prices,
                              train_layers='logic',
                              epochs=800,
                              lr=0.01):
    X, y, current = prepare_data(prices, network.lookback_long)
    X_t = torch.FloatTensor(X)
    # layer control
    if train_layers == 'logic':
        params = list(network.logic_layer.parameters())
        for p in network.feature_layer.parameters():
            p.requires_grad = False
        network.w_0_1.requires_grad = False
        network.w_0_2.requires_grad = False
    elif train_layers == 'logic_feature':
        params = list(network.logic_layer.parameters()) + list(network.feature_layer.parameters())
        network.w_0_1.requires_grad = False
        network.w_0_2.requires_grad = False
    else:  # 'all'
        params = list(network.parameters())

    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pos = network(X_t)
        rets = (y - current) / current
        rets_t = torch.FloatTensor(rets)
        w = pos[:, 0] - pos[:, 1]
        strat = w * rets_t
        clip = torch.clamp(1 + strat, min=1e-8)
        loss = -torch.mean(torch.log(clip))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    return network
