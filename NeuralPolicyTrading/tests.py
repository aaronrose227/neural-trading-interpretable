import torch
import numpy as np
import math

from models import InputFeature, FeatureNet, LogicNet
from train import prepare_artificial

def test_input_feature():
    x = torch.arange(10.0).repeat(2,1)
    module = InputFeature(window_short=3, window_long=5)
    out = module(x)
    ref_short = np.mean(np.arange(10)[-3:])
    ref_long  = np.mean(np.arange(10)[-5:])
    assert np.isclose(out[0,0].item(), ref_short, atol=1e-7)
    assert np.isclose(out[0,1].item(), ref_long,  atol=1e-7)

def test_feature_net():
    net = FeatureNet(tau_init=0.1)
    m = torch.tensor([[2.0,1.0]])
    o1, o2 = net(m)[0]
    assert math.isclose(o1.item(), torch.sigmoid(1.0 + 0.1).item(), rel_tol=1e-6)
    assert math.isclose(o2.item(), torch.sigmoid(-1.0 + 0.1).item(), rel_tol=1e-6)

def test_logic_net():
    net = LogicNet()
    inputs = torch.tensor([[1,1],[1,0],[0,1],[0,0]], dtype=torch.float32)
    out = net(inputs)
    expected_idx = [1, 2, 2, 0]  # AND, NOR, NOR, NAND
    for i, idx in enumerate(expected_idx):
        assert out[i].argmax().item() == idx

def test_end_to_end_sharpe():
    data = prepare_artificial('trend', window_long=200, window_short=50)
    prices, returns, positions = data['prices'], data['returns'], data['positions']
    strat = returns * positions
    mu = strat.mean().item()
    sigma = strat.std().item()
    sharpe = (mu * 252) / (sigma * math.sqrt(252))
    assert abs(sharpe - 0.5400) < 1e-4
