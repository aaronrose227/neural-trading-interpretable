"""
specialists_meta.py
Purpose: Build/train/load specialist policies (momentum short/long, reversion, neutral)
for the portable meta-agent. Specialists are trained independently and then frozen.
"""

import os
from pathlib import Path
import torch
from typing import Dict

from paper_exact_network_meta import PaperExactTradingNetwork, paper_exact_train_network
from evaluation_meta import prepare_data, calculate_returns, calculate_sharpe_ratio


SPECIALIST_KINDS = ("mom_short", "mom_long", "reversion", "neutral")


def make_specialist(kind: str, lookback_long=200, lookback_short=50, beta=1.0):
    if kind not in SPECIALIST_KINDS:
        raise ValueError(f"Unknown specialist kind: {kind}")
    net = PaperExactTradingNetwork(lookback_long=lookback_long, lookback_short=lookback_short, beta=beta)
    if kind == "reversion":
        net._initialize_paper_reversion()
    elif kind == "neutral":
        net._initialize_paper_buy_and_hold()
    else:
        net._initialize_paper_momentum()
    return net


def train_specialist(kind: str,
                     train_prices,
                     out_dir: str,
                     lookback_long=200,
                     lookback_short=50,
                     train_layers="logic",
                     epochs=800,
                     lr=0.01,
                     beta=1.0,
                     seed=42):
    torch.manual_seed(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    net = make_specialist(kind, lookback_long, lookback_short, beta)
    net = paper_exact_train_network(net, train_prices, train_layers=train_layers, epochs=epochs, lr=lr)
    ckpt_path = os.path.join(out_dir, f"{kind}_lb{lookback_short}-{lookback_long}_beta{beta}.pt")
    torch.save({
        "state_dict": net.state_dict(),
        "kind": kind,
        "lookback_long": lookback_long,
        "lookback_short": lookback_short,
        "beta": beta
    }, ckpt_path)
    return ckpt_path


def load_specialist(ckpt_path: str) -> PaperExactTradingNetwork:
    data = torch.load(ckpt_path, map_location="cpu")
    net = make_specialist(data["kind"], data["lookback_long"], data["lookback_short"], data["beta"])
    net.load_state_dict(data["state_dict"])
    net.eval()
    return net


def eval_specialist_on_prices(net: PaperExactTradingNetwork, prices):
    X, _, _ = prepare_data(prices, net.lookback_long)
    with torch.no_grad():
        pos = net(torch.FloatTensor(X)).numpy()
    price_subset = prices[net.lookback_long:]
    rets = calculate_returns(price_subset, pos)
    return calculate_sharpe_ratio(rets)


def build_default_specialists(train_prices,
                              out_dir="meta_agent_checkpoints",
                              seed=42) -> Dict[str, str]:
    specs = {
        "mom_short": dict(lookback_short=5, lookback_long=200, train_layers="logic"),
        "mom_long":  dict(lookback_short=50, lookback_long=200, train_layers="logic"),
        "reversion": dict(lookback_short=50, lookback_long=200, train_layers="logic"),
        "neutral":   dict(lookback_short=50, lookback_long=200, train_layers="logic"),
    }
    paths = {}
    for kind, cfg in specs.items():
        ck = train_specialist(kind=kind,
                              train_prices=train_prices,
                              out_dir=out_dir,
                              lookback_short=cfg["lookback_short"],
                              lookback_long=cfg["lookback_long"],
                              train_layers=cfg["train_layers"],
                              epochs=800,
                              lr=0.01,
                              beta=1.0,
                              seed=seed)
        paths[kind] = ck
    return paths
