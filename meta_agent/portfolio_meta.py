"""
portfolio_meta.py
Purpose: Precompute specialist positions, blend with allocator weights,
and evaluate final Sharpe for the meta-agent.
"""

from typing import Dict, List
import numpy as np
import torch

from evaluation_meta import prepare_data, calculate_returns, calculate_sharpe_ratio
from specialists_meta import load_specialist


def _positions_for_specialist(ckpt_path: str, prices: np.ndarray) -> np.ndarray:
    net = load_specialist(ckpt_path)
    X, _, _ = prepare_data(prices, net.lookback_long)
    with torch.no_grad():
        pos = net(torch.FloatTensor(X)).numpy()
    return pos  # aligned to prices[lookback_long:]


def precompute_specialist_positions(ckpts: Dict[str, str], prices: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: _positions_for_specialist(path, prices) for k, path in ckpts.items()}


def blend_positions(positions_by_kind: Dict[str, np.ndarray],
                    allocator_weights_per_t: np.ndarray,
                    order: List[str]) -> np.ndarray:
    mats = [positions_by_kind[k] for k in order]
    T = min(m.shape[0] for m in mats)
    mats = [m[:T] for m in mats]
    W = allocator_weights_per_t[:T]
    blended = np.zeros_like(mats[0])
    for i in range(len(order)):
        blended += W[:, i:i+1] * mats[i]
    return blended


def evaluate_blended(prices: np.ndarray, blended_positions: np.ndarray, lookback_long: int) -> float:
    price_subset = prices[lookback_long:]
    rets = calculate_returns(price_subset, blended_positions)
    return calculate_sharpe_ratio(rets)
