"""
evaluation_meta.py
Purpose: Minimal metrics & data prep used by the meta-agent (returns, Sharpe, windows).
"""

import numpy as np


def calculate_returns(prices, positions):
    """
    prices: price series aligned to the positions start (i.e., prices[lookback:])
    positions: (T, 3) probs [long, short, neutral]
    """
    returns = (prices[1:] - prices[:-1]) / prices[:-1]
    min_len = min(len(returns), len(positions))
    returns = returns[:min_len]
    positions = positions[:min_len]
    w = positions[:, 0] - positions[:, 1]
    strat = w * returns
    return strat


def calculate_sharpe_ratio(returns, periods_per_year=252):
    if len(returns) == 0:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns)
    if sigma == 0:
        return 0.0
    return (mu * periods_per_year) / (sigma * np.sqrt(periods_per_year))


def prepare_data(prices, lookback):
    """
    Build windows for networks:
    X: (N, lookback), y: next price, current: current price for return calc
    """
    X, y = [], []
    for i in range(lookback, len(prices) - 1):
        X.append(prices[i - lookback:i])
        y.append(prices[i + 1])
    return np.array(X), np.array(y), prices[lookback:-1]
