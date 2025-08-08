"""
benchmark_strategies_meta.py
Purpose: Simple SMA momentum/reversion benchmarks and buy & hold.
Used only for reference comparison in the portable meta-agent experiment.
"""

import numpy as np


def sma_momentum_strategy(prices, lookback_short=50, lookback_long=200, threshold=0):
    positions = []
    for i in range(lookback_long, len(prices) - 1):
        sma_s = np.mean(prices[i - lookback_short:i])
        sma_l = np.mean(prices[i - lookback_long:i])
        if sma_s - sma_l > threshold:
            positions.append([1, 0, 0])
        elif sma_s - sma_l < -threshold:
            positions.append([0, 1, 0])
        else:
            positions.append([0, 0, 1])
    return np.array(positions)


def sma_reversion_strategy(prices, lookback_short=50, lookback_long=200, threshold=0):
    positions = []
    for i in range(lookback_long, len(prices) - 1):
        sma_s = np.mean(prices[i - lookback_short:i])
        sma_l = np.mean(prices[i - lookback_long:i])
        if sma_s - sma_l < -threshold:
            positions.append([1, 0, 0])
        elif sma_s - sma_l > threshold:
            positions.append([0, 1, 0])
        else:
            positions.append([0, 0, 1])
    return np.array(positions)


def buy_and_hold_strategy(prices, lookback_long=200):
    n = len(prices) - lookback_long - 1
    positions = np.zeros((n, 3))
    positions[:, 0] = 1
    return positions
