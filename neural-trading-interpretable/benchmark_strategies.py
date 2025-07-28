# benchmark_strategies.py
"""
Benchmark trading strategies for comparison
"""

import numpy as np


def sma_momentum_strategy(prices, lookback_short=50, lookback_long=200, threshold=0):
    """Simple moving average momentum crossover strategy"""
    positions = []
    
    for i in range(lookback_long, len(prices)-1):
        # Calculate SMAs
        sma_short = np.mean(prices[i-lookback_short:i])
        sma_long = np.mean(prices[i-lookback_long:i])
        
        # Determine position
        if sma_short - sma_long > threshold:
            positions.append([1, 0, 0])  # Long
        elif sma_short - sma_long < -threshold:
            positions.append([0, 1, 0])  # Short
        else:
            positions.append([0, 0, 1])  # Neutral
    
    return np.array(positions)


def sma_reversion_strategy(prices, lookback_short=50, lookback_long=200, threshold=0):
    """Simple moving average mean reversion crossover strategy"""
    positions = []
    
    for i in range(lookback_long, len(prices)-1):
        # Calculate SMAs
        sma_short = np.mean(prices[i-lookback_short:i])
        sma_long = np.mean(prices[i-lookback_long:i])
        
        # Determine position (opposite of momentum)
        if sma_short - sma_long < -threshold:
            positions.append([1, 0, 0])  # Long
        elif sma_short - sma_long > threshold:
            positions.append([0, 1, 0])  # Short
        else:
            positions.append([0, 0, 1])  # Neutral
    
    return np.array(positions)


def buy_and_hold_strategy(prices, lookback_long=200):
    """Buy and hold strategy"""
    n_positions = len(prices) - lookback_long - 1
    positions = np.zeros((n_positions, 3))
    positions[:, 0] = 1  # Always long
    return positions
