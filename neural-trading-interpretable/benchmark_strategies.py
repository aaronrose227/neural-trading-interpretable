# benchmark_strategies.py
"""
Benchmark trading strategies for comparison with neural network approach.
Implements simple moving average momentum and reversion strategies as described in the paper.
"""

import numpy as np


def sma_momentum_strategy(prices, lookback_short=50, lookback_long=200, threshold=0):
    """
    Simple moving average momentum crossover strategy.
    Goes long when short MA > long MA, short otherwise.
    
    Args:
        prices: Price time series
        lookback_short: Short-term moving average window (default: 50)
        lookback_long: Long-term moving average window (default: 200)
        threshold: Crossover threshold (default: 0)
    
    Returns:
        Array of positions [long, short, neutral] for each time step
    """
    positions = []
    
    for i in range(lookback_long, len(prices)-1):
        # Calculate simple moving averages
        sma_short = np.mean(prices[i-lookback_short:i])
        sma_long = np.mean(prices[i-lookback_long:i])
        
        # Momentum logic: follow the trend
        if sma_short - sma_long > threshold:
            positions.append([1, 0, 0])  # Long position
        elif sma_short - sma_long < -threshold:
            positions.append([0, 1, 0])  # Short position
        else:
            positions.append([0, 0, 1])  # Neutral position
    
    return np.array(positions)


def sma_reversion_strategy(prices, lookback_short=50, lookback_long=200, threshold=0):
    """
    Simple moving average mean reversion strategy.
    Goes long when short MA < long MA (opposite of momentum).
    
    Args:
        prices: Price time series
        lookback_short: Short-term moving average window (default: 50)
        lookback_long: Long-term moving average window (default: 200)
        threshold: Crossover threshold (default: 0)
    
    Returns:
        Array of positions [long, short, neutral] for each time step
    """
    positions = []
    
    for i in range(lookback_long, len(prices)-1):
        # Calculate simple moving averages
        sma_short = np.mean(prices[i-lookback_short:i])
        sma_long = np.mean(prices[i-lookback_long:i])
        
        # Reversion logic: trade against the trend
        if sma_short - sma_long < -threshold:
            positions.append([1, 0, 0])  # Long position (reversion)
        elif sma_short - sma_long > threshold:
            positions.append([0, 1, 0])  # Short position (reversion)
        else:
            positions.append([0, 0, 1])  # Neutral position
    
    return np.array(positions)


def buy_and_hold_strategy(prices, lookback_long=200):
    """
    Buy and hold strategy - always maintains a long position.
    
    Args:
        prices: Price time series
        lookback_long: Lookback period to match other strategies (default: 200)
    s
    Returns:
        Array of positions [long, short, neutral] - always [1, 0, 0]
    """
    n_positions = len(prices) - lookback_long - 1
    positions = np.zeros((n_positions, 3))
    positions[:, 0] = 1  # Always long
    return positions