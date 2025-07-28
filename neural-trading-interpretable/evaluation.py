#evaluation.py
"""
Performance evaluation metrics and utilities
Fixed to prevent overflow in cumulative returns calculation
"""

import numpy as np


def calculate_returns(prices, positions):
    """
    Calculate returns given prices and positions
    Following Section 2 of the paper: r_t = (p_{t+1} - p_t) / p_t
    """
    # Ensure we have the right shapes
    # positions should have one less row than prices since we can't trade on the last price
    
    # Calculate single period returns
    returns = (prices[1:] - prices[:-1]) / prices[:-1]
    
    # Make sure positions and returns align
    min_len = min(len(positions), len(returns))
    positions = positions[:min_len]
    returns = returns[:min_len]
    
    # Apply positions (long=1, short=-1, neutral=0)
    position_weights = positions[:, 0] - positions[:, 1]  # long - short
    
    # Calculate strategy returns
    strategy_returns = position_weights * returns
    
    return strategy_returns


def calculate_sharpe_ratio(returns, periods_per_year=252):
    """
    Calculate annualized Sharpe ratio as specified in Section 4
    Paper uses 252 trading days per year
    """
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    # Annualize as per paper
    sharpe = (mean_return * periods_per_year) / (std_return * np.sqrt(periods_per_year))
    
    return sharpe


def calculate_cumulative_returns(returns):
    """
    Calculate cumulative returns with overflow protection
    """
    # Clip returns to prevent overflow
    clipped_returns = np.clip(returns, -0.99, 10.0)  # Prevent -100% loss and extreme gains
    
    # Calculate cumulative returns
    cumulative = np.cumprod(1 + clipped_returns)
    
    return cumulative


def calculate_log_returns(prices, positions):
    """Calculate log returns for loss function"""
    returns = (prices[1:] - prices[:-1]) / prices[:-1]
    
    # Align positions and returns
    min_len = min(len(positions), len(returns))
    positions = positions[:min_len]
    returns = returns[:min_len]
    
    position_weights = positions[:, 0] - positions[:, 1]
    strategy_returns = position_weights * returns
    
    # Log returns with clipping as per paper's loss function
    log_returns = np.log(np.maximum(1 + strategy_returns, 1e-8))
    
    return log_returns


def prepare_data(prices, lookback):
    """
    Prepare data for neural network input
    Following the paper's data preparation approach
    """
    X = []
    y = []
    
    for i in range(lookback, len(prices)-1):
        X.append(prices[i-lookback:i])
        y.append(prices[i+1])
    
    return np.array(X), np.array(y), prices[lookback:-1]
