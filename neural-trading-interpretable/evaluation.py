# evaluation.py
"""
Performance evaluation metrics and data preparation utilities.
Implements the evaluation approach described in Section 4 of the paper.
"""

import numpy as np


def calculate_returns(prices, positions):
    """
    Calculate strategy returns given prices and positions.
    Following Section 2 of the paper: r_t = (p_{t+1} - p_t) / p_t
    
    Args:
        prices: Price time series
        positions: Position array [long, short, neutral] for each time step
    
    Returns:
        Array of strategy returns
    """
    # Calculate single period returns
    returns = (prices[1:] - prices[:-1]) / prices[:-1]
    
    # Align positions and returns
    min_len = min(len(positions), len(returns))
    positions = positions[:min_len]
    returns = returns[:min_len]
    
    # Convert positions to weights: long - short (neutral contributes 0)
    position_weights = positions[:, 0] - positions[:, 1]
    
    # Calculate strategy returns
    strategy_returns = position_weights * returns
    
    return strategy_returns


def calculate_sharpe_ratio(returns, periods_per_year=252):
    """
    Calculate annualized Sharpe ratio as specified in Section 4 of the paper.
    
    Args:
        returns: Array of strategy returns
        periods_per_year: Number of trading periods per year (default: 252)
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    # Annualize as per paper formula
    sharpe = (mean_return * periods_per_year) / (std_return * np.sqrt(periods_per_year))
    
    return sharpe


def calculate_cumulative_returns(returns):
    """
    Calculate cumulative returns with overflow protection.
    
    Args:
        returns: Array of strategy returns
    
    Returns:
        Array of cumulative returns
    """
    # Clip returns to prevent numerical overflow
    clipped_returns = np.clip(returns, -0.99, 10.0)
    
    # Calculate cumulative returns
    cumulative = np.cumprod(1 + clipped_returns)
    
    return cumulative


def prepare_data(prices, lookback):
    """
    Prepare data for neural network input.
    Uses raw prices without normalization as specified in paper comments.
    
    Args:
        prices: Price time series
        lookback: Number of historical prices to include in each sample
    
    Returns:
        X: Input features (price windows)
        y: Target prices (next period price)
        current_prices: Current period prices for return calculation
    """
    X = []
    y = []
    
    for i in range(lookback, len(prices)-1):
        # Use raw price windows (no normalization)
        X.append(prices[i-lookback:i])
        y.append(prices[i+1])
    
    return np.array(X), np.array(y), prices[lookback:-1]
