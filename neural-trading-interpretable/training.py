# training.py
"""
Training utilities for the neural trading network.
Implements the policy learning approach described in Section 4 of the paper.
"""

import torch
import torch.optim as optim
import numpy as np
from evaluation import calculate_sharpe_ratio, prepare_data


def train_network(network, prices, epochs=800, lr=0.001, train_layers='all', loss_type='returns', verbose=False):
    """
    Train the network using return-based loss function.
    
    Args:
        network: TradingNetwork instance
        prices: Price time series for training
        epochs: Number of training epochs (default: 800, as per paper)
        lr: Learning rate (default: 0.001)
        train_layers: Which layers to train - 'all', 'logic', 'logic_feature'
        loss_type: Type of loss function - 'returns' or 'sharpe'
        verbose: Whether to print training progress
    
    Returns:
        List of loss values during training
    """
    # Prepare training data
    X, y, current_prices = prepare_data(prices, network.lookback_long)
    X_tensor = torch.FloatTensor(X)
    
    # Configure which parameters to train based on experiment setup
    params_to_train = _get_trainable_parameters(network, train_layers)
    optimizer = optim.Adam(params_to_train, lr=lr)
    
    # Track training progress
    losses = []
    initial_sharpe = _calculate_initial_performance(network, X_tensor, y, current_prices)
    
    if verbose:
        print(f"    Training: {train_layers} layers, lr={lr}, epochs={epochs}")
        print(f"    Initial Sharpe: {initial_sharpe:.3f}")
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        positions = network(X_tensor)
        
        # Calculate strategy returns
        returns = (y - current_prices) / current_prices
        returns_tensor = torch.FloatTensor(returns)
        position_weights = positions[:, 0] - positions[:, 1]
        strategy_returns = position_weights * returns_tensor
        
        # Compute loss based on paper's log return formulation
        loss = _calculate_loss(strategy_returns, loss_type)
        
        # Backpropagation with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        # Progress reporting
        if verbose and epoch % 200 == 0:
            current_sharpe = calculate_sharpe_ratio(strategy_returns.detach().numpy())
            print(f"    Epoch {epoch}: Sharpe {current_sharpe:.3f}")
    
    # Final performance assessment
    if verbose:
        final_sharpe = _calculate_final_performance(network, X_tensor, y, current_prices)
        improvement = final_sharpe - initial_sharpe
        print(f"    Final Training Sharpe: {final_sharpe:.3f} ({improvement:+.3f})")
    
    return losses


def _get_trainable_parameters(network, train_layers):
    """Configure which network parameters to train based on experiment type."""
    if train_layers == 'all':
        return network.parameters()
    elif train_layers == 'logic':
        return network.logic_layer.parameters()
    elif train_layers == 'logic_feature':
        return list(network.logic_layer.parameters()) + list(network.feature_layer.parameters())
    else:
        raise ValueError(f"Unknown train_layers option: {train_layers}")


def _calculate_loss(strategy_returns, loss_type):
    """Calculate training loss based on paper's formulation."""
    if loss_type == 'returns':
        # Paper's log return loss: -∑ log(1 + r_t)
        clipped_returns = torch.clamp(1 + strategy_returns, min=1e-8)
        return -torch.mean(torch.log(clipped_returns))
    elif loss_type == 'sharpe':
        # Sharpe-based loss (negative Sharpe ratio)
        mean_return = torch.mean(strategy_returns)
        std_return = torch.std(strategy_returns)
        sharpe = (mean_return * 252) / (std_return * torch.sqrt(torch.tensor(252.0)) + 1e-8)
        return -sharpe
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def _calculate_initial_performance(network, X_tensor, y, current_prices):
    """Calculate initial network performance before training."""
    with torch.no_grad():
        positions = network(X_tensor)
        returns = (y - current_prices) / current_prices
        position_weights = positions[:, 0] - positions[:, 1]
        strategy_returns = position_weights * torch.FloatTensor(returns)
        return calculate_sharpe_ratio(strategy_returns.numpy())


def _calculate_final_performance(network, X_tensor, y, current_prices):
    """Calculate final network performance after training."""
    with torch.no_grad():
        positions = network(X_tensor)
        returns = (y - current_prices) / current_prices
        position_weights = positions[:, 0] - positions[:, 1]
        strategy_returns = position_weights * torch.FloatTensor(returns)
        return calculate_sharpe_ratio(strategy_returns.numpy())