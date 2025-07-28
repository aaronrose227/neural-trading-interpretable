#training.py

"""
Training utilities for the neural trading network
"""

import torch
import torch.optim as optim
import numpy as np
from evaluation import calculate_sharpe_ratio, prepare_data


def train_network(network, prices, epochs=800, lr=0.001, train_layers='all', loss_type='returns'):
    """
    Train the network using return-based or Sharpe-based loss
    
    Args:
        network: TradingNetwork instance
        prices: Price time series
        epochs: Number of training epochs
        lr: Learning rate
        train_layers: 'all', 'logic', 'logic_feature'
        loss_type: 'returns' or 'sharpe'
    """
    # Prepare data
    X, y, current_prices = prepare_data(prices, network.lookback_long)
    X_tensor = torch.FloatTensor(X)
    
    # Set which parameters to train
    params_to_train = []
    
    if train_layers == 'all':
        params_to_train = network.parameters()
    elif train_layers == 'logic':
        params_to_train = network.logic_layer.parameters()
    elif train_layers == 'logic_feature':
        params_to_train = list(network.logic_layer.parameters()) + \
                         list(network.feature_layer.parameters())
    
    optimizer = optim.Adam(params_to_train, lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        positions = network(X_tensor)
        
        # Calculate returns
        returns = (y - current_prices) / current_prices
        returns_tensor = torch.FloatTensor(returns)
        
        # Position weights: long - short
        position_weights = positions[:, 0] - positions[:, 1]
        
        # Strategy returns
        strategy_returns = position_weights * returns_tensor
        
        if loss_type == 'returns':
            # Return-based loss with clipping to prevent NaN
            # Clip strategy returns to prevent log(negative number)
            clipped_returns = torch.clamp(1 + strategy_returns, min=1e-8)
            loss = -torch.mean(torch.log(clipped_returns))
        elif loss_type == 'sharpe':
            # Sharpe-based loss
            mean_return = torch.mean(strategy_returns)
            std_return = torch.std(strategy_returns)
            sharpe = (mean_return * 252) / (std_return * torch.sqrt(torch.tensor(252.0)) + 1e-8)
            loss = -sharpe
        
        # Backpropagation
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            sharpe = calculate_sharpe_ratio(strategy_returns.detach().numpy())
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Sharpe: {sharpe:.2f}")
    
    return losses
