# fix_momentum_reversion.py
"""
Fix the momentum and reversion strategies
The issue is likely with feature extraction and normalization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_generation import OrnsteinUhlenbeckGenerator
from evaluation import calculate_returns, calculate_sharpe_ratio, prepare_data
from benchmark_strategies import sma_momentum_strategy, sma_reversion_strategy
import matplotlib.pyplot as plt


class FixedTradingNetwork(nn.Module):
    """
    Fixed network that properly handles momentum and reversion
    """
    
    def __init__(self, lookback_long=200, lookback_short=50, strategy_type='momentum'):
        super(FixedTradingNetwork, self).__init__()
        
        self.lookback_long = lookback_long
        self.lookback_short = lookback_short
        self.strategy_type = strategy_type
        
        # Moving average weights - NOT trainable for momentum/reversion
        self.register_buffer('w_0_1', torch.zeros(lookback_long))
        self.register_buffer('w_0_2', torch.zeros(lookback_long))
        
        # Initialize MAs
        self.w_0_1[:lookback_short] = 1.0 / lookback_short
        self.w_0_2[:] = 1.0 / lookback_long
        
        # Single layer that maps MA difference to positions
        # Input: [ma_diff], Output: [long, short, neutral]
        self.decision_layer = nn.Linear(1, 3, bias=True)
        
        # Initialize based on strategy
        self._initialize_strategy()
        
    def _initialize_strategy(self):
        """Initialize for specific strategy"""
        with torch.no_grad():
            if self.strategy_type == 'momentum':
                # If MA_diff > 0 → Long, if MA_diff < 0 → Short
                self.decision_layer.weight[0, 0] = 10.0   # Long likes positive diff
                self.decision_layer.weight[1, 0] = -10.0  # Short likes negative diff
                self.decision_layer.weight[2, 0] = 0.0    # Neutral doesn't care
                
                self.decision_layer.bias[0] = 0.0
                self.decision_layer.bias[1] = 0.0
                self.decision_layer.bias[2] = -5.0  # Discourage neutral
                
            elif self.strategy_type == 'reversion':
                # If MA_diff > 0 → Short, if MA_diff < 0 → Long (opposite)
                self.decision_layer.weight[0, 0] = -10.0  # Long likes negative diff
                self.decision_layer.weight[1, 0] = 10.0   # Short likes positive diff
                self.decision_layer.weight[2, 0] = 0.0
                
                self.decision_layer.bias[0] = 0.0
                self.decision_layer.bias[1] = 0.0
                self.decision_layer.bias[2] = -5.0
    
    def forward(self, x):
        # Don't normalize the whole window - compute MAs on raw prices
        # This preserves the actual price relationships
        
        # Compute moving averages on RAW prices
        ma_short = torch.sum(x * self.w_0_1, dim=1)
        ma_long = torch.sum(x * self.w_0_2, dim=1)
        
        # The key signal: difference between MAs
        ma_diff = (ma_short - ma_long).unsqueeze(1)
        
        # Normalize the difference for stability
        ma_diff_norm = ma_diff / (torch.abs(ma_diff).mean() + 1e-8)
        
        # Make decision based on MA difference
        logits = self.decision_layer(ma_diff_norm)
        
        # Softmax for final positions
        positions = torch.softmax(logits * 5.0, dim=1)  # Higher temperature for decisive positions
        
        return positions, ma_diff.squeeze()  # Also return MA diff for analysis


def train_fixed_network(network, prices, epochs=1000, lr=0.1):
    """
    Train with focus on correct signal
    """
    X, y, current_prices = prepare_data(prices, network.lookback_long)
    X_tensor = torch.FloatTensor(X)
    
    # Only train the decision layer
    optimizer = optim.Adam(network.decision_layer.parameters(), lr=lr)
    
    best_sharpe = -float('inf')
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Get positions and MA difference
        positions, ma_diff = network(X_tensor)
        
        # Calculate returns
        returns = (y - current_prices) / (current_prices + 1e-8)
        returns_tensor = torch.FloatTensor(returns)
        
        # Position weights
        position_weights = positions[:, 0] - positions[:, 1]
        
        # Strategy returns
        strategy_returns = position_weights * returns_tensor
        
        # Loss: maximize Sharpe ratio
        sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8)
        loss = -sharpe
        
        # Add penalty for taking wrong positions
        if network.strategy_type == 'momentum':
            # Penalize going short when MA_diff > 0 or long when MA_diff < 0
            wrong_long = torch.relu(-ma_diff) * positions[:, 0]  # Long when diff < 0
            wrong_short = torch.relu(ma_diff) * positions[:, 1]  # Short when diff > 0
            penalty = wrong_long.mean() + wrong_short.mean()
        else:  # reversion
            # Opposite penalties
            wrong_long = torch.relu(ma_diff) * positions[:, 0]   # Long when diff > 0
            wrong_short = torch.relu(-ma_diff) * positions[:, 1]  # Short when diff < 0
            penalty = wrong_long.mean() + wrong_short.mean()
        
        loss += penalty * 0.5
        
        loss.backward()
        optimizer.step()
        
        # Monitor
        if epoch % 200 == 0:
            current_sharpe = calculate_sharpe_ratio(strategy_returns.detach().numpy())
            long_pct = positions[:, 0].mean().item()
            short_pct = positions[:, 1].mean().item()
            
            print(f"Epoch {epoch}: Sharpe={current_sharpe:.3f}, "
                  f"Long={long_pct:.1%}, Short={short_pct:.1%}")
            
            if current_sharpe > best_sharpe:
                best_sharpe = current_sharpe


def test_fixed_strategies():
    """Test the fixed implementation"""
    
    print("TESTING FIXED MOMENTUM AND REVERSION")
    print("=" * 60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test 1: Switching Trend (Momentum)
    print("\n1. SWITCHING TREND - Fixed Momentum")
    print("-" * 40)
    
    ou = OrnsteinUhlenbeckGenerator(theta=7.5, mu=50, sigma=10)
    prices = ou.generate_switching_trend(10000)
    train_prices = prices[:8000]
    test_prices = prices[8000:]
    
    # Create and train
    network = FixedTradingNetwork(strategy_type='momentum')
    train_fixed_network(network, train_prices, epochs=1000, lr=0.5)
    
    # Evaluate
    X_test, _, _ = prepare_data(test_prices, network.lookback_long)
    with torch.no_grad():
        positions, ma_diff = network(torch.FloatTensor(X_test))
        positions = positions.numpy()
        ma_diff = ma_diff.numpy()
    
    returns_nn = calculate_returns(test_prices[200:], positions)
    sharpe_nn = calculate_sharpe_ratio(returns_nn)
    
    # Benchmark
    pos_mom = sma_momentum_strategy(test_prices)
    ret_mom = calculate_returns(test_prices[200:], pos_mom)
    sharpe_mom = calculate_sharpe_ratio(ret_mom)
    
    print(f"\nResults:")
    print(f"Neural Network: Sharpe={sharpe_nn:.3f}")
    print(f"SMA Momentum: Sharpe={sharpe_mom:.3f}")
    print(f"Average positions: Long={np.mean(positions[:, 0]):.1%}, "
          f"Short={np.mean(positions[:, 1]):.1%}")
    print(f"Success: {'YES' if sharpe_nn >= sharpe_mom * 0.7 else 'NO'}")
    
    # Analyze decisions
    print(f"\nDecision Analysis:")
    print(f"When MA_diff > 0: Long={np.mean(positions[ma_diff > 0, 0]):.1%}, "
          f"Short={np.mean(positions[ma_diff > 0, 1]):.1%}")
    print(f"When MA_diff < 0: Long={np.mean(positions[ma_diff < 0, 0]):.1%}, "
          f"Short={np.mean(positions[ma_diff < 0, 1]):.1%}")
    
    # Test 2: Mean Reversion
    print("\n\n2. MEAN REVERSION - Fixed")
    print("-" * 40)
    
    ou = OrnsteinUhlenbeckGenerator(theta=20, mu=50, sigma=50)
    prices_rev = ou.generate_reversion(10000)
    train_rev = prices_rev[:8000]
    test_rev = prices_rev[8000:]
    
    # Create and train
    network_rev = FixedTradingNetwork(strategy_type='reversion')
    train_fixed_network(network_rev, train_rev, epochs=1000, lr=0.5)
    
    # Evaluate
    X_test, _, _ = prepare_data(test_rev, network_rev.lookback_long)
    with torch.no_grad():
        positions_rev, ma_diff_rev = network_rev(torch.FloatTensor(X_test))
        positions_rev = positions_rev.numpy()
        ma_diff_rev = ma_diff_rev.numpy()
    
    returns_nn_rev = calculate_returns(test_rev[200:], positions_rev)
    sharpe_nn_rev = calculate_sharpe_ratio(returns_nn_rev)
    
    # Benchmark
    pos_rev = sma_reversion_strategy(test_rev)
    ret_rev = calculate_returns(test_rev[200:], pos_rev)
    sharpe_rev = calculate_sharpe_ratio(ret_rev)
    
    print(f"\nResults:")
    print(f"Neural Network: Sharpe={sharpe_nn_rev:.3f}")
    print(f"SMA Reversion: Sharpe={sharpe_rev:.3f}")
    print(f"Success: {'YES' if abs(sharpe_nn_rev - sharpe_rev) <= abs(sharpe_rev) else 'NO'}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Momentum MA difference vs positions
    axes[0, 0].scatter(ma_diff[:500], positions[:500, 0], alpha=0.5, s=10, label='Long')
    axes[0, 0].scatter(ma_diff[:500], positions[:500, 1], alpha=0.5, s=10, label='Short')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('MA Difference')
    axes[0, 0].set_ylabel('Position Probability')
    axes[0, 0].set_title('Momentum: MA Diff vs Positions')
    axes[0, 0].legend()
    
    # Momentum positions over time
    axes[0, 1].plot(positions[:500, 0], label='Long', alpha=0.7)
    axes[0, 1].plot(positions[:500, 1], label='Short', alpha=0.7)
    axes[0, 1].set_title('Momentum: Positions Over Time')
    axes[0, 1].legend()
    
    # Reversion MA difference vs positions
    axes[1, 0].scatter(ma_diff_rev[:500], positions_rev[:500, 0], alpha=0.5, s=10, label='Long')
    axes[1, 0].scatter(ma_diff_rev[:500], positions_rev[:500, 1], alpha=0.5, s=10, label='Short')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('MA Difference')
    axes[1, 0].set_ylabel('Position Probability')
    axes[1, 0].set_title('Reversion: MA Diff vs Positions')
    axes[1, 0].legend()
    
    # Compare with benchmark
    axes[1, 1].plot(pos_mom[:500, 0], label='Momentum Benchmark', alpha=0.7)
    axes[1, 1].plot(positions[:500, 0], label='NN Momentum', alpha=0.7)
    axes[1, 1].set_title('NN vs Benchmark (Long Positions)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('fixed_strategies.png')
    print("\nPlots saved to 'fixed_strategies.png'")


if __name__ == "__main__":
    test_fixed_strategies()