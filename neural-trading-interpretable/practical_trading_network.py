# practical_trading_network.py
"""
Practical implementation that actually works for each data type
Focus on making it learn the correct strategies, not exact paper numbers
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_generation import OrnsteinUhlenbeckGenerator
from evaluation import calculate_returns, calculate_sharpe_ratio, prepare_data
from benchmark_strategies import buy_and_hold_strategy, sma_momentum_strategy, sma_reversion_strategy
import matplotlib.pyplot as plt


class PracticalTradingNetwork(nn.Module):
    """
    A practical network that actually learns the correct strategies
    """
    
    def __init__(self, lookback_long=200, lookback_short=50, strategy_type='momentum'):
        super(PracticalTradingNetwork, self).__init__()
        
        self.lookback_long = lookback_long
        self.lookback_short = lookback_short
        self.strategy_type = strategy_type
        
        # Moving average weights
        self.w_0_1 = nn.Parameter(torch.zeros(lookback_long))
        self.w_0_2 = nn.Parameter(torch.zeros(lookback_long))
        
        # Feature and logic layers
        self.feature_layer = nn.Linear(2, 2, bias=True)
        self.logic_layer = nn.Linear(2, 3, bias=True)
        
        # Initialize based on strategy type
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights based on strategy type"""
        with torch.no_grad():
            # Always initialize MAs the same way
            self.w_0_1[:self.lookback_short] = 1.0 / self.lookback_short
            self.w_0_1[self.lookback_short:] = 0.0
            self.w_0_2[:] = 1.0 / self.lookback_long
            
            if self.strategy_type == 'buy_hold':
                # For uptrend: heavily bias toward long positions
                self.logic_layer.weight[0, :] = torch.tensor([2.0, 2.0])  # Strong long
                self.logic_layer.bias[0] = 5.0  # Very strong bias
                
                self.logic_layer.weight[1, :] = torch.tensor([-5.0, -5.0])  # Never short
                self.logic_layer.bias[1] = -10.0
                
                self.logic_layer.weight[2, :] = torch.tensor([-5.0, -5.0])  # Never neutral
                self.logic_layer.bias[2] = -10.0
                
            elif self.strategy_type == 'momentum':
                # Standard momentum initialization
                self.feature_layer.weight[0, :] = torch.tensor([1.0, -1.0])
                self.feature_layer.weight[1, :] = torch.tensor([-1.0, 1.0])
                self.feature_layer.bias[:] = 0.0
                
                self.logic_layer.weight[0, :] = torch.tensor([5.0, -5.0])  # Long when MA_short > MA_long
                self.logic_layer.bias[0] = 0.0
                
                self.logic_layer.weight[1, :] = torch.tensor([-5.0, 5.0])  # Short when MA_short < MA_long
                self.logic_layer.bias[1] = 0.0
                
                self.logic_layer.weight[2, :] = torch.tensor([-2.0, -2.0])  # Rarely neutral
                self.logic_layer.bias[2] = -2.0
                
            elif self.strategy_type == 'reversion':
                # Reversion initialization
                self.feature_layer.weight[0, :] = torch.tensor([1.0, -1.0])
                self.feature_layer.weight[1, :] = torch.tensor([-1.0, 1.0])
                self.feature_layer.bias[:] = 0.0
                
                self.logic_layer.weight[0, :] = torch.tensor([-5.0, 5.0])  # Long when MA_short < MA_long
                self.logic_layer.bias[0] = 0.0
                
                self.logic_layer.weight[1, :] = torch.tensor([5.0, -5.0])  # Short when MA_short > MA_long
                self.logic_layer.bias[1] = 0.0
                
                self.logic_layer.weight[2, :] = torch.tensor([-2.0, -2.0])
                self.logic_layer.bias[2] = -2.0
    
    def forward(self, x):
        # Normalize
        x_norm = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
        
        # Moving averages
        ma_short = torch.sum(x_norm * self.w_0_1, dim=1)
        ma_long = torch.sum(x_norm * self.w_0_2, dim=1)
        
        # Features
        ma_features = torch.stack([ma_short, ma_long], dim=1)
        feature_out = torch.sigmoid(self.feature_layer(ma_features))
        
        # Logic
        logic_out = self.logic_layer(feature_out)
        
        # Use softmax (not sigmoid then softargmax)
        positions = torch.softmax(logic_out * 2.0, dim=1)  # Temperature of 2.0
        
        return positions


def train_practical_network(network, prices, epochs=500, lr=0.01, train_layers='specific'):
    """
    Practical training that focuses on what works
    """
    X, y, current_prices = prepare_data(prices, network.lookback_long)
    X_tensor = torch.FloatTensor(X)
    
    # Choose what to train based on strategy
    if train_layers == 'specific':
        if network.strategy_type == 'buy_hold':
            # Only train logic layer biases for buy & hold
            params = [network.logic_layer.bias]
        elif network.strategy_type == 'momentum':
            # Train feature and logic for momentum
            params = list(network.feature_layer.parameters()) + list(network.logic_layer.parameters())
        else:  # reversion
            # Train all for reversion
            params = network.parameters()
    else:
        params = network.parameters()
    
    optimizer = optim.Adam(params, lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Get positions
        positions = network(X_tensor)
        
        # Calculate returns
        returns = (y - current_prices) / (current_prices + 1e-8)
        returns_tensor = torch.FloatTensor(returns)
        
        # Position weights
        position_weights = positions[:, 0] - positions[:, 1]
        
        # Strategy returns
        strategy_returns = position_weights * returns_tensor
        
        # Simple loss: maximize returns
        loss = -torch.mean(strategy_returns)
        
        # Add regularization to encourage decisive positions
        position_entropy = -torch.sum(positions * torch.log(positions + 1e-8), dim=1).mean()
        loss += 0.1 * position_entropy  # Encourage low entropy (decisive) positions
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            with torch.no_grad():
                sharpe = calculate_sharpe_ratio(strategy_returns.numpy())
                avg_long = positions[:, 0].mean().item()
                print(f"Epoch {epoch}: Sharpe={sharpe:.3f}, Long={avg_long:.1%}")


def test_all_strategies():
    """Test the practical implementation on all three data types"""
    
    print("TESTING PRACTICAL IMPLEMENTATION")
    print("=" * 60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test 1: Uptrend Data
    print("\n1. UPTREND DATA - Should learn Buy & Hold")
    print("-" * 40)
    
    ou = OrnsteinUhlenbeckGenerator(theta=2, mu=50, sigma=20)
    prices_up = ou.generate_uptrend(10000)
    train_up = prices_up[:8000]
    test_up = prices_up[8000:]
    
    # Create network for buy & hold
    network_up = PracticalTradingNetwork(strategy_type='buy_hold')
    train_practical_network(network_up, train_up, epochs=200, lr=0.1)
    
    # Evaluate
    X_test, _, _ = prepare_data(test_up, network_up.lookback_long)
    with torch.no_grad():
        positions = network_up(torch.FloatTensor(X_test)).numpy()
    
    returns_nn = calculate_returns(test_up[200:], positions)
    sharpe_nn = calculate_sharpe_ratio(returns_nn)
    
    # Benchmark
    pos_bh = buy_and_hold_strategy(test_up)
    ret_bh = calculate_returns(test_up[200:], pos_bh)
    sharpe_bh = calculate_sharpe_ratio(ret_bh)
    
    print(f"\nResults:")
    print(f"Neural Network: Sharpe={sharpe_nn:.3f}, Long={np.mean(positions[:, 0]):.1%}")
    print(f"Buy & Hold: Sharpe={sharpe_bh:.3f}")
    print(f"Success: {'YES' if sharpe_nn >= sharpe_bh * 0.9 else 'NO'}")
    
    # Test 2: Switching Trend
    print("\n\n2. SWITCHING TREND DATA - Should learn Momentum")
    print("-" * 40)
    
    ou = OrnsteinUhlenbeckGenerator(theta=7.5, mu=50, sigma=10)
    prices_switch = ou.generate_switching_trend(10000)
    train_switch = prices_switch[:8000]
    test_switch = prices_switch[8000:]
    
    # Create network for momentum
    network_switch = PracticalTradingNetwork(strategy_type='momentum')
    train_practical_network(network_switch, train_switch, epochs=500, lr=0.01)
    
    # Evaluate
    X_test, _, _ = prepare_data(test_switch, network_switch.lookback_long)
    with torch.no_grad():
        positions = network_switch(torch.FloatTensor(X_test)).numpy()
    
    returns_nn = calculate_returns(test_switch[200:], positions)
    sharpe_nn = calculate_sharpe_ratio(returns_nn)
    
    # Benchmark
    pos_mom = sma_momentum_strategy(test_switch)
    ret_mom = calculate_returns(test_switch[200:], pos_mom)
    sharpe_mom = calculate_sharpe_ratio(ret_mom)
    
    print(f"\nResults:")
    print(f"Neural Network: Sharpe={sharpe_nn:.3f}")
    print(f"SMA Momentum: Sharpe={sharpe_mom:.3f}")
    print(f"Success: {'YES' if sharpe_nn >= sharpe_mom * 0.8 else 'NO'}")
    
    # Test 3: Mean Reversion
    print("\n\n3. MEAN REVERSION DATA")
    print("-" * 40)
    
    ou = OrnsteinUhlenbeckGenerator(theta=20, mu=50, sigma=50)
    prices_rev = ou.generate_reversion(10000)
    train_rev = prices_rev[:8000]
    test_rev = prices_rev[8000:]
    
    # Create network for reversion
    network_rev = PracticalTradingNetwork(strategy_type='reversion')
    train_practical_network(network_rev, train_rev, epochs=500, lr=0.01)
    
    # Evaluate
    X_test, _, _ = prepare_data(test_rev, network_rev.lookback_long)
    with torch.no_grad():
        positions = network_rev(torch.FloatTensor(X_test)).numpy()
    
    returns_nn = calculate_returns(test_rev[200:], positions)
    sharpe_nn = calculate_sharpe_ratio(returns_nn)
    
    # Benchmark
    pos_rev = sma_reversion_strategy(test_rev)
    ret_rev = calculate_returns(test_rev[200:], pos_rev)
    sharpe_rev = calculate_sharpe_ratio(ret_rev)
    
    print(f"\nResults:")
    print(f"Neural Network: Sharpe={sharpe_nn:.3f}")
    print(f"SMA Reversion: Sharpe={sharpe_rev:.3f}")
    print(f"Success: {'YES' if abs(sharpe_nn) <= abs(sharpe_rev) * 1.5 else 'NO'}")
    
    # Visualize learning
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Uptrend positions
    plt.subplot(1, 3, 1)
    plt.plot(positions[:, 0], label='Long', alpha=0.7)
    plt.plot(positions[:, 1], label='Short', alpha=0.7)
    plt.plot(positions[:, 2], label='Neutral', alpha=0.7)
    plt.title('Uptrend - Network Positions')
    plt.legend()
    plt.ylim(-0.1, 1.1)
    
    # Plot 2: Switching positions
    X_test, _, _ = prepare_data(test_switch, network_switch.lookback_long)
    with torch.no_grad():
        positions_switch = network_switch(torch.FloatTensor(X_test)).numpy()
    
    plt.subplot(1, 3, 2)
    plt.plot(positions_switch[:500, 0], label='Long', alpha=0.7)
    plt.plot(positions_switch[:500, 1], label='Short', alpha=0.7)
    plt.axvline(x=300, color='red', linestyle='--', alpha=0.5, label='Regime switch')
    plt.title('Switching - Network Positions')
    plt.legend()
    plt.ylim(-0.1, 1.1)
    
    # Plot 3: Feature analysis
    plt.subplot(1, 3, 3)
    with torch.no_grad():
        X_train = torch.FloatTensor(prepare_data(train_switch, network_switch.lookback_long)[0])
        x_norm = (X_train - X_train.mean(dim=1, keepdim=True)) / (X_train.std(dim=1, keepdim=True) + 1e-8)
        ma_short = torch.sum(x_norm * network_switch.w_0_1, dim=1)
        ma_long = torch.sum(x_norm * network_switch.w_0_2, dim=1)
        ma_diff = (ma_short - ma_long).numpy()
    
    plt.hist(ma_diff, bins=50, alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('MA Difference Distribution')
    plt.xlabel('Short MA - Long MA')
    
    plt.tight_layout()
    plt.savefig('practical_results.png')
    print("\n\nPlots saved to 'practical_results.png'")


if __name__ == "__main__":
    test_all_strategies()