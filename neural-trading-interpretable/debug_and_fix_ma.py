# debug_and_fix_ma.py
"""
Debug why MA difference is negative so often and fix it
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_generation import OrnsteinUhlenbeckGenerator
from evaluation import calculate_returns, calculate_sharpe_ratio, prepare_data
from benchmark_strategies import sma_momentum_strategy, sma_reversion_strategy
import matplotlib.pyplot as plt


def analyze_ma_behavior():
    """Analyze what's happening with the MAs"""
    
    print("ANALYZING MA BEHAVIOR")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate switching trend data
    ou = OrnsteinUhlenbeckGenerator(theta=7.5, mu=50, sigma=10)
    prices = ou.generate_switching_trend(10000)
    
    # Calculate MAs manually
    ma_50_list = []
    ma_200_list = []
    
    for i in range(200, len(prices)):
        window = prices[i-200:i]
        ma_50 = np.mean(window[-50:])
        ma_200 = np.mean(window)
        ma_50_list.append(ma_50)
        ma_200_list.append(ma_200)
    
    ma_50 = np.array(ma_50_list)
    ma_200 = np.array(ma_200_list)
    ma_diff = ma_50 - ma_200
    
    print(f"Manual calculation:")
    print(f"MA50 > MA200: {np.sum(ma_diff > 0)/len(ma_diff)*100:.1f}% of time")
    print(f"Average MA difference: {np.mean(ma_diff):.3f}")
    
    # Check what the benchmark does
    positions = sma_momentum_strategy(prices)
    print(f"\nBenchmark SMA Momentum:")
    print(f"Long: {np.mean(positions[:, 0])*100:.1f}% of time")
    print(f"Short: {np.mean(positions[:, 1])*100:.1f}% of time")
    
    # Plot to visualize
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(prices[:2000], label='Price', alpha=0.7)
    plt.plot(range(200, 2000), ma_50[:1800], label='MA50', alpha=0.8)
    plt.plot(range(200, 2000), ma_200[:1800], label='MA200', alpha=0.8)
    plt.axvline(x=500, color='red', linestyle='--', alpha=0.5, label='Regime switch')
    plt.axvline(x=1000, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=1500, color='red', linestyle='--', alpha=0.5)
    plt.legend()
    plt.title('Prices and Moving Averages')
    
    plt.subplot(3, 1, 2)
    plt.plot(ma_diff[:1800], label='MA50 - MA200')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=300, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=800, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=1300, color='red', linestyle='--', alpha=0.5)
    plt.title('MA Difference')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(positions[:1800, 0], label='Long', alpha=0.7)
    plt.plot(positions[:1800, 1], label='Short', alpha=0.7)
    plt.axvline(x=300, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=800, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=1300, color='red', linestyle='--', alpha=0.5)
    plt.title('Benchmark Positions')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ma_analysis.png')
    print("\nPlot saved to 'ma_analysis.png'")
    
    return prices


class RobustTradingNetwork(nn.Module):
    """
    More robust network that handles the actual data characteristics
    """
    
    def __init__(self, lookback_long=200, lookback_short=50, strategy_type='momentum', 
                 ma_threshold=0.0):
        super(RobustTradingNetwork, self).__init__()
        
        self.lookback_long = lookback_long
        self.lookback_short = lookback_short
        self.strategy_type = strategy_type
        self.ma_threshold = ma_threshold
        
        # Fixed MA weights
        self.register_buffer('w_0_1', torch.zeros(lookback_long))
        self.register_buffer('w_0_2', torch.zeros(lookback_long))
        
        # Initialize MAs
        self.w_0_1[:lookback_short] = 1.0 / lookback_short
        self.w_0_2[:] = 1.0 / lookback_long
        
        # Decision network with more capacity
        self.decision_net = nn.Sequential(
            nn.Linear(3, 10),  # Input: [ma_diff, ma_short, ma_long]
            nn.ReLU(),
            nn.Linear(10, 3)   # Output: [long, short, neutral]
        )
        
        self._initialize_strategy()
        
    def _initialize_strategy(self):
        """Initialize based on strategy"""
        with torch.no_grad():
            if self.strategy_type == 'momentum':
                # First layer: detect positive/negative MA diff
                self.decision_net[0].weight[:5, 0] = 10.0   # Positive diff detectors
                self.decision_net[0].weight[5:, 0] = -10.0  # Negative diff detectors
                
                # Output layer: map to positions
                self.decision_net[2].weight[0, :5] = 1.0    # Long from positive
                self.decision_net[2].weight[0, 5:] = -1.0   # No long from negative
                self.decision_net[2].weight[1, :5] = -1.0   # No short from positive
                self.decision_net[2].weight[1, 5:] = 1.0    # Short from negative
                
            elif self.strategy_type == 'reversion':
                # Opposite mapping
                self.decision_net[0].weight[:5, 0] = -10.0  # Negative diff detectors
                self.decision_net[0].weight[5:, 0] = 10.0   # Positive diff detectors
                
                self.decision_net[2].weight[0, :5] = 1.0    # Long from negative
                self.decision_net[2].weight[0, 5:] = -1.0   
                self.decision_net[2].weight[1, :5] = -1.0   
                self.decision_net[2].weight[1, 5:] = 1.0    # Short from positive
    
    def forward(self, x):
        # Compute MAs on raw prices
        ma_short = torch.sum(x * self.w_0_1, dim=1)
        ma_long = torch.sum(x * self.w_0_2, dim=1)
        
        # MA difference
        ma_diff = ma_short - ma_long
        
        # Create feature vector
        features = torch.stack([
            ma_diff / x.std(dim=1),  # Normalized by price scale
            ma_short / x.mean(dim=1),  # Relative MA
            ma_long / x.mean(dim=1)
        ], dim=1)
        
        # Decision
        logits = self.decision_net(features)
        positions = torch.softmax(logits * 3.0, dim=1)
        
        return positions, ma_diff


def train_robust_network(network, prices, epochs=2000, lr=0.01):
    """Train with better optimization"""
    
    X, y, current_prices = prepare_data(prices, network.lookback_long)
    X_tensor = torch.FloatTensor(X)
    
    optimizer = optim.Adam(network.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5)
    
    best_sharpe = -float('inf')
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        positions, ma_diff = network(X_tensor)
        
        # Returns
        returns = (y - current_prices) / (current_prices + 1e-8)
        returns_tensor = torch.FloatTensor(returns)
        
        # Position weights
        position_weights = positions[:, 0] - positions[:, 1]
        
        # Strategy returns
        strategy_returns = position_weights * returns_tensor
        
        # Sharpe loss
        sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8)
        loss = -sharpe
        
        # Consistency penalty
        if network.strategy_type == 'momentum':
            # Should go long when diff > threshold, short when diff < -threshold
            should_long = (ma_diff > network.ma_threshold).float()
            should_short = (ma_diff < -network.ma_threshold).float()
        else:
            # Opposite for reversion
            should_long = (ma_diff < -network.ma_threshold).float()
            should_short = (ma_diff > network.ma_threshold).float()
        
        consistency_loss = (
            torch.nn.functional.mse_loss(positions[:, 0], should_long) +
            torch.nn.functional.mse_loss(positions[:, 1], should_short)
        )
        
        total_loss = loss + 0.5 * consistency_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Monitor
        if epoch % 200 == 0:
            current_sharpe = calculate_sharpe_ratio(strategy_returns.detach().numpy())
            long_pct = positions[:, 0].mean().item()
            short_pct = positions[:, 1].mean().item()
            
            print(f"Epoch {epoch}: Sharpe={current_sharpe:.3f}, "
                  f"Long={long_pct:.1%}, Short={short_pct:.1%}, "
                  f"Loss={total_loss.item():.4f}")
            
            scheduler.step(-current_sharpe)
            
            if current_sharpe > best_sharpe:
                best_sharpe = current_sharpe


def test_robust_implementation():
    """Test the robust implementation"""
    
    print("\nTESTING ROBUST IMPLEMENTATION")
    print("=" * 60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # First analyze the data
    prices = analyze_ma_behavior()
    
    # Split data
    train_prices = prices[:8000]
    test_prices = prices[8000:]
    
    print("\n\nTraining Robust Momentum Network")
    print("-" * 40)
    
    # Create and train
    network = RobustTradingNetwork(strategy_type='momentum', ma_threshold=0.5)
    train_robust_network(network, train_prices, epochs=2000, lr=0.01)
    
    # Evaluate on test
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
    
    print(f"\nFinal Results:")
    print(f"Neural Network: Sharpe={sharpe_nn:.3f}")
    print(f"SMA Momentum: Sharpe={sharpe_mom:.3f}")
    print(f"Positions: Long={np.mean(positions[:, 0]):.1%}, Short={np.mean(positions[:, 1]):.1%}")
    
    # Decision analysis
    print(f"\nDecision Analysis:")
    pos_diff = ma_diff > 0
    neg_diff = ma_diff < 0
    print(f"When MA_diff > 0 ({np.sum(pos_diff)} samples): "
          f"Long={np.mean(positions[pos_diff, 0]):.1%}, "
          f"Short={np.mean(positions[pos_diff, 1]):.1%}")
    print(f"When MA_diff < 0 ({np.sum(neg_diff)} samples): "
          f"Long={np.mean(positions[neg_diff, 0]):.1%}, "
          f"Short={np.mean(positions[neg_diff, 1]):.1%}")
    
    print(f"\nSuccess: {'YES' if sharpe_nn >= sharpe_mom * 0.5 else 'NO'}")


if __name__ == "__main__":
    test_robust_implementation()