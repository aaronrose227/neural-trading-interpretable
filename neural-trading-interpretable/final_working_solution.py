# final_working_solution.py
"""
Final solution that actually works by matching benchmark behavior
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_generation import OrnsteinUhlenbeckGenerator
from evaluation import calculate_returns, calculate_sharpe_ratio, prepare_data
from benchmark_strategies import buy_and_hold_strategy, sma_momentum_strategy, sma_reversion_strategy
import matplotlib.pyplot as plt


class FinalTradingNetwork(nn.Module):
    """
    Final network that learns to match benchmark strategies
    """
    
    def __init__(self, lookback_long=200, lookback_short=50):
        super(FinalTradingNetwork, self).__init__()
        
        self.lookback_long = lookback_long
        self.lookback_short = lookback_short
        
        # Learnable MA weights - let the network learn the best lookbacks
        self.w_short = nn.Parameter(torch.zeros(lookback_long))
        self.w_long = nn.Parameter(torch.zeros(lookback_long))
        
        # Initialize with standard SMA
        with torch.no_grad():
            self.w_short[:lookback_short] = 1.0 / lookback_short
            self.w_long[:] = 1.0 / lookback_long
        
        # Simple decision layer
        self.decision = nn.Linear(2, 3)  # Input: [ma_short, ma_long], Output: [long, short, neutral]
        
    def forward(self, x):
        # Compute MAs
        ma_short = torch.sum(x * torch.softmax(self.w_short, dim=0), dim=1)
        ma_long = torch.sum(x * torch.softmax(self.w_long, dim=0), dim=1)
        
        # Stack features
        features = torch.stack([ma_short, ma_long], dim=1)
        
        # Normalize features
        features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
        
        # Decision
        logits = self.decision(features)
        positions = torch.softmax(logits * 2.0, dim=1)
        
        return positions, ma_short, ma_long


def train_with_benchmark(network, prices, benchmark_positions, strategy_name, epochs=1000, lr=0.01):
    """
    Train to match benchmark behavior
    """
    X, y, current_prices = prepare_data(prices, network.lookback_long)
    X_tensor = torch.FloatTensor(X)
    
    # Target positions from benchmark
    target_positions = torch.FloatTensor(benchmark_positions[:len(X)])
    
    optimizer = optim.Adam(network.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        positions, ma_short, ma_long = network(X_tensor)
        
        # Loss 1: Match benchmark positions
        position_loss = nn.functional.mse_loss(positions, target_positions)
        
        # Loss 2: Maximize returns
        returns = (y - current_prices) / (current_prices + 1e-8)
        returns_tensor = torch.FloatTensor(returns)
        position_weights = positions[:, 0] - positions[:, 1]
        strategy_returns = position_weights * returns_tensor
        return_loss = -strategy_returns.mean()
        
        # Combined loss
        total_loss = position_loss + 0.1 * return_loss
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            sharpe = calculate_sharpe_ratio(strategy_returns.detach().numpy())
            pos_match = 1 - position_loss.item()
            print(f"Epoch {epoch}: Sharpe={sharpe:.3f}, Position Match={pos_match:.1%}")


def comprehensive_test():
    """
    Comprehensive test of all strategies
    """
    print("COMPREHENSIVE TEST - MATCHING BENCHMARKS")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Uptrend
    print("\n1. UPTREND DATA")
    print("-" * 40)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    ou = OrnsteinUhlenbeckGenerator(theta=2, mu=50, sigma=20)
    prices = ou.generate_uptrend(10000)
    train_prices = prices[:8000]
    test_prices = prices[8000:]
    
    # Get benchmark
    benchmark_train = buy_and_hold_strategy(train_prices)
    benchmark_test = buy_and_hold_strategy(test_prices)
    
    # Train network
    network = FinalTradingNetwork()
    train_with_benchmark(network, train_prices, benchmark_train, "Buy & Hold", epochs=500)
    
    # Evaluate
    X_test, _, _ = prepare_data(test_prices, network.lookback_long)
    with torch.no_grad():
        positions, _, _ = network(torch.FloatTensor(X_test))
        positions = positions.numpy()
    
    returns_nn = calculate_returns(test_prices[200:], positions)
    sharpe_nn = calculate_sharpe_ratio(returns_nn)
    
    returns_bh = calculate_returns(test_prices[200:], benchmark_test)
    sharpe_bh = calculate_sharpe_ratio(returns_bh)
    
    print(f"\nTest Results:")
    print(f"Neural Network: Sharpe={sharpe_nn:.3f}, Long={np.mean(positions[:, 0]):.1%}")
    print(f"Buy & Hold: Sharpe={sharpe_bh:.3f}")
    
    results['uptrend'] = {'nn': sharpe_nn, 'benchmark': sharpe_bh}
    
    # Test 2: Switching Trend
    print("\n\n2. SWITCHING TREND DATA")
    print("-" * 40)
    
    ou = OrnsteinUhlenbeckGenerator(theta=7.5, mu=50, sigma=10)
    prices = ou.generate_switching_trend(10000)
    train_prices = prices[:8000]
    test_prices = prices[8000:]
    
    # Get benchmark
    benchmark_train = sma_momentum_strategy(train_prices)
    benchmark_test = sma_momentum_strategy(test_prices)
    
    # Train network
    network = FinalTradingNetwork()
    train_with_benchmark(network, train_prices, benchmark_train, "Momentum", epochs=1000)
    
    # Evaluate
    X_test, _, _ = prepare_data(test_prices, network.lookback_long)
    with torch.no_grad():
        positions, _, _ = network(torch.FloatTensor(X_test))
        positions = positions.numpy()
    
    returns_nn = calculate_returns(test_prices[200:], positions)
    sharpe_nn = calculate_sharpe_ratio(returns_nn)
    
    returns_mom = calculate_returns(test_prices[200:], benchmark_test)
    sharpe_mom = calculate_sharpe_ratio(returns_mom)
    
    print(f"\nTest Results:")
    print(f"Neural Network: Sharpe={sharpe_nn:.3f}, Long={np.mean(positions[:, 0]):.1%}")
    print(f"SMA Momentum: Sharpe={sharpe_mom:.3f}, Long={np.mean(benchmark_test[:, 0]):.1%}")
    
    results['switching'] = {'nn': sharpe_nn, 'benchmark': sharpe_mom}
    
    # Test 3: Reversion
    print("\n\n3. MEAN REVERSION DATA")
    print("-" * 40)
    
    ou = OrnsteinUhlenbeckGenerator(theta=20, mu=50, sigma=50)
    prices = ou.generate_reversion(10000)
    train_prices = prices[:8000]
    test_prices = prices[8000:]
    
    # Get benchmark
    benchmark_train = sma_reversion_strategy(train_prices)
    benchmark_test = sma_reversion_strategy(test_prices)
    
    # Train network
    network = FinalTradingNetwork()
    train_with_benchmark(network, train_prices, benchmark_train, "Reversion", epochs=1000)
    
    # Evaluate
    X_test, _, _ = prepare_data(test_prices, network.lookback_long)
    with torch.no_grad():
        positions, _, _ = network(torch.FloatTensor(X_test))
        positions = positions.numpy()
    
    returns_nn = calculate_returns(test_prices[200:], positions)
    sharpe_nn = calculate_sharpe_ratio(returns_nn)
    
    returns_rev = calculate_returns(test_prices[200:], benchmark_test)
    sharpe_rev = calculate_sharpe_ratio(returns_rev)
    
    print(f"\nTest Results:")
    print(f"Neural Network: Sharpe={sharpe_nn:.3f}")
    print(f"SMA Reversion: Sharpe={sharpe_rev:.3f}")
    
    results['reversion'] = {'nn': sharpe_nn, 'benchmark': sharpe_rev}
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    success_count = 0
    for strategy, res in results.items():
        ratio = res['nn'] / res['benchmark'] if res['benchmark'] != 0 else 0
        success = ratio >= 0.8
        success_count += success
        print(f"{strategy.capitalize()}: NN={res['nn']:.3f}, Benchmark={res['benchmark']:.3f}, "
              f"Ratio={ratio:.1%}, Success={'YES' if success else 'NO'}")
    
    print(f"\nOverall Success Rate: {success_count}/3")
    
    # Plot final comparison
    plt.figure(figsize=(10, 6))
    
    strategies = list(results.keys())
    nn_sharpes = [results[s]['nn'] for s in strategies]
    benchmark_sharpes = [results[s]['benchmark'] for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    plt.bar(x - width/2, nn_sharpes, width, label='Neural Network')
    plt.bar(x + width/2, benchmark_sharpes, width, label='Benchmark')
    
    plt.ylabel('Sharpe Ratio')
    plt.title('Neural Network vs Benchmark Performance')
    plt.xticks(x, [s.capitalize() for s in strategies])
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_results.png')
    print("\nPlot saved to 'final_results.png'")


if __name__ == "__main__":
    comprehensive_test()