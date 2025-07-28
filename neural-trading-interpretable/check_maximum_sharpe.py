# check_maximum_sharpe.py
"""
Check what maximum Sharpe ratio is theoretically possible on switching data
"""

import numpy as np
from data_generation import OrnsteinUhlenbeckGenerator
from evaluation import calculate_returns, calculate_sharpe_ratio
import matplotlib.pyplot as plt

np.random.seed(42)

print("ANALYZING MAXIMUM POSSIBLE SHARPE RATIOS")
print("=" * 60)

# Generate switching trend data
ou = OrnsteinUhlenbeckGenerator(theta=7.5, mu=50, sigma=10)
prices = ou.generate_switching_trend(10000)

# Test different scenarios
print("\n1. PERFECT FORESIGHT TEST")
print("-" * 40)

# Perfect strategy: always on the right side
perfect_positions = []
for i in range(1, len(prices)):
    if prices[i] > prices[i-1]:
        perfect_positions.append([1, 0, 0])  # Long
    else:
        perfect_positions.append([0, 1, 0])  # Short

perfect_positions = np.array(perfect_positions)
returns_perfect = calculate_returns(prices[1:], perfect_positions[:-1])
sharpe_perfect = calculate_sharpe_ratio(returns_perfect)
print(f"Perfect foresight Sharpe: {sharpe_perfect:.3f}")

# Oracle at regime switches
print("\n2. ORACLE AT REGIME SWITCHES")
print("-" * 40)

oracle_positions = []
for i in range(200, len(prices)-1):
    # Detect regime
    regime = (i // 500) % 2
    if regime == 0:  # Uptrend regime
        oracle_positions.append([1, 0, 0])  # Always long
    else:  # Downtrend regime
        oracle_positions.append([0, 1, 0])  # Always short

oracle_positions = np.array(oracle_positions)
returns_oracle = calculate_returns(prices[200:], oracle_positions)
sharpe_oracle = calculate_sharpe_ratio(returns_oracle)
print(f"Oracle (knows regimes) Sharpe: {sharpe_oracle:.3f}")

# Check with different parameters
print("\n3. DIFFERENT DATA PARAMETERS")
print("-" * 40)

param_tests = [
    {'theta_up': 7.5, 'theta_down': -2.5, 'sigma': 10, 'name': 'Paper params'},
    {'theta_up': 10.0, 'theta_down': -5.0, 'sigma': 5, 'name': 'Stronger trends'},
    {'theta_up': 15.0, 'theta_down': -10.0, 'sigma': 2, 'name': 'Very strong trends'},
]

for params in param_tests:
    ou_test = OrnsteinUhlenbeckGenerator(theta=params['theta_up'], mu=50, sigma=params['sigma'])
    
    # Custom switching generation
    prices_test = np.zeros(10000)
    prices_test[0] = 50
    
    for t in range(1, 10000):
        regime = (t // 500) % 2
        if regime == 0:
            theta_t = params['theta_up']
        else:
            theta_t = params['theta_down']
        
        mu_t = 50 + 0.01 * t
        drift = theta_t * (mu_t - prices_test[t-1]) * (1.0/10000)
        diffusion = params['sigma'] * np.sqrt(1.0/10000) * np.random.normal()
        prices_test[t] = prices_test[t-1] + drift + diffusion
    
    # Test momentum strategy
    from benchmark_strategies import sma_momentum_strategy
    positions = sma_momentum_strategy(prices_test)
    returns = calculate_returns(prices_test[200:], positions)
    sharpe = calculate_sharpe_ratio(returns)
    
    print(f"{params['name']}: Sharpe = {sharpe:.3f}")

# Analyze return distribution
print("\n4. RETURN DISTRIBUTION ANALYSIS")
print("-" * 40)

returns_all = (prices[1:] - prices[:-1]) / prices[:-1]
print(f"Daily return stats:")
print(f"  Mean: {np.mean(returns_all)*100:.3f}%")
print(f"  Std: {np.std(returns_all)*100:.3f}%")
print(f"  Sharpe (daily): {np.mean(returns_all)/np.std(returns_all):.3f}")
print(f"  Sharpe (annualized): {np.mean(returns_all)*np.sqrt(252)/np.std(returns_all):.3f}")

# Check if different test/train splits matter
print("\n5. DIFFERENT DATA SPLITS")
print("-" * 40)

from benchmark_strategies import sma_momentum_strategy

splits = [
    (8000, 2000, "Paper split (8000/2000)"),
    (5000, 5000, "50/50 split"),
    (9000, 1000, "90/10 split"),
    (0, 10000, "Full data (no split)")
]

for train_size, test_size, name in splits:
    if train_size > 0:
        test_data = prices[train_size:train_size+test_size]
    else:
        test_data = prices
    
    positions = sma_momentum_strategy(test_data)
    returns = calculate_returns(test_data[200:], positions)
    sharpe = calculate_sharpe_ratio(returns)
    print(f"{name}: Sharpe = {sharpe:.3f}")

# Visualize
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(prices[:2000])
for i in range(4):
    plt.axvline(x=i*500, color='r', linestyle='--', alpha=0.5)
plt.title('Switching Trend Price Data')
plt.ylabel('Price')

plt.subplot(2, 2, 2)
plt.hist(returns_all, bins=50, density=True)
plt.title('Return Distribution')
plt.xlabel('Daily Returns')
plt.ylabel('Density')

plt.subplot(2, 2, 3)
# Compare different strategies
strategies = ['Perfect', 'Oracle', 'SMA Mom', 'Buy Hold']
sharpes = [
    sharpe_perfect,
    sharpe_oracle,
    calculate_sharpe_ratio(calculate_returns(prices[200:], sma_momentum_strategy(prices))),
    calculate_sharpe_ratio(calculate_returns(prices[200:], np.tile([1,0,0], (len(prices)-201, 1))))
]
plt.bar(strategies, sharpes)
plt.title('Sharpe Ratios by Strategy')
plt.ylabel('Sharpe Ratio')

plt.subplot(2, 2, 4)
# Cumulative returns
positions = sma_momentum_strategy(prices)
returns = calculate_returns(prices[200:], positions)
cum_returns = np.cumprod(1 + returns)
plt.plot(cum_returns)
plt.title('Cumulative Returns (SMA Momentum)')
plt.ylabel('Cumulative Return')

plt.tight_layout()
plt.savefig('sharpe_analysis.png')

print("\nPlot saved to 'sharpe_analysis.png'")
print("\nCONCLUSION: The paper's Sharpe of ~2.5 seems unrealistically high")
print("unless they used different data parameters or calculation methods.")