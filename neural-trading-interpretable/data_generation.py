# data_generation.py - PAPER-MATCHING VERSION
"""
Synthetic data generation using Ornstein-Uhlenbeck processes
Based on user's implementation that matches paper's Figure 2
"""

import numpy as np


class OrnsteinUhlenbeckGenerator:
    """
    Generate synthetic data using Ornstein-Uhlenbeck process.
    Implementation based on matching the paper's actual figures.
    """
    
    def __init__(self, theta, mu, sigma):
        """
        Parameters from paper:
        θ: mean-reversion speed
        μ: long-term mean  
        σ: volatility
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
    
    def generate_uptrend(self, n_steps, initial_price=50, trend_rate=0.01):
        """
        Generate trending OU process - Paper Section 5.1
        θ = 2, μ = 50, σ = 20, trend = 0.01 per step
        """
        dt = 1.0 / n_steps  # As stated in paper
        sqrt_dt = np.sqrt(dt)
        
        prices = np.zeros(n_steps)
        prices[0] = initial_price
        
        for t in range(1, n_steps):
            # Time-varying mean
            mu_t = self.mu + trend_rate * t
            
            # Standard OU dynamics
            drift = self.theta * (mu_t - prices[t-1]) * dt
            diffusion = self.sigma * sqrt_dt * np.random.randn()
            
            prices[t] = prices[t-1] + drift + diffusion
        
        return prices
    
    def generate_switching_trend(self, n_steps, initial_price=50, switch_period=500):
        """
        Generate OU with alternating regimes
        Based on user's working implementation that matches paper figures
        
        Key insight: Use negative theta for down regimes!
        """
        dt = 1.0 / n_steps
        sqrt_dt = np.sqrt(dt)
        
        prices = np.zeros(n_steps)
        prices[0] = initial_price
        
        # Parameters for switching
        theta_up = 7.5    # During up regime
        theta_down = -2.5  # NEGATIVE for down regime (key difference!)
        
        for t in range(1, n_steps):
            # Always-increasing mean
            mu_t = self.mu + 0.01 * t
            
            # Determine regime and theta
            regime = (t // switch_period) % 2
            if regime == 0:
                theta_t = theta_up
            else:
                theta_t = theta_down  # Negative!
            
            # OU dynamics with regime-dependent theta
            drift = theta_t * (mu_t - prices[t-1]) * dt
            diffusion = self.sigma * sqrt_dt * np.random.randn()
            
            prices[t] = prices[t-1] + drift + diffusion
        
        return prices
    
    def generate_reversion(self, n_steps, initial_price=50):
        """
        Generate mean-reverting OU - Paper Section 5.1
        θ = 20, μ = 50, σ = 50
        """
        dt = 1.0 / n_steps
        sqrt_dt = np.sqrt(dt)
        
        prices = np.zeros(n_steps)
        prices[0] = initial_price
        
        for t in range(1, n_steps):
            # Fixed mean (no trend)
            drift = self.theta * (self.mu - prices[t-1]) * dt
            diffusion = self.sigma * sqrt_dt * np.random.randn()
            
            prices[t] = prices[t-1] + drift + diffusion
        
        return prices


# Test to verify it matches paper
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from benchmark_strategies import buy_and_hold_strategy, sma_momentum_strategy, sma_reversion_strategy
    from evaluation import calculate_returns, calculate_sharpe_ratio
    
    np.random.seed(42)
    
    print("Testing Paper-Matching Implementation")
    print("=" * 60)
    print("Key insight: Using negative theta for down regimes in switching trend")
    print("=" * 60)
    
    # Generate all three types
    # 1. Up-trend
    ou_up = OrnsteinUhlenbeckGenerator(theta=2, mu=50, sigma=20)
    prices_up = ou_up.generate_uptrend(10000)
    
    # 2. Switching trend (with corrected parameters)
    ou_switch = OrnsteinUhlenbeckGenerator(theta=7.5, mu=50, sigma=10)
    prices_switch = ou_switch.generate_switching_trend(10000)
    
    # 3. Mean reversion
    ou_rev = OrnsteinUhlenbeckGenerator(theta=20, mu=50, sigma=50)
    prices_rev = ou_rev.generate_reversion(10000)
    
    # Calculate Sharpe ratios on test sets
    # Up-trend
    test_up = prices_up[8000:]
    pos_bh = buy_and_hold_strategy(test_up)
    ret_bh = calculate_returns(test_up[200:], pos_bh)
    sharpe_bh = calculate_sharpe_ratio(ret_bh)
    
    pos_mom = sma_momentum_strategy(test_up)
    ret_mom = calculate_returns(test_up[200:], pos_mom)
    sharpe_mom = calculate_sharpe_ratio(ret_mom)
    
    print(f"\nUp-trend Results:")
    print(f"  Price range: {prices_up.min():.2f} to {prices_up.max():.2f}")
    print(f"  Buy & Hold Sharpe: {sharpe_bh:.2f} (Paper: 0.54)")
    print(f"  SMA-MOM Sharpe: {sharpe_mom:.2f} (Paper: -0.22)")
    
    # Switching
    test_switch = prices_switch[8000:]
    pos_mom2 = sma_momentum_strategy(test_switch)
    ret_mom2 = calculate_returns(test_switch[200:], pos_mom2)
    sharpe_mom2 = calculate_sharpe_ratio(ret_mom2)
    
    print(f"\nSwitching Trend Results:")
    print(f"  Price range: {prices_switch.min():.2f} to {prices_switch.max():.2f}")
    print(f"  SMA-MOM Sharpe: {sharpe_mom2:.2f} (Paper: 2.42)")
    
    # Reversion
    test_rev = prices_rev[8000:]
    pos_rev = sma_reversion_strategy(test_rev)
    ret_rev = calculate_returns(test_rev[200:], pos_rev)
    sharpe_rev = calculate_sharpe_ratio(ret_rev)
    
    print(f"\nMean Reversion Results:")
    print(f"  Price range: {prices_rev.min():.2f} to {prices_rev.max():.2f}")
    print(f"  SMA-REV Sharpe: {sharpe_rev:.2f} (Paper: 0.37)")
    
    # Plot all three series (matching paper's Figure 2)
    plt.figure(figsize=(12, 6))
    
    # Plot first 10000 steps to match paper
    plt.plot(prices_up, label="Up-Trend", linewidth=1, alpha=0.8)
    plt.plot(prices_switch, label="Switching Trend", linewidth=1, alpha=0.8)
    plt.plot(prices_rev, label="Reversion", linewidth=1, alpha=0.8)
    
    # Add regime boundaries for switching
    for k in range(1, 10000 // 500):
        plt.axvline(x=k*500, color="gray", linestyle="--", alpha=0.3)
    
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Artificial data series (Matching Paper Figure 2)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('paper_matching_data.png')
    
    # Zoom on switching behavior
    plt.figure(figsize=(10, 4))
    plt.plot(prices_switch[:1000], label="Switching Trend (first 2 regimes)", linewidth=1.2)
    plt.axvline(x=500, color="red", linestyle="--", alpha=0.7, label="Regime switch")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.title("Detailed View: Switching Trend with Negative Theta")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('switching_detail.png')
    
    print("\nPlots saved to 'paper_matching_data.png' and 'switching_detail.png'")