"""
data_generation_meta.py
Purpose: Self-contained synthetic data generator (Ornstein–Uhlenbeck variants)
used by the portable meta-agent. Matches the behavior used in your original repo.
"""

import numpy as np


class OrnsteinUhlenbeckGenerator:
    """
    Generate synthetic data using Ornstein-Uhlenbeck processes.
    """

    def __init__(self, theta, mu, sigma):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def generate_uptrend(self, n_steps, initial_price=50, trend_rate=0.01):
        dt = 1.0 / n_steps
        sqrt_dt = np.sqrt(dt)
        prices = np.zeros(n_steps)
        prices[0] = initial_price
        for t in range(1, n_steps):
            mu_t = self.mu + trend_rate * t
            drift = self.theta * (mu_t - prices[t - 1]) * dt
            diffusion = self.sigma * sqrt_dt * np.random.randn()
            prices[t] = prices[t - 1] + drift + diffusion
        return prices

    def generate_switching_trend(self, n_steps, initial_price=50, switch_period=500):
        dt = 1.0 / n_steps
        sqrt_dt = np.sqrt(dt)
        prices = np.zeros(n_steps)
        prices[0] = initial_price
        theta_up = 7.5
        theta_down = -2.5  # negative for down regime
        for t in range(1, n_steps):
            mu_t = self.mu + 0.01 * t
            regime = (t // switch_period) % 2
            theta_t = theta_up if regime == 0 else theta_down
            drift = theta_t * (mu_t - prices[t - 1]) * dt
            diffusion = self.sigma * sqrt_dt * np.random.randn()
            prices[t] = prices[t - 1] + drift + diffusion
        return prices

    def generate_reversion(self, n_steps, initial_price=50):
        dt = 1.0 / n_steps
        sqrt_dt = np.sqrt(dt)
        prices = np.zeros(n_steps)
        prices[0] = initial_price
        for t in range(1, n_steps):
            drift = self.theta * (self.mu - prices[t - 1]) * dt
            diffusion = self.sigma * sqrt_dt * np.random.randn()
            prices[t] = prices[t - 1] + drift + diffusion
        return prices
