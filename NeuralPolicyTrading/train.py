import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import random
from torch.optim import Adam

from models import TradingPolicy

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Synthetic‐series generator (your code)
WINDOW = 10000

def simulate_ou_standard(mu, theta, sigma, dt):
    T = mu.shape[0]
    P = torch.empty(T)
    P[0] = mu[0]
    for t in range(1, T):
        drift     = theta * (mu[t] - P[t-1]) * dt
        diffusion = sigma * math.sqrt(dt) * torch.randn(())
        P[t] = P[t-1] + drift + diffusion
    return P

def simulate_ou_switching(mu, theta_pos, theta_neg, sigma, dt, switch_interval):
    T = mu.shape[0]
    P = torch.empty(T)
    P[0] = mu[0]
    for t in range(1, T):
        theta = theta_pos if ((t // switch_interval) % 2)==0 else theta_neg
        drift     = theta * (mu[t] - P[t-1]) * dt
        diffusion = sigma * math.sqrt(dt) * torch.randn(())
        P[t] = P[t-1] + drift + diffusion
    return P

def get_data(series_type: str) -> pd.DataFrame:
    """Return DataFrame with column 'Price' of length WINDOW."""
    T, dt = WINDOW, 1.0/WINDOW
    trend_rate = 0.01
    mu_trend   = 50.0 + trend_rate * torch.arange(T)
    mu_flat    = torch.full((T,), 50.0)
    mu_switch  = mu_trend.clone()
    torch.manual_seed(42)

    if series_type=='trend':
        P = simulate_ou_standard(mu_trend, 2.0, 20.0, dt)
    elif series_type=='flat':
        P = simulate_ou_standard(mu_flat, 20.0, 50.0, dt)
    elif series_type=='switch':
        P = simulate_ou_switching(mu_switch, 7.5, -2.5, 10.0, dt, switch_interval=500)
    else:
        raise ValueError("series_type must be 'trend','flat','switch'")

    return pd.DataFrame({'Price': P.numpy()})

# Negative log-return loss (synthetic)
def compute_loss(returns, positions):
    return -torch.mean(torch.log(1 + returns * positions))

def train_synthetic(series_type: str,
                    window_short: int,
                    window_long: int,
                    beta: float = 10.0,
                    lr: float = 0.001,
                    epochs: int = 800):
    """
    1) Generate series and split 8000/2000
    2) Full-batch gradient descent on train slice
    3) Returns trained model, loss history, full prices & returns
    """
    # 1) build DataFrame
    df = get_data(series_type)
    df['return'] = df['Price'].pct_change().fillna(0)
    prices  = torch.tensor(df['Price'].values,  dtype=torch.float32).unsqueeze(1)
    returns = torch.tensor(df['return'].values, dtype=torch.float32)

    # 2) split
    train_len = 8000
    train_prices  = prices[:train_len]
    train_returns = returns[:train_len]

    # 3) prepare windows
    # X: (N, window_long), Y: (N,)
    N = train_len - window_long
    X = torch.stack([
        train_prices[i:i+window_long].squeeze(-1)
        for i in range(N)
    ], dim=0)
    Y = train_returns[window_long:window_long+N]

    # 4) model & optimizer
    model = TradingPolicy(window_short, window_long, beta)
    opt   = Adam(model.parameters(), lr=lr)

    # 5) training loop (full-batch)
    loss_history = []
    for _ in range(epochs):
        opt.zero_grad()
        probs = model(X)                      # (N,3)
        positions = probs[:,1] - probs[:,2]   # (N,)
        loss = compute_loss(Y, positions)
        loss.backward()
        opt.step()
        loss_history.append(loss.item())

    return model, loss_history, prices, returns




