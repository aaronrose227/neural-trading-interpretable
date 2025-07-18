import torch
import math
import random
import numpy as np
import pandas as pd
from torch.optim import Adam
from models import TradingPolicy

# ─── 0. Reproducibility ───────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ─── 1. Original Synthetic‐Data Generators ────────────────────────────
WINDOW = 10000

def simulate_ou_standard(mu: torch.Tensor,
                         theta: float,
                         sigma: float,
                         dt: float) -> torch.Tensor:
    T = mu.shape[0]
    sqrt_dt = torch.sqrt(torch.tensor(dt))
    P = torch.empty(T)
    P[0] = mu[0]
    for t in range(1, T):
        drift     = theta * (mu[t] - P[t-1]) * dt
        diffusion = sigma * sqrt_dt * torch.randn(())
        P[t] = P[t-1] + drift + diffusion
    return P

def simulate_ou_switching(mu: torch.Tensor,
                          theta_pos: float,
                          theta_neg: float,
                          sigma: float,
                          dt: float,
                          switch_interval: int) -> torch.Tensor:
    T = mu.shape[0]
    sqrt_dt = torch.sqrt(torch.tensor(dt))
    P = torch.empty(T)
    P[0] = mu[0]
    for t in range(1, T):
        theta = theta_pos if ((t // switch_interval) % 2) == 0 else theta_neg
        drift     = theta * (mu[t] - P[t-1]) * dt
        diffusion = sigma * sqrt_dt * torch.randn(())
        P[t] = P[t-1] + drift + diffusion
    return P

def get_data(regime: str) -> pd.DataFrame:
    """
    series_type: 'trend', 'flat', or 'switch'
    returns a DataFrame with column 'Price' of length WINDOW,
    using your original parameterization.
    """
    T      = WINDOW
    dt     = 1.0 / T

    trend_rate   = 0.01
    sigma_trend  = 20.0
    sigma_flat   = 50.0
    sigma_switch = 10.0
    theta_trend  = 2.0
    theta_flat   = 20.0
    theta_pos    = 7.5
    theta_neg    = -2.5
    regime_len   = 500

    mu_trend  = 50.0 + trend_rate * torch.arange(T, dtype=torch.float32)
    mu_flat   = torch.full((T,), 50.0, dtype=torch.float32)
    mu_switch = mu_trend.clone()

    if regime == 'trend':
        P = simulate_ou_standard(mu_trend, theta_trend, sigma_trend, dt)
    elif regime == 'flat':
        P = simulate_ou_standard(mu_flat, theta_flat, sigma_flat, dt)
    elif regime == 'switch':
        P = simulate_ou_switching(mu_switch,
                                  theta_pos, theta_neg,
                                  sigma_switch, dt,
                                  switch_interval=regime_len)
    else:
        raise ValueError("regime must be 'trend','flat' or 'switch'")

    return pd.DataFrame({'Price': P.numpy()})

# ─── 2. Loss ───────────────────────────────────────────────────────────
def neg_log_return_loss(R: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    # sum‐based per paper
    return -torch.sum(torch.log(1 + R * X))

# ─── 3. Train Synthetic ────────────────────────────────────────────────
def train_synthetic(regime: str,
                    window_short: int,
                    window_long:  int,
                    variant:      str,
                    beta:         float = 10.0,
                    lr:           float = 0.001,
                    epochs:       int = 800):
    """
    Train on points 200–8199 (8 000 samples), test on 8200–9999 (2 000 samples).
    variant in {'logic','logic+feature','all'}.

    Returns:
      model, loss_hist, prices, returns, train_start
    """
    # 1) Data
    df = get_data(regime)
    df['return'] = df['Price'].pct_change().fillna(0)
    prices  = torch.tensor(df['Price'].values,  dtype=torch.float32).unsqueeze(1)
    returns = torch.tensor(df['return'].values, dtype=torch.float32)

    # 2) Indices
    warmup      = window_long            # 200
    train_start = warmup                 # idx 200
    train_end   = train_start + 8000     # idx 8200

    # 3) Build train windows
    X = torch.stack([
        prices[i : i + window_long].squeeze(-1)
        for i in range(train_start, train_end)
    ], dim=0)                              # (8000, window_long)
    Y = returns[train_start : train_end]   # (8000,)

    # 4) Model
    model = TradingPolicy(
        window_short,
        window_long,
        beta=beta,
        input_trainable   = (variant == 'all'),
        feature_trainable = (variant in ['logic+feature','all']),
        logic_trainable   = True
    )

    # 5) Optimizer
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # 6) Training loop
    loss_hist = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X)             # (8000,3)
        pos    = logits[:,1] - logits[:,2]
        loss   = neg_log_return_loss(Y, pos)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())

    return model, loss_hist, prices, returns, train_start
