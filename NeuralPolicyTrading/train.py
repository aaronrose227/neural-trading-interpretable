# train.py

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
import random

from alpha_vantage.timeseries import TimeSeries  # for real-data prep
from models import TradingPolicy  # your unified model class

# ─── 0. Reproducibility ─────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ─── 1. Synthetic‐series Simulation ──────────────────────────────────────
WINDOW = 10000  # length of each synthetic series

def simulate_ou_standard(mu: torch.Tensor,
                         theta: float,
                         sigma: float,
                         dt: float) -> torch.Tensor:
    T = mu.shape[0]
    sqrt_dt = math.sqrt(dt)
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
    sqrt_dt = math.sqrt(dt)
    P = torch.empty(T)
    P[0] = mu[0]
    for t in range(1, T):
        theta = theta_pos if ((t // switch_interval) % 2) == 0 else theta_neg
        drift     = theta * (mu[t] - P[t-1]) * dt
        diffusion = sigma * sqrt_dt * torch.randn(())
        P[t] = P[t-1] + drift + diffusion
    return P

def get_data(series_type: str) -> pd.DataFrame:
    """
    Returns DataFrame with column 'Price' of length WINDOW for:
      'trend', 'flat', or 'switch' regimes per your custom code.
    """
    T = WINDOW
    dt = 1.0 / T

    # mean-series
    trend_rate = 0.01
    mu_trend   = 50.0 + trend_rate * torch.arange(T, dtype=torch.float32)
    mu_flat    = torch.full((T,), 50.0)
    mu_switch  = mu_trend.clone()

    # OU params
    theta_trend, sigma_trend = 2.0, 20.0
    theta_flat,  sigma_flat  = 20.0, 50.0
    theta_pos,   theta_neg   = 7.5, -2.5
    sigma_switch = 10.0
    regime_len   = 500

    torch.manual_seed(42)
    if series_type == 'trend':
        P = simulate_ou_standard(mu_trend, theta_trend, sigma_trend, dt)
    elif series_type == 'flat':
        P = simulate_ou_standard(mu_flat, theta_flat, sigma_flat, dt)
    elif series_type == 'switch':
        P = simulate_ou_switching(mu_switch,
                                  theta_pos, theta_neg,
                                  sigma_switch, dt,
                                  switch_interval=regime_len)
    else:
        raise ValueError("series_type must be 'trend','flat' or 'switch'")

    return pd.DataFrame({'Price': P.numpy()})


# ─── 2. Sliding‐window Dataset ───────────────────────────────────────────
class WindowDataset(Dataset):
    def __init__(self, prices: torch.Tensor, returns: torch.Tensor,
                 positions: torch.Tensor, window_long: int):
        # prices: (T,1), returns & positions: (T,)
        seq = prices.squeeze(-1).numpy()    # shape (T,)
        self.returns   = returns.numpy()    # (T,)
        self.positions = positions.numpy()  # (T,)
        self.window    = window_long
        self.seq       = seq

    def __len__(self):
        return len(self.seq) - self.window

    def __getitem__(self, idx):
        # x: (window_long,)
        x     = torch.tensor(self.seq[idx:idx + self.window],
                             dtype=torch.float32)
        y_ret = torch.tensor(self.returns[idx + self.window],
                             dtype=torch.float32)
        y_pos = torch.tensor(self.positions[idx + self.window],
                             dtype=torch.float32)
        return x, y_ret, y_pos  # NO unsqueeze here

# ─── 3. Loss Function ───────────────────────────────────────────────────
def compute_loss(returns: torch.Tensor,
                 positions: torch.Tensor,
                 is_real: bool) -> torch.Tensor:
    if is_real:
        mu = returns.mean()
        sigma = returns.std()
        return -(mu * 252) / (sigma * math.sqrt(252))
    else:
        return -torch.mean(torch.log(1 + returns * positions))


# ─── 4. Data Preparation ─────────────────────────────────────────────────
def prepare_artificial(series_type: str,
                       window_long: int,
                       window_short: int):
    df = get_data(series_type)  # Price only
    df['return'] = df['Price'].pct_change().fillna(0)

    prices    = torch.tensor(df['Price'].values,
                             dtype=torch.float32).unsqueeze(1)
    returns   = torch.tensor(df['return'].values,
                             dtype=torch.float32)
    positions = torch.zeros_like(returns)  # dummy

    return {'prices': prices,
            'returns': returns,
            'positions': positions}

def prepare_real(symbol: str,
                 window_long: int,
                 window_short: int,
                 api_key: str):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='full')
    df = data['adjusted_close'].iloc[::-1].reset_index(drop=True) \
           .to_frame('Price')
    df['return'] = df['Price'].pct_change().fillna(0)

    prices  = torch.tensor(df['Price'].values,
                          dtype=torch.float32).unsqueeze(1)
    returns = torch.tensor(df['return'].values,
                          dtype=torch.float32)
    # dummy
    positions = torch.zeros_like(returns)

    return {'prices': prices,
            'returns': returns,
            'positions': positions}


# ─── 5. Training Loop ───────────────────────────────────────────────────
def train_model(model: nn.Module,
                data: dict,
                optimizer: Optimizer,
                is_real: bool,
                window_long: int,
                epochs: int = 800):
    prices    = data['prices']    # (T,1)
    returns   = data['returns']   # (T,)
    positions = data['positions'] # (T,)

    dataset = WindowDataset(prices, returns, positions, window_long)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False)

    loss_history = []
    for epoch in range(epochs):
        for x, y_ret, y_pos in loader:
            optimizer.zero_grad()
            # x: (batch, window_long)
            probs = model(x)  # forward expects (batch, seq_len)

            if is_real:
                pred_pos = probs[:, 1]
                loss = compute_loss(y_ret, pred_pos, True)
            else:
                pred_pos = probs[:, 1] - probs[:, 2]
                loss = compute_loss(y_ret, pred_pos, False)

            loss.backward()
            optimizer.step()

        loss_history.append(loss.item())

    return model, loss_history

