import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
import random

from models import TradingPolicy   # your unified Input→Feature→Logic→Policy model

# ─── Reproducibility ────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ─── Synthetic Data Generator ───────────────────────────────────────
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
    """Return DataFrame with 'Price' length WINDOW for trend/flat/switch."""
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


# ─── Sliding‐Window Dataset ──────────────────────────────────────────
class WindowDataset(Dataset):
    def __init__(self, prices, returns, positions, window_long):
        # prices: (T,1), returns & positions: (T,)
        seq = prices.squeeze(-1).numpy()
        self.returns   = returns.numpy()
        self.positions = positions.numpy()
        self.window    = window_long
        self.seq       = seq

    def __len__(self):
        return len(self.seq) - self.window

    def __getitem__(self, idx):
        x     = torch.tensor(self.seq[idx:idx+self.window], dtype=torch.float32)
        y_ret = torch.tensor(self.returns[idx+self.window], dtype=torch.float32)
        y_pos = torch.tensor(self.positions[idx+self.window], dtype=torch.float32)
        return x, y_ret, y_pos


# ─── Loss ────────────────────────────────────────────────────────────
def compute_loss(returns, positions, is_real):
    if is_real:
        mu    = returns.mean()
        sigma = returns.std()
        return -(mu*252)/(sigma*math.sqrt(252))
    else:
        return -torch.mean(torch.log(1 + returns * positions))


# ─── Data Prep ───────────────────────────────────────────────────────
def prepare_artificial(series_type, window_long, window_short):
    df = get_data(series_type)
    df['return'] = df['Price'].pct_change().fillna(0)
    prices    = torch.tensor(df['Price'].values, dtype=torch.float32).unsqueeze(1)
    returns   = torch.tensor(df['return'].values, dtype=torch.float32)
    positions = torch.zeros_like(returns)   # dummy
    return {'prices': prices, 'returns': returns, 'positions': positions}

def prepare_real(symbol, window_long, window_short, api_key):
    from alpha_vantage.timeseries import TimeSeries
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='full')
    df = data['adjusted_close'].iloc[::-1].reset_index(drop=True).to_frame('Price')
    df['return'] = df['Price'].pct_change().fillna(0)
    prices    = torch.tensor(df['Price'].values, dtype=torch.float32).unsqueeze(1)
    returns   = torch.tensor(df['return'].values, dtype=torch.float32)
    positions = torch.zeros_like(returns)
    return {'prices':prices,'returns':returns,'positions':positions}


# ─── Training Loop ───────────────────────────────────────────────────
def train_model(model: nn.Module,
                data: dict,
                optimizer: Optimizer,
                is_real: bool,
                window_long: int,
                epochs: int = 800):
    prices    = data['prices']
    returns   = data['returns']
    positions = data['positions']

    dataset = WindowDataset(prices, returns, positions, window_long)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False)

    loss_history = []
    for epoch in range(epochs):
        for x, y_ret, y_pos in loader:
            optimizer.zero_grad()
            probs = model(x)   # x: (batch, window_long)

            if is_real:
                pred_pos = probs[:,1]
                loss     = compute_loss(y_ret, pred_pos, True)
            else:
                pred_pos = probs[:,1] - probs[:,2]
                loss     = compute_loss(y_ret, pred_pos, False)

            loss.backward()
            optimizer.step()

        loss_history.append(loss.item())

    return model, loss_history


