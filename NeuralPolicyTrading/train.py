import torch
import random
import numpy as np
import pandas as pd
from torch.optim import Adam
from models import TradingPolicy

# ─── 0. Reproducibility ───────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ─── 1. Synthetic Data Generators (unchanged) ────────────────────────
WINDOW = 10000

def simulate_ou_standard(mu, theta, sigma, dt):
    T = mu.shape[0]
    sqrt_dt = torch.sqrt(torch.tensor(dt))
    P = torch.empty(T)
    P[0] = mu[0]
    for t in range(1, T):
        drift     = theta * (mu[t] - P[t-1]) * dt
        diffusion = sigma * sqrt_dt * torch.randn(())
        P[t] = P[t-1] + drift + diffusion
    return P

def simulate_ou_switching(mu, theta_pos, theta_neg, sigma, dt, switch_interval):
    T = mu.shape[0]
    sqrt_dt = torch.sqrt(torch.tensor(dt))
    P = torch.empty(T)
    P[0] = mu[0]
    for t in range(1, T):
        θ = theta_pos if ((t // switch_interval) % 2) == 0 else theta_neg
        drift     = θ * (mu[t] - P[t-1]) * dt
        diffusion = sigma * sqrt_dt * torch.randn(())
        P[t] = P[t-1] + drift + diffusion
    return P

def get_data(regime: str) -> pd.DataFrame:
    T, dt = WINDOW, 1.0/WINDOW
    trend_rate = 0.01
    # paper’s params
    σ_tr, σ_fl, σ_sw = 20.0, 50.0, 10.0
    θ_tr, θ_fl       = 2.0, 20.0
    θ_pos, θ_neg     = 7.5, -2.5
    reglen           = 500

    torch.manual_seed(42)
    mu_tr  = 50.0 + trend_rate * torch.arange(T)
    mu_fl  = torch.full((T,), 50.0)
    mu_sw  = mu_tr.clone()

    if regime=='trend':
        P = simulate_ou_standard(mu_tr, θ_tr, σ_tr, dt)
    elif regime=='flat':
        P = simulate_ou_standard(mu_fl, θ_fl, σ_fl, dt)
    elif regime=='switch':
        P = simulate_ou_switching(mu_sw, θ_pos, θ_neg, σ_sw, dt, reglen)
    else:
        raise ValueError

    return pd.DataFrame({'Price': P.numpy()})

# ─── 2. Loss ───────────────────────────────────────────────────────────
def neg_log_return_loss(R, X):
    return -torch.sum(torch.log(1 + R * X))

# ─── 3. Training Routine ───────────────────────────────────────────────
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
    warmup      = window_long
    train_start = warmup
    train_end   = train_start + 8000

    # 3) Build normalized windows
    windows = []
    for i in range(train_start, train_end):
        raw = prices[i:i+window_long].squeeze(-1)
        X_norm = (raw - raw.mean()) / (raw.std() + 1e-6)
        windows.append(X_norm)
    X = torch.stack(windows, dim=0)             # (8000, window_long)
    Y = returns[train_start:train_end]          # (8000,)

    # 4) Model
    model = TradingPolicy(
        window_short,
        window_long,
        beta=beta,
        input_trainable   = (variant == 'all'),
        feature_trainable = (variant in ['logic+feature','all'])
    )

    # 5) Optimizer: only if there are trainable params
    param_groups = [
        {'params': model.input_f.parameters(), 'lr': 0.01},
        {'params': model.feat_n.parameters(),  'lr': 0.01},
    ]
    param_groups = [g for g in param_groups if any(p.requires_grad for p in g['params'])]
    optimizer = Adam(param_groups) if param_groups else None

    # 6) Optional init prints
    print("  [Init] lin_short w[0:3]:", model.input_f.lin_short.weight.data.flatten()[:3].tolist())
    print("         tau1, tau2:", model.feat_n.tau1.item(), model.feat_n.tau2.item())

    # 7) Training loop (skip if no optimizer)
    loss_hist = []
    if optimizer is not None:
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = model(X)                    # (8000,3)
            pos    = logits[:,1] - logits[:,2]
            loss   = neg_log_return_loss(Y, pos)

            loss.backward()

            # (you may insert grad‐sum logging here)

            optimizer.step()
            loss_hist.append(loss.item())
    else:
        print("  [Info] No trainable params—skipping training loop for logic‐only variant.")

    # 8) Optional final prints
    print("  [Final] lin_short w[0:3]:", model.input_f.lin_short.weight.data.flatten()[:3].tolist())
    print("         tau1, tau2:", model.feat_n.tau1.item(), model.feat_n.tau2.item())

    return model, loss_hist, prices, returns, train_start
