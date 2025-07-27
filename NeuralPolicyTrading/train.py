import torch, random, numpy as np, pandas as pd
from torch.optim import Adam
from models import TradingPolicy

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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
        θ = theta_pos if ((t // switch_interval) % 2)==0 else theta_neg
        drift     = θ * (mu[t] - P[t-1]) * dt
        diffusion = sigma * sqrt_dt * torch.randn(())
        P[t] = P[t-1] + drift + diffusion
    return P

def get_data(regime: str) -> pd.DataFrame:
    T, dt = WINDOW, 1.0 / WINDOW
    trend_rate = 0.01
    σ_tr, σ_fl, σ_sw = 20.0, 50.0, 10.0
    θ_tr, θ_fl       = 2.0, 20.0
    θ_pos, θ_neg     = 7.5, -2.5
    reglen           = 500

    torch.manual_seed(42)
    mu_tr = 50.0 + trend_rate * torch.arange(T)
    mu_fl = torch.full((T,), 50.0)
    mu_sw = mu_tr.clone()

    if regime=='trend':
        P = simulate_ou_standard(mu_tr, θ_tr, σ_tr, dt)
    elif regime=='flat':
        P = simulate_ou_standard(mu_fl, θ_fl, σ_fl, dt)
    elif regime=='switch':
        P = simulate_ou_switching(mu_sw, θ_pos, θ_neg, σ_sw, dt, reglen)
    else:
        raise ValueError

    return pd.DataFrame({'Price': P.numpy()})

def neg_log_return_loss(R: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    # average log‐return per step
    return -torch.mean(torch.log(1 + R * X))

def train_synthetic(regime: str,
                    window_short: int,
                    window_long:  int,
                    variant:      str,
                    beta:         float = 10.0,
                    lr:           float = 0.01,
                    epochs:       int = 800):
    df      = get_data(regime)
    df['return'] = df['Price'].pct_change().fillna(0)
    prices  = torch.tensor(df['Price'].values,  dtype=torch.float32).unsqueeze(1)
    returns = torch.tensor(df['return'].values, dtype=torch.float32)

    warmup      = window_long
    train_start = warmup
    train_end   = train_start + 8000

    X = torch.stack([
        prices[i : i + window_long].squeeze(-1)
        for i in range(train_start, train_end)
    ], dim=0)
    Y = returns[train_start : train_end]

    model = TradingPolicy(
        window_short, window_long, beta=beta,
        input_trainable   = (variant == 'all'),
        feature_trainable = (variant in ['logic+feature','all']),
        logic_trainable   = (variant in ['logic','logic+feature','all'])  # <-- now True for 'all'
    )

    groups = []
    if variant in ['logic','logic+feature','all']:
        groups.append({'params': model.logic.parameters(), 'lr': lr})
    if variant in ['logic+feature','all']:
        groups.append({'params': model.feat_n.parameters(),  'lr': lr})
    if variant == 'all':
        groups.append({'params': model.input_f.parameters(), 'lr': lr})

    optimizer = Adam(groups)

    loss_hist = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        pos    = logits[:,1] - logits[:,2]
        loss   = neg_log_return_loss(Y, pos)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())

    return model, loss_hist, prices, returns, train_start
