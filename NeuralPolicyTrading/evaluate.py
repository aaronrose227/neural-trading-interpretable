import torch
import numpy as np
import matplotlib.pyplot as plt
from train import train_synthetic, get_data, WINDOW

def get_model_positions(model, prices, window_long, train_start):
    """
    Roll through the full price series, feeding in normalized windows
    and extracting the long–short position = p[1] – p[2].
    Returns a Tensor of positions for the test slice.
    """
    seq = prices.squeeze(-1).numpy()
    T   = len(seq)
    pos = np.zeros(T)
    for t in range(window_long, T):
        raw = torch.tensor(seq[t-window_long:t], dtype=torch.float32)
        w   = ((raw - raw.mean()) / (raw.std() + 1e-6)).unsqueeze(0)
        with torch.no_grad():
            p = model(w)
        pos[t] = (p[0,1] - p[0,2]).item()
    return torch.tensor(pos[train_start:], dtype=torch.float32)

def evaluate_synthetic(regime: str,
                       window_short: int,
                       window_long:  int,
                       variant:      str,
                       beta:         float = 10.0,
                       lr:           float = 0.001,
                       epochs:       int = 800,
                       plot:         bool  = True):
    """
    1) Trains model on synthetic data via train_synthetic(...)
    2) Optionally plots:
       - The full price series
       - Cumulative returns of ANN vs. BH on test slice
       - Training‐loss curve
    3) Returns the test‐slice Sharpe ratio.
    """
    # 1) Train
    model, loss_hist, prices, returns, train_start = train_synthetic(
        regime, window_short, window_long,
        variant, beta, lr, epochs
    )

    if plot:
        # Fig 2: Price series
        plt.figure(figsize=(5,2))
        plt.plot(prices.numpy(), label=regime.title())
        plt.title(f"{regime.title()} Series")
        plt.xlabel("Time"); plt.ylabel("Price")
        plt.grid(alpha=0.3)
        plt.show()

        # Fig 3: Strategy vs. Buy & Hold on test slice
        test_R = returns[train_start:]
        pos    = get_model_positions(model, prices, window_long, train_start)
        cum_s  = torch.cumsum(test_R * pos, dim=0).numpy()
        cum_bh = np.cumsum(test_R.numpy())
        plt.figure(figsize=(5,2))
        plt.plot(cum_s,  label='ANN')
        plt.plot(cum_bh, label='BH')
        plt.title(f"{regime.title()} Strategy vs BH [{variant}]")
        plt.xlabel("Test Day"); plt.ylabel("Cumulative Return")
        plt.grid(alpha=0.3)
        plt.show()

        # Fig 4: Training‐loss curve
        plt.figure(figsize=(5,2))
        plt.plot(loss_hist)
        plt.title(f"{regime.title()} Training Loss [{variant}]")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        plt.show()

    # 2) Compute & return test Sharpe
    test_R = returns[train_start:]
    pos    = get_model_positions(model, prices, window_long, train_start)
    r      = test_R * pos
    μ, σ   = r.mean().item(), r.std().item()
    return (μ * 252) / (σ * np.sqrt(252))
