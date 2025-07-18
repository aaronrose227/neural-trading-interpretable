# evaluate.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

from train import get_data, prepare_artificial  # ensure these are importable
from train import WINDOW  # for consistency, if needed
from models import TradingPolicy  # if you need to refer to it

def get_model_positions(model: nn.Module,
                        prices: torch.Tensor,
                        window_long: int) -> torch.Tensor:
    """
    Roll forward the model in a sliding-window fashion to produce a
    time series of positions (long minus short) of length T.
    """
    seq = prices.squeeze(-1).numpy()  # shape (T,)
    T   = len(seq)
    pos = np.zeros(T, dtype=float)
    for t in range(window_long, T):
        window = torch.tensor(seq[t-window_long:t],
                              dtype=torch.float32).unsqueeze(0)  # (1, window_long)
        with torch.no_grad():
            p = model(window)  # (1,3)
        pos[t] = (p[0,1] - p[0,2]).item()
    return torch.tensor(pos, dtype=torch.float32)  # shape (T,)

def evaluate_model(model: nn.Module,
                   test_data: dict,
                   loss_history: list,
                   window_long: int,
                   window_short: int,
                   legend_labels=['Model','Buy & Hold']):
    """
    1) Fig 2: Plot the three synthetic regimes side-by-side.
    2) Fig 3: Rolling-window Strategy vs. Buy & Hold.
    3) Fig 4: Training Loss curve.
    4) Print numeric Sharpe.
    """
    prices  = test_data['prices']   # (T,1)
    returns = test_data['returns']  # (T,)

    # ---- Figure 2: three regimes ----
    regimes = [('Up-Trend', 'trend'),
               ('Mean-Revert', 'flat'),
               ('Switching',   'switch')]
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    for ax, (title, kind) in zip(axes, regimes):
        df = get_data(kind)
        ax.plot(df['Price'], label=title)
        ax.set_title(f"{title} Regime")
        ax.set_xlabel('Time Step')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---- Figure 3: rolling performance ----
    pos = get_model_positions(model, prices, window_long)  # (T,)
    strat_returns = returns * pos
    cum_strat     = torch.cumsum(strat_returns, dim=0).numpy()
    cum_bh        = np.cumsum(returns.numpy())

    plt.figure(figsize=(8,4))
    plt.plot(cum_strat, label=legend_labels[0])
    plt.plot(cum_bh,     label=legend_labels[1])
    plt.xlabel('Time (days)')
    plt.ylabel('Cumulative Returns')
    plt.title('Strategy vs. Buy & Hold')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- Figure 4: training loss curve ----
    plt.figure(figsize=(8,4))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- Numeric Sharpe ----
    mu     = strat_returns.mean().item()
    sigma  = strat_returns.std().item()
    sharpe = (mu * 252) / (sigma * np.sqrt(252))
    print(f"Sharpe ratio: {sharpe:.4f}")


