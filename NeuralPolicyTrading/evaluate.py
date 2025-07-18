import torch
import numpy as np
import matplotlib.pyplot as plt
from train import get_data, train_synthetic, WINDOW
from torch import nn

def get_model_positions(model: nn.Module,
                        prices: torch.Tensor,
                        window_long: int,
                        train_len: int = 8000) -> torch.Tensor:
    """
    Returns a (2000,) tensor of positions on the test slice [8000:].
    """
    seq = prices.squeeze(-1).numpy()
    T   = len(seq)
    pos = np.zeros(T, dtype=float)

    for t in range(train_len, T):
        w = torch.tensor(seq[t-window_long:t], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p = model(w)
        pos[t] = (p[0,1] - p[0,2]).item()

    return torch.tensor(pos[train_len:], dtype=torch.float32)

def evaluate_synthetic(series_type: str,
                       window_short: int,
                       window_long: int,
                       beta: float = 10.0,
                       lr: float = 0.001,
                       epochs: int = 800):
    """
    Runs the full train/test pipeline on one regime and plots:
      - price series (Fig 2 panel)
      - performance vs BH on test slice (Fig 3 panel)
      - training loss (Fig 4 panel)
    Returns the test‐slice Sharpe.
    """
    # 1) train
    model, loss_history, prices, returns = train_synthetic(
        series_type,
        window_short,
        window_long,
        beta, lr, epochs
    )

    # 2) Figure 2 (this regime)
    plt.figure(figsize=(6,3))
    plt.plot(prices.numpy().flatten(), label=f"{series_type.title()}")
    plt.title(f"{series_type.title()} Series"); plt.xlabel("Time"); plt.ylabel("Price")
    plt.grid(alpha=0.3); plt.show()

    # 3) Figure 3 (test-slice)
    train_len = 8000
    test_returns = returns[train_len:]
    pos_test = get_model_positions(model, prices, window_long, train_len)
    cum_strat = torch.cumsum(test_returns * pos_test, dim=0).numpy()
    cum_bh    = np.cumsum(test_returns.numpy())

    plt.figure(figsize=(6,3))
    plt.plot(cum_strat, label='InterpretableNet')
    plt.plot(cum_bh,     label='Buy & Hold')
    plt.title(f"{series_type.title()} Strategy vs. BH (Test)")  
    plt.xlabel("Test Day"); plt.ylabel("Cumulative Return")
    plt.legend(); plt.grid(alpha=0.3); plt.show()

    # 4) Figure 4 (training loss)
    plt.figure(figsize=(6,3))
    plt.plot(loss_history, label='Training Loss')
    plt.title(f"{series_type.title()} Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(alpha=0.3); plt.show()

    # 5) Compute Sharpe on test slice
    r = test_returns * pos_test
    mu, sigma = r.mean().item(), r.std().item()
    sharpe = (mu * 252) / (sigma * np.sqrt(252))
    print(f"{series_type.title()} Sharpe (test): {sharpe:.2f}")
    return sharpe



