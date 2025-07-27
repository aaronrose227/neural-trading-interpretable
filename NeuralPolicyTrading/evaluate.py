import torch, numpy as np, matplotlib.pyplot as plt
from train import train_synthetic, WINDOW

def get_model_positions(model, prices, window_long, train_start):
    seq = prices.squeeze(-1).numpy()
    T   = len(seq)
    pos = np.zeros(T)
    for t in range(window_long, T):
        w = torch.tensor(seq[t-window_long:t], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p = model(w)
        pos[t] = (p[0,1] - p[0,2]).item()
    return torch.tensor(pos[train_start:], dtype=torch.float32)

def evaluate_synthetic(regime, window_short, window_long, variant,
                       beta=10.0, lr=0.01, epochs=800, plot=True):
    model, loss_hist, prices, returns, train_start = train_synthetic(
        regime, window_short, window_long,
        variant, beta, lr, epochs
    )

    if plot:
        # Fig 2
        plt.figure(figsize=(5,2))
        plt.plot(prices.numpy()); plt.title(f"{regime.title()} Series"); plt.show()

        # Fig 3
        test_R = returns[train_start:]
        pos    = get_model_positions(model, prices, window_long, train_start)
        cum_s  = torch.cumsum(test_R*pos, dim=0).numpy()
        cum_bh = np.cumsum(test_R.numpy())
        plt.figure(figsize=(5,2))
        plt.plot(cum_s,label='ANN'); plt.plot(cum_bh,label='BH')
        plt.title(f"{regime.title()} Strat vs BH [{variant}]"); plt.show()

        # Fig 4
        plt.figure(figsize=(5,2))
        plt.plot(loss_hist); plt.title(f"{regime.title()} Loss [{variant}]"); plt.show()

    r  = returns[train_start:] * get_model_positions(model, prices, window_long, train_start)
    μ,σ = r.mean().item(), r.std().item()
    return (μ*252)/(σ*np.sqrt(252))
