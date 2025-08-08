"""
meta_experiments_meta.py
Purpose: End-to-end script to train specialists, fit the allocator, and evaluate
the meta-agent on synthetic OU data (Switching Trend). No transaction costs.
"""

import numpy as np
import torch

from data_generation_meta import OrnsteinUhlenbeckGenerator
from benchmark_strategies_meta import sma_momentum_strategy, sma_reversion_strategy, buy_and_hold_strategy
from evaluation_meta import calculate_returns, calculate_sharpe_ratio
from specialists_meta import build_default_specialists, load_specialist, eval_specialist_on_prices
from allocators_meta import XGBAllocator, AllocatorConfig
from portfolio_meta import precompute_specialist_positions, blend_positions, evaluate_blended


def _split_series(prices):
    train = prices[:6400]
    val = prices[6400:8000]
    test = prices[8000:]
    return train, val, test


def _specialist_oos_returns_on_window(positions_by_kind, prices, lookback_long):
    returns = {}
    for k, pos in positions_by_kind.items():
        price_subset = prices[lookback_long:]
        returns[k] = calculate_returns(price_subset, pos)
    return returns


def run_meta_agent_experiment(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("META-AGENT (portable) — OU synthetic, no costs")
    print("=" * 60)

    # 1) Data
    gen = OrnsteinUhlenbeckGenerator(theta=7.5, mu=50, sigma=10)
    prices = gen.generate_switching_trend(10_000)
    train_prices, val_prices, test_prices = _split_series(prices)
    print(f"Data split: Train={len(train_prices)}, Val={len(val_prices)}, Test={len(test_prices)}")

    # 2) Specialists
    print("\nTraining specialists...")
    ckpts = build_default_specialists(train_prices, out_dir="meta_agent_checkpoints", seed=seed)
    order = ["mom_short", "mom_long", "reversion", "neutral"]
    lookback_long = 200

    print("\nSpecialists OOS Sharpe on TEST:")
    for k in order:
        net = load_specialist(ckpts[k])
        s = eval_specialist_on_prices(net, test_prices)
        print(f"  {k:11s}: Sharpe {s:.2f}")

    # 3) Precompute positions on Val/Test
    print("\nPrecomputing positions...")
    pos_val = precompute_specialist_positions(ckpts, val_prices)
    pos_test = precompute_specialist_positions(ckpts, test_prices)

    # 4) Allocator (train on Val)
    print("\nTraining allocator (XGBoost)...")
    spec_returns_val = _specialist_oos_returns_on_window(pos_val, val_prices, lookback_long)
    alloc = XGBAllocator(order, cfg=AllocatorConfig(forward_horizon=20))
    ok = alloc.fit(train_prices, val_prices, spec_returns_val)
    if not ok:
        print("Allocator training failed.")
        return

    # 5) Inference on Test (blend)
    print("\nBlending specialists on TEST...")
    T = min(m.shape[0] for m in pos_test.values())
    weights = np.zeros((T, len(order)))
    for t in range(T):
        prefix = test_prices[: (t + lookback_long + 1)]
        weights[t] = alloc.predict_weights(prefix)
    blended_positions = blend_positions(pos_test, weights, order)
    meta_sharpe = evaluate_blended(test_prices, blended_positions, lookback_long)
    print(f"\nMeta-agent OOS Sharpe (TEST): {meta_sharpe:.2f}")

    # 6) Benchmarks
    print("\nBenchmarks (TEST):")
    pos_m = sma_momentum_strategy(test_prices)
    pos_r = sma_reversion_strategy(test_prices)
    pos_b = buy_and_hold_strategy(test_prices)
    sr_m = calculate_sharpe_ratio(calculate_returns(test_prices[200:], pos_m))
    sr_r = calculate_sharpe_ratio(calculate_returns(test_prices[200:], pos_r))
    sr_b = calculate_sharpe_ratio(calculate_returns(test_prices[200:], pos_b))
    print(f"  SMA-MOM:    {sr_m:.2f}")
    print(f"  SMA-REV:    {sr_r:.2f}")
    print(f"  Buy & Hold: {sr_b:.2f}")

    print("\nDone.")
    return {"meta_sharpe": meta_sharpe, "ckpts": ckpts}


if __name__ == "__main__":
    run_meta_agent_experiment()
