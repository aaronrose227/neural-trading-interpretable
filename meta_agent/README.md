# Portable Meta-Agent (Specialists + XGBoost Allocator)

Self-contained meta-agent for synthetic OU data with **no transaction costs**.  
It trains **specialist policies** (momentum short/long, reversion, neutral) and a supervised **allocator** (XGBoost) that blends the specialists based on regime features.

---

## What’s inside

- `data_generation_meta.py` — Ornstein–Uhlenbeck generators (uptrend / switching / reversion)  
- `evaluation_meta.py` — returns, Sharpe, window builder  
- `benchmark_strategies_meta.py` — SMA-Momentum / SMA-Reversion / Buy-&-Hold baselines  
- `regime_features_meta.py` — regime feature extractor (vol, trend strength, mean-rev, momentum consistency, autocorr)  
- `paper_exact_network_meta.py` — interpretable trading net (paper-style) + training loop  
- `specialists_meta.py` — build/train/load specialists; save checkpoints  
- `allocators_meta.py` — **XGBoost** allocator (supervised blender)  
- `portfolio_meta.py` — precompute specialist positions, blend with allocator, evaluate  
- `meta_experiments_meta.py` — **main script** (end-to-end run)  
- `__init__.py` — package marker  

---

## Folder structure

meta_agent_portable/
├─ init.py
├─ allocators_meta.py
├─ benchmark_strategies_meta.py
├─ data_generation_meta.py
├─ evaluation_meta.py
├─ meta_experiments_meta.py # run this
├─ paper_exact_network_meta.py
├─ portfolio_meta.py
├─ regime_features_meta.py
├─ specialists_meta.py
└─ README_meta.md

yaml
Copy
Edit

A `meta_agent_checkpoints/` folder will be created automatically to store trained specialists.

---

## Requirements

Known-good versions (match your existing pins):

```bash
pip install \
  numpy==1.24.3 \
  pandas==2.0.3 \
  torch==2.0.1 \
  scipy==1.11.1 \
  matplotlib==3.7.2 \
  xgboost==1.7.6 \
  scikit-learn==1.3.0
If you already have close versions, it’ll likely work. If you hit issues, use the exact pins above.

Quickstart
From inside meta_agent_portable/:

bash
Copy
Edit
python meta_experiments_meta.py

This will:

Generate Switching Trend OU data (10,000 points).

Split: Train (0–6,400) • Val (6,400–8,000) • Test (8,000–10,000).

Train 4 specialists (saved to meta_agent_checkpoints/).

Precompute specialist positions on Val/Test.

Train XGBoost allocator on Val (forward horizon = 20).

Blend specialists on Test and print OOS Sharpe.

Print SMA baselines for reference.

How it works (pipeline)
Specialists: Interpretable paper-style nets:

mom_short → short MA ~ 5, long MA 200

mom_long → short MA 50, long MA 200

reversion → same MAs, reversed logic

neutral → buy-and-hold-biased

Features: Volatility, trend strength, mean-reversion distance, momentum consistency, return autocorr.

Allocator (XGBoost): Maps features → non-negative weights over specialists using validation data (targets are forward-horizon summed returns per specialist, ReLU + normalize).

Changing regime / data type
Open meta_experiments_meta.py and switch the generator (default is Switching Trend):

python
Copy
Edit
from data_generation_meta import OrnsteinUhlenbeckGenerator

gen = OrnsteinUhlenbeckGenerator(theta=7.5, mu=50, sigma=10)
prices = gen.generate_switching_trend(10_000)

# Alternatives:
# prices = gen.generate_uptrend(10_000, trend_rate=0.01)
# prices = gen.generate_reversion(10_000)
Useful knobs (in code)
Validation forward horizon: in allocators_meta.py → AllocatorConfig(forward_horizon=20)

Split: in meta_experiments_meta.py → _split_series(...)

Specialist set: in specialists_meta.py → build_default_specialists(...)

Lookbacks: momentum short (5/200), momentum long (50/200) — tweak there if you want

No costs: returns are computed directly from positions (long-short-neutral probs)

Outputs you’ll see
OOS Sharpe per specialist on TEST

Meta-agent OOS Sharpe on TEST

Baselines: Sharpe for SMA-MOM, SMA-REV, Buy & Hold (TEST)

Artifacts (specialist checkpoints) are written to meta_agent_checkpoints/.

Notes & limitations
Synthetic data only (Ornstein–Uhlenbeck variants).

No transaction costs or slippage modeled.

Allocator is supervised (not RL); easy to extend later.

Specialists are frozen during allocator training/evaluation.

Troubleshooting
xgboost not found

bash
Copy
Edit
pip install xgboost==1.7.6
Torch CPU wheel issues
Install a platform-specific wheel from PyTorch’s site, then re-run.

Shape/alignment errors
Run from inside this folder so relative imports resolve; if you change split sizes, adjust alignment accordingly.

License / attribution
This package re-implements your interpretable trading network and experiments in a portable form for research use.
Feel free to adapt as needed.