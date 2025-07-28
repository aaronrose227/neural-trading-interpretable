# README.md
"""
# Neural Policy Learning of Interpretable Trading Strategies

Implementation of the paper "Neural Policy Learning of Interpretable Trading Strategies using Inductive Prior Knowledge" by Krause & Calliess (2021).

## Overview

This project implements a neural network architecture that:
- Encodes traditional trading strategies (momentum, mean reversion) as neural networks
- Uses inductive priors in network design and weight initialization
- Provides interpretable trading decisions through logical rules
- Achieves performance comparable to complex black-box models

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run all experiments from the paper:
```bash
python experiments.py
```

### Train a custom network:
```python
from neural_policy_trading import TradingNetwork
from training import train_network

# Create network
network = TradingNetwork(lookback_long=200, lookback_short=50)

# Train on your data
train_network(network, prices, epochs=800, lr=0.001)
```

### Interpret learned strategies:
```python
interpretation = network.interpret_weights()
print(interpretation['logic_layer'])
```

## Project Structure

- `neural_policy_trading.py` - Main network architecture
- `data_generation.py` - Ornstein-Uhlenbeck process for synthetic data
- `benchmark_strategies.py` - SMA momentum/reversion strategies
- `evaluation.py` - Performance metrics (Sharpe ratio, returns)
- `training.py` - Training loops with different loss functions
- `experiments.py` - Reproduce paper results

## Key Features

1. **Interpretable Architecture**: Three-layer design mapping to logical operations
2. **Inductive Priors**: Initialize with known trading strategies
3. **Flexible Training**: Train specific layers or entire network
4. **Performance Metrics**: Sharpe ratio and return-based evaluation

## Results

The implementation successfully reproduces the paper's findings:
- Neural networks match or outperform traditional strategies
- Interpretable weights reveal learned trading logic
- Inductive priors improve training efficiency

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{krause2021neural,
  title={Neural Policy Learning of Interpretable Trading Strategies using Inductive Prior Knowledge},
  author={Krause, Fabian and Calliess, Jan-Peter},
  booktitle={ICAIF 2021 Workshop on Explainable AI in Finance},
  year={2021}
}
```
"""