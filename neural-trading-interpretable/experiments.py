#experiments.py

"""
Main experiments to reproduce paper results
Following Section 5 experimental setup exactly
"""

import numpy as np
import matplotlib.pyplot as plt
from neural_policy_trading import TradingNetwork
from data_generation import OrnsteinUhlenbeckGenerator
from benchmark_strategies import (
    sma_momentum_strategy,
    sma_reversion_strategy,
    buy_and_hold_strategy
)
from evaluation import (
    calculate_returns,
    calculate_sharpe_ratio,
    prepare_data,
    calculate_cumulative_returns
)
from training import train_network
import torch


def evaluate_strategy(network, prices, strategy_name="Neural Network"):
    """
    Evaluate a trading strategy and return performance metrics
    Following the paper's evaluation approach
    """
    X, y, current_prices = prepare_data(prices, network.lookback_long)
    X_tensor = torch.FloatTensor(X)

    with torch.no_grad():
        positions = network(X_tensor).numpy()

    price_subset      = prices[network.lookback_long:]
    returns           = calculate_returns(price_subset, positions)
    cumulative_returns = calculate_cumulative_returns(returns)
    sharpe            = calculate_sharpe_ratio(returns)

    return {
        'returns': returns,
        'cumulative_returns': cumulative_returns,
        'sharpe': sharpe,
        'positions': positions,
        'name': strategy_name
    }


def run_uptrend_experiment():
    """
    Experiment 1: Up-trend data
    Paper Section 5.2.1: Only train logic layer
    Expected result: Network learns buy-and-hold strategy
    """
    print("=" * 60)
    print("EXPERIMENT 1: UP-TREND DATA")
    print("=" * 60)

    # Generate data with exact paper parameters
    ou = OrnsteinUhlenbeckGenerator(theta=2, mu=50, sigma=20)
    prices = ou.generate_uptrend(10_000, trend_rate=0.01)

    # Split train/test as per paper: 8000/2000
    train_prices = prices[:8_000]
    test_prices  = prices[8_000:]

    # Train network (logic layer only) - Section 5.2.1
    print("\nTraining neural network (logic layer only)...")
    network = TradingNetwork()
    losses = train_network(
        network,
        train_prices,
        epochs=800,
        lr=0.001,
        train_layers='logic'
    )

    # Evaluate on test set
    results_nn = evaluate_strategy(network, test_prices, "Neural Network")
    sharpe_nn  = results_nn['sharpe']

    # Benchmark strategies
    positions_mom = sma_momentum_strategy(test_prices)
    returns_mom   = calculate_returns(test_prices[200:], positions_mom)
    sharpe_mom    = calculate_sharpe_ratio(returns_mom)

    positions_bh = buy_and_hold_strategy(test_prices)
    returns_bh   = calculate_returns(test_prices[200:], positions_bh)
    sharpe_bh    = calculate_sharpe_ratio(returns_bh)

    print(f"\nTest Set Results:")
    print(f"Neural Network Sharpe: {sharpe_nn:.2f}")
    print(f"SMA-MOM Sharpe:        {sharpe_mom:.2f}")
    print(f"Buy & Hold Sharpe:     {sharpe_bh:.2f}")

    print("\nLearned strategy interpretation:")
    interp = network.interpret_weights()
    print(f"Long logic:    {interp['logic_layer']['long']}")
    print(f"Short logic:   {interp['logic_layer']['short']}")
    print(f"Neutral logic: {interp['logic_layer']['neutral']}")

    return {
        'network': network,
        'sharpe_nn': sharpe_nn,
        'sharpe_mom': sharpe_mom,
        'sharpe_bh': sharpe_bh
    }


def run_switching_trend_experiment():
    """
    Experiment 2: Switching trend data
    Paper Section 5.2.2: Train logic and feature layers
    Expected result: Network adapts momentum strategy
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: SWITCHING TREND DATA")
    print("=" * 60)

    # Generate data with exact paper parameters
    ou = OrnsteinUhlenbeckGenerator(theta=7.5, mu=50, sigma=10)
    prices = ou.generate_switching_trend(10_000)

    # Handle NaNs if any
    if np.any(np.isnan(prices)):
        print("Warning: NaN values in generated prices; filling with μ")
        prices = np.nan_to_num(prices, nan=50.0)

    # Split train/test
    train_prices = prices[:8_000]
    test_prices  = prices[8_000:]

    # Train network (logic + feature layers) - Section 5.2.2
    print("\nTraining neural network (logic and feature layers)...")
    network = TradingNetwork()
    losses = train_network(
        network,
        train_prices,
        epochs=800,
        lr=0.001,
        train_layers='logic_feature'
    )

    # Evaluate on test set
    results_nn = evaluate_strategy(network, test_prices, "Neural Network")
    sharpe_nn  = results_nn['sharpe']

    # Benchmark
    positions_mom = sma_momentum_strategy(test_prices)
    returns_mom   = calculate_returns(test_prices[200:], positions_mom)
    sharpe_mom    = calculate_sharpe_ratio(returns_mom)

    print(f"\nTest Set Results:")
    print(f"Neural Network Sharpe: {sharpe_nn:.2f}")
    print(f"SMA-MOM Sharpe:        {sharpe_mom:.2f}")

    print("\nFinal feature layer weights:")
    print("Feature neuron 1:", network.feature_layer.weight[0].detach().numpy())
    print("Feature neuron 2:", network.feature_layer.weight[1].detach().numpy())

    return {
        'network': network,
        'sharpe_nn': sharpe_nn,
        'sharpe_mom': sharpe_mom
    }


def run_reversion_experiment():
    """
    Experiment 3: Mean reversion data
    Paper Section 5.2.3: Train all layers with reversion initialization
    Expected result: Network learns reversion strategy
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: MEAN REVERSION DATA")
    print("=" * 60)

    # Generate data with exact paper parameters
    ou = OrnsteinUhlenbeckGenerator(theta=20, mu=50, sigma=50)
    prices = ou.generate_reversion(10_000)

    # Handle NaNs if any
    if np.any(np.isnan(prices)):
        print("Warning: NaN values in generated prices; filling with μ")
        prices = np.nan_to_num(prices, nan=50.0)

    # Split train/test
    train_prices = prices[:8_000]
    test_prices  = prices[8_000:]

    # Train network (all layers, reversion init) - Section 5.2.3
    print("\nTraining neural network (all layers, reversion init)...")
    network = TradingNetwork()
    network._initialize_reversion_strategy()
    losses = train_network(
        network,
        train_prices,
        epochs=800,
        lr=0.001,
        train_layers='all'
    )

    # Evaluate on test set
    results_nn = evaluate_strategy(network, test_prices, "Neural Network")
    sharpe_nn  = results_nn['sharpe']

    # Benchmark
    positions_rev = sma_reversion_strategy(test_prices)
    returns_rev   = calculate_returns(test_prices[200:], positions_rev)
    sharpe_rev    = calculate_sharpe_ratio(returns_rev)

    print(f"\nTest Set Results:")
    print(f"Neural Network Sharpe: {sharpe_nn:.2f}")
    print(f"SMA-REV Sharpe:        {sharpe_rev:.2f}")

    return {
        'network': network,
        'sharpe_nn': sharpe_nn,
        'sharpe_rev': sharpe_rev
    }


def main():
    """Run all experiments to reproduce Table 1"""
    np.random.seed(42)
    torch.manual_seed(42)

    results_uptrend  = run_uptrend_experiment()
    results_switch  = run_switching_trend_experiment()
    results_reversion = run_reversion_experiment()

    # Summary table (matching Table 1 from paper)
    print("\n" + "=" * 60)
    print("SUMMARY - Table 1 Reproduction")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Up-trend':<15} {'Switching':<15} {'Reversion':<15}")
    print("-" * 65)
    print(f"{'Neural Network':<20} {results_uptrend['sharpe_nn']:<15.2f} {results_switch['sharpe_nn']:<15.2f} {results_reversion['sharpe_nn']:<15.2f}")
    print(f"{'SMA-MOM':<20} {results_uptrend['sharpe_mom']:<15.2f} {results_switch['sharpe_mom']:<15.2f} {-results_reversion['sharpe_rev']:<15.2f}")
    print(f"{'SMA-REV':<20} {-results_uptrend['sharpe_mom']:<15.2f} {-results_switch['sharpe_mom']:<15.2f} {results_reversion['sharpe_rev']:<15.2f}")
    print(f"{'Buy & Hold':<20} {results_uptrend['sharpe_bh']:<15.2f} {'N/A':<15} {'N/A':<15}")

    print("\nExperiments completed!")

    # Paper target values for reference
    print("\n" + "=" * 60)
    print("Paper's Table 1 Target Values:")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Up-trend':<15} {'Switching':<15} {'Reversion':<15}")
    print("-" * 65)
    print(f"{'ANN Logic only':<20} {'0.54':<15} {'2.40':<15} {'0.50':<15}")
    print(f"{'ANN Logic+Feature':<20} {'0.54':<15} {'2.47':<15} {'0.34':<15}")
    print(f"{'ANN all':<20} {'-0.23':<15} {'2.47':<15} {'0.41':<15}")
    print(f"{'SMA-MOM':<20} {'-0.22':<15} {'2.42':<15} {'-0.37':<15}")
    print(f"{'SMA-REV':<20} {'0.22':<15} {'-2.42':<15} {'0.37':<15}")
    print(f"{'Buy & Hold':<20} {'0.54':<15} {'0.70':<15} {'0.01':<15}")


if __name__ == "__main__":
    main()