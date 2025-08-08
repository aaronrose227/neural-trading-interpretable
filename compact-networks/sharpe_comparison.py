# sharpe_comparison.py
"""
Focused Sharpe ratio comparison across market regimes.
Creates the specific table and graph you requested.
UPDATED: Uses the NEW VERSION of compact_network.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from data_generation import OrnsteinUhlenbeckGenerator
from financial_indicators import prepare_indicator_data
from compact_network import CompactFinancialNetwork, CompactNetworkTrainer
from benchmark_strategies import sma_momentum_strategy, sma_reversion_strategy
from evaluation import calculate_sharpe_ratio


def run_sharpe_comparison():
    """Run comprehensive Sharpe ratio comparison across regimes"""
    
    print("SHARPE RATIO COMPARISON ACROSS MARKET REGIMES")
    print("=" * 60)
    print("Using NEW VERSION compact network based on original paper approach")
    
    # Define test regimes matching your original work
    regimes = {
        'Switching_Trend': {
            'generator_params': {'theta': 7.5, 'mu': 50, 'sigma': 10},
            'data_method': 'generate_switching_trend',
            'network_regime': 'momentum',
            'target_sharpe': 2.2  # Your original target
        },
        'Mean_Reversion': {
            'generator_params': {'theta': 20, 'mu': 50, 'sigma': 50},
            'data_method': 'generate_reversion', 
            'network_regime': 'reversion',
            'target_sharpe': 0.37  # Approximate target
        },
        'Up_Trend': {
            'generator_params': {'theta': 2, 'mu': 50, 'sigma': 20},
            'data_method': 'generate_uptrend',
            'network_regime': 'trending',
            'target_sharpe': 0.68  # Your original target
        }
    }
    
    results = {}
    
    for regime_name, config in regimes.items():
        print(f"\nAnalyzing {regime_name}...")
        results[regime_name] = analyze_regime_sharpe(regime_name, config)
    
    # Create results table and visualization
    create_sharpe_table_and_graph(results)
    
    return results


def analyze_regime_sharpe(regime_name, config):
    """Analyze Sharpe ratios for a specific regime"""
    
    # Generate data exactly like your original paper
    np.random.seed(42)  # Consistent seed for reproducibility
    ou = OrnsteinUhlenbeckGenerator(**config['generator_params'])
    
    if config['data_method'] == 'generate_switching_trend':
        prices = ou.generate_switching_trend(10000)  # Like your paper
    elif config['data_method'] == 'generate_reversion':
        prices = ou.generate_reversion(10000)
    elif config['data_method'] == 'generate_uptrend':
        prices = ou.generate_uptrend(10000, trend_rate=0.01)
    
    # Split data like your original work (80% train, 20% test)
    train_size = int(0.8 * len(prices))
    train_prices = prices[:train_size]
    test_prices = prices[train_size:]
    
    print(f"  Data: {len(prices)} total, {len(train_prices)} train, {len(test_prices)} test")
    
    # Prepare indicator data for compact network
    X, y, feature_names = prepare_indicator_data(train_prices)
    X_test, y_test, _ = prepare_indicator_data(test_prices)
    
    # Train compact network using NEW VERSION
    print("  Training compact network (NEW VERSION)...")
    network = CompactFinancialNetwork(n_indicators=len(feature_names))
    trainer = CompactNetworkTrainer(network, feature_names)
    
    losses, initial_sharpe, final_sharpe, improvement = trainer.train_like_paper(
        X, y, config['network_regime'], 
        train_layers='logic_feature',  # Most successful in your experiments
        epochs=800, lr=0.01
    )
    
    # Evaluate compact network on test set
    with torch.no_grad():
        positions = network(torch.FloatTensor(X_test))
        position_weights = positions[:, 0] - positions[:, 1]
        compact_returns = position_weights.numpy() * y_test
    
    compact_sharpe = calculate_sharpe_ratio(compact_returns)
    
    # Evaluate benchmark strategies on test set
    print("  Evaluating benchmarks...")
    
    # SMA Momentum
    mom_positions = sma_momentum_strategy(test_prices)
    if len(mom_positions) > 0:
        mom_returns = []
        for i in range(min(len(mom_positions), len(test_prices)-1)):
            if i+1 < len(test_prices):
                ret = (test_prices[i+1] - test_prices[i]) / test_prices[i]
                pos_weight = mom_positions[i][0] - mom_positions[i][1]
                mom_returns.append(pos_weight * ret)
        mom_sharpe = calculate_sharpe_ratio(np.array(mom_returns)) if mom_returns else 0.0
    else:
        mom_sharpe = 0.0
    
    # SMA Reversion
    rev_positions = sma_reversion_strategy(test_prices)
    if len(rev_positions) > 0:
        rev_returns = []
        for i in range(min(len(rev_positions), len(test_prices)-1)):
            if i+1 < len(test_prices):
                ret = (test_prices[i+1] - test_prices[i]) / test_prices[i]
                pos_weight = rev_positions[i][0] - rev_positions[i][1]
                rev_returns.append(pos_weight * ret)
        rev_sharpe = calculate_sharpe_ratio(np.array(rev_returns)) if rev_returns else 0.0
    else:
        rev_sharpe = 0.0
    
    # Buy & Hold
    bh_returns = []
    for i in range(len(test_prices)-1):
        ret = (test_prices[i+1] - test_prices[i]) / test_prices[i]
        bh_returns.append(ret)
    bh_sharpe = calculate_sharpe_ratio(np.array(bh_returns))
    
    print(f"  Results: Compact={compact_sharpe:.3f}, SMA-MOM={mom_sharpe:.3f}, SMA-REV={rev_sharpe:.3f}, B&H={bh_sharpe:.3f}")
    
    return {
        'compact_network': compact_sharpe,
        'sma_momentum': mom_sharpe,
        'sma_reversion': rev_sharpe,
        'buy_hold': bh_sharpe,
        'target_sharpe': config['target_sharpe'],
        'top_indicators': network.get_indicator_importance(feature_names)['top_3'],
        'param_count': network.param_count,
        'train_improvement': improvement
    }


def create_sharpe_table_and_graph(results):
    """Create the focused Sharpe ratio table and graph you requested"""
    
    # Prepare data for table
    table_data = []
    
    strategies = ['Compact Network', 'SMA Momentum', 'SMA Reversion', 'Buy & Hold']
    strategy_keys = ['compact_network', 'sma_momentum', 'sma_reversion', 'buy_hold']
    
    for regime_name, result in results.items():
        row = {
            'Market Regime': regime_name.replace('_', ' '),
            'Compact Network': f"{result['compact_network']:.3f}",
            'SMA Momentum': f"{result['sma_momentum']:.3f}",
            'SMA Reversion': f"{result['sma_reversion']:.3f}",
            'Buy & Hold': f"{result['buy_hold']:.3f}",
            'Target': f"{result['target_sharpe']:.3f}",
            'Top Indicators': ', '.join(result['top_indicators'][:2]),  # Top 2 for space
            'Parameters': result['param_count'],
            'Train Improvement': f"{result['train_improvement']:+.3f}"
        }
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    df.to_csv('sharpe_comparison_results.csv', index=False)
    print(f"\n📊 Results saved to 'sharpe_comparison_results.csv'")
    
    # Create side-by-side table and graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left side: Table
    ax1.axis('tight')
    ax1.axis('off')
    
    # Create table with just the core metrics
    table_for_display = df[['Market Regime', 'Compact Network', 'SMA Momentum', 'SMA Reversion', 'Buy & Hold', 'Target']].copy()
    
    table = ax1.table(cellText=table_for_display.values, 
                     colLabels=table_for_display.columns,
                     cellLoc='center', 
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Color code the best performer in each row
    for i in range(len(table_for_display)):
        values = [float(table_for_display.iloc[i][col]) for col in ['Compact Network', 'SMA Momentum', 'SMA Reversion', 'Buy & Hold']]
        best_idx = np.argmax(values)
        
        # Color the best performer green
        table[(i+1, best_idx+1)].set_facecolor('#90EE90')
        
        # Color compact network performance based on target
        compact_val = float(table_for_display.iloc[i]['Compact Network'])
        target_val = float(table_for_display.iloc[i]['Target'])
        
        if compact_val >= target_val * 0.8:  # Within 80% of target
            table[(i+1, 1)].set_facecolor('#ADD8E6')  # Light blue for good performance
    
    ax1.set_title('Sharpe Ratio Comparison Across Market Regimes\n(Green = Best Performer, Blue = Near Target)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Right side: Bar graph
    regimes = [row['Market Regime'] for row in table_data]
    x = np.arange(len(regimes))
    width = 0.2
    
    compact_values = [float(row['Compact Network']) for row in table_data]
    momentum_values = [float(row['SMA Momentum']) for row in table_data]
    reversion_values = [float(row['SMA Reversion']) for row in table_data]
    buyhold_values = [float(row['Buy & Hold']) for row in table_data]
    target_values = [float(row['Target']) for row in table_data]
    
    bars1 = ax2.bar(x - 1.5*width, compact_values, width, label='Compact Network', alpha=0.8, color='#1f77b4')
    bars2 = ax2.bar(x - 0.5*width, momentum_values, width, label='SMA Momentum', alpha=0.8, color='#ff7f0e')
    bars3 = ax2.bar(x + 0.5*width, reversion_values, width, label='SMA Reversion', alpha=0.8, color='#2ca02c')
    bars4 = ax2.bar(x + 1.5*width, buyhold_values, width, label='Buy & Hold', alpha=0.8, color='#d62728')
    
    # Add target line
    ax2.plot(x, target_values, 'ro--', linewidth=2, markersize=8, label='Target Performance')
    
    ax2.set_xlabel('Market Regime')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Sharpe Ratio Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regimes, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    add_value_labels(bars4)
    
    plt.tight_layout()
    plt.savefig('sharpe_comparison_table_and_graph.png', dpi=300, bbox_inches='tight')
    print("📊 Sharpe comparison table and graph saved as 'sharpe_comparison_table_and_graph.png'")
    plt.close()
    
    # Print summary analysis
    print("\n" + "=" * 60)
    print("SHARPE RATIO ANALYSIS SUMMARY")
    print("=" * 60)
    
    total_regimes = len(results)
    compact_wins = 0
    near_target_count = 0
    
    for regime_name, result in results.items():
        compact_sharpe = result['compact_network']
        target_sharpe = result['target_sharpe']
        benchmark_sharpes = [result['sma_momentum'], result['sma_reversion'], result['buy_hold']]
        
        # Check if compact network wins
        if compact_sharpe > max(benchmark_sharpes):
            compact_wins += 1
        
        # Check if near target
        if compact_sharpe >= target_sharpe * 0.8:
            near_target_count += 1
        
        print(f"{regime_name:15}: Compact {compact_sharpe:6.3f} vs Target {target_sharpe:6.3f} "
              f"({'✓' if compact_sharpe >= target_sharpe * 0.8 else '✗'}) "
              f"Train Δ{result['train_improvement']:+.3f}")
    
    # Overall parameter efficiency
    param_count = list(results.values())[0]['param_count']
    avg_sharpe = np.mean([result['compact_network'] for result in results.values()])
    
    print(f"\n🏆 Compact network outperformed all benchmarks: {compact_wins}/{total_regimes} regimes")
    print(f"🎯 Near target performance (≥80%): {near_target_count}/{total_regimes} regimes")
    print(f"⚡ Parameter efficiency: {avg_sharpe/param_count:.6f} Sharpe per parameter")
    print(f"🎯 Network parameter count: {param_count} (ultra-compact!)")
    
    # Regime-specific insights
    print(f"\n📊 REGIME-SPECIFIC INSIGHTS:")
    for regime_name, result in results.items():
        indicators = result['top_indicators']
        improvement = result['train_improvement']
        print(f"{regime_name:15}: Key indicators = {indicators}, Training Δ = {improvement:+.3f}")
    
    return df


if __name__ == "__main__":
    # Test the compact network first
    print("Testing NEW VERSION compact network...")
    from compact_network import test_compact_network
    
    # Quick test
    test_results = test_compact_network()
    
    print("\n" + "="*60)
    
    # Run full comparison
    final_results = run_sharpe_comparison()
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*60)
    print("✅ NEW VERSION compact network tested")
    print("✅ Sharpe ratio comparison completed")
    print("✅ Results table and graph created")
    print("✅ Performance analysis generated")