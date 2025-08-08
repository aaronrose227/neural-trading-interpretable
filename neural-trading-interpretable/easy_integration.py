# easy_integration.py
"""
Simple example showing how to integrate the enhanced system into your existing code.
You can literally copy-paste this into your experiments.py and run it.
"""

# Your existing imports (keep these exactly the same)
import numpy as np
import torch
from data_generation import OrnsteinUhlenbeckGenerator
from benchmark_strategies import sma_momentum_strategy, sma_reversion_strategy, buy_and_hold_strategy
from evaluation import calculate_returns, calculate_sharpe_ratio, prepare_data
import pandas as pd

# Import your original classes
try:
    from neural_policy_trading import TradingNetwork
    from training import train_network
    ORIGINAL_AVAILABLE = True
except ImportError:
    print("Warning: Original TradingNetwork not found. Will only run enhanced experiments.")
    ORIGINAL_AVAILABLE = False

# NEW: Import the enhanced components (add these)
from enhanced_neural_policy_trading import (
    EnhancedTradingNetwork, 
    RegimeDetector, 
    train_enhanced_network,
    run_enhanced_experiment
)


def compare_original_vs_enhanced():
    """
    Direct comparison: Your original approach vs enhanced approach
    """
    if not ORIGINAL_AVAILABLE:
        print("❌ Original TradingNetwork not available. Running enhanced only.")
        return run_enhanced_only_experiment()
    
    print("ORIGINAL vs ENHANCED COMPARISON")
    print("=" * 50)
    
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate your switching trend data (the most interesting case)
    ou_switch = OrnsteinUhlenbeckGenerator(theta=7.5, mu=50, sigma=10)
    prices = ou_switch.generate_switching_trend(10000)
    
    train_prices = prices[:8000]
    test_prices = prices[8000:]
    
    print(f"Generated switching trend data: {len(prices)} points")
    print(f"Training on first 8000, testing on last 2000")
    
    # ===== ORIGINAL APPROACH (your exact code) =====
    print("\n1. ORIGINAL KRAUSE-CALLIESS")
    print("-" * 30)
    
    # Your original network
    original_network = TradingNetwork()
    original_network._initialize_momentum_strategy()  # Your initialization
    
    print("Training original network...")
    train_network(
        original_network,
        train_prices,
        epochs=800,
        lr=0.001,
        train_layers='logic_feature',  # Your paper setup
        verbose=True
    )
    
    # Evaluate original
    X_test, y_test, current_test_prices = prepare_data(test_prices, original_network.lookback_long)
    X_test_tensor = torch.FloatTensor(X_test)
    
    with torch.no_grad():
        original_positions = original_network(X_test_tensor)
    
    original_returns = calculate_returns(test_prices[original_network.lookback_long:], 
                                       original_positions.numpy())
    original_sharpe = calculate_sharpe_ratio(original_returns)
    
    # Get interpretation
    original_interp = original_network.interpret_weights()
    
    print(f"Original Results:")
    print(f"  Test Sharpe: {original_sharpe:.3f}")
    print(f"  Strategy: {original_interp['logic_layer']['long']}")
    
    # ===== ENHANCED APPROACH =====
    print("\n2. ENHANCED WITH REGIME DETECTION")
    print("-" * 30)
    
    # Reset random seeds for fair comparison
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Use our enhanced experiment runner
    enhanced_results = run_enhanced_experiment(
        "Switching", 
        prices, 
        lambda net: net._initialize_momentum_strategy(),
        train_layers='enhanced'
    )
    
    # ===== COMPARISON =====
    print("\n3. SIDE-BY-SIDE COMPARISON")
    print("-" * 30)
    
    improvement = enhanced_results['enhanced_sharpe'] - original_sharpe
    
    print(f"Original Sharpe:     {original_sharpe:.3f}")
    print(f"Enhanced Sharpe:     {enhanced_results['enhanced_sharpe']:.3f}")
    print(f"Improvement:         {improvement:+.3f} ({improvement/abs(original_sharpe)*100:+.1f}%)")
    
    # Show what was learned
    enhanced_interp = enhanced_results['interpretation']
    
    print(f"\nOriginal Strategy:   {original_interp['logic_layer']['long']}")
    print(f"Enhanced Strategy:   {enhanced_interp.get('base_strategy', 'Unknown')}")
    
    if enhanced_interp.get('adaptive_periods'):
        short_p, long_p = enhanced_interp['adaptive_periods']
        print(f"Original Periods:    50/200 (fixed)")
        print(f"Enhanced Periods:    {short_p}/{long_p} (adaptive)")
    
    if enhanced_results['regime_detection']:
        print(f"Regime Detection:    ✓ Active")
    else:
        print(f"Regime Detection:    ✗ Failed (fallback to original)")
    
    # Benchmark comparison
    benchmarks = enhanced_results['benchmark_sharpes']
    print(f"\nBenchmark Comparison:")
    print(f"  SMA Momentum:      {benchmarks['SMA-MOM']:.3f}")
    print(f"  SMA Reversion:     {benchmarks['SMA-REV']:.3f}")
    print(f"  Buy & Hold:        {benchmarks['Buy & Hold']:.3f}")
    
    return {
        'original': original_sharpe,
        'enhanced': enhanced_results['enhanced_sharpe'],
        'improvement': improvement
    }


def run_enhanced_only_experiment():
    """
    Run enhanced experiment without original comparison
    """
    print("ENHANCED KRAUSE-CALLIESS EXPERIMENT")
    print("=" * 40)
    
    # Generate switching trend data
    np.random.seed(42)
    torch.manual_seed(42)
    
    ou_switch = OrnsteinUhlenbeckGenerator(theta=7.5, mu=50, sigma=10)
    prices = ou_switch.generate_switching_trend(10000)
    
    print(f"Generated switching trend data: {len(prices)} points")
    
    # Run enhanced experiment
    enhanced_results = run_enhanced_experiment(
        "Switching", 
        prices, 
        lambda net: net._initialize_momentum_strategy(),
        train_layers='enhanced'
    )
    
    print("\nENHANCED RESULTS:")
    print(f"Enhanced Sharpe:     {enhanced_results['enhanced_sharpe']:.3f}")
    
    enhanced_interp = enhanced_results['interpretation']
    print(f"Enhanced Strategy:   {enhanced_interp.get('base_strategy', 'Unknown')}")
    
    if enhanced_interp.get('adaptive_periods'):
        short_p, long_p = enhanced_interp['adaptive_periods']
        print(f"Adaptive Periods:    {short_p}/{long_p} (vs 50/200 standard)")
    
    if enhanced_results['regime_detection']:
        print(f"Regime Detection:    ✓ Active")
    
    # Benchmark comparison
    benchmarks = enhanced_results['benchmark_sharpes']
    print(f"\nBenchmark Comparison:")
    print(f"  Enhanced Network:   {enhanced_results['enhanced_sharpe']:.3f}")
    print(f"  SMA Momentum:       {benchmarks['SMA-MOM']:.3f}")
    print(f"  SMA Reversion:      {benchmarks['SMA-REV']:.3f}")
    print(f"  Buy & Hold:         {benchmarks['Buy & Hold']:.3f}")
    
    best_benchmark = max(benchmarks.values())
    improvement = enhanced_results['enhanced_sharpe'] - best_benchmark
    print(f"\nImprovement over best benchmark: {improvement:+.3f}")
    
    return enhanced_results


def quick_test_on_your_data():
    """
    Quick test you can run to see if everything works
    """
    print("QUICK INTEGRATION TEST")
    print("=" * 30)
    
    # Generate small dataset for quick test
    ou = OrnsteinUhlenbeckGenerator(theta=5, mu=50, sigma=15)
    prices = ou.generate_switching_trend(2000)  # Smaller for quick test
    
    print("Testing regime detector...")
    detector = RegimeDetector()
    success = detector.train_regime_detector(prices[:1500])
    
    if success:
        probs = detector.predict_regime_probabilities(prices[:1500])
        regime_names = ['Low Vol', 'High Vol', 'Trending', 'Mean Rev']
        print(f"✓ Regime detection working: {regime_names[np.argmax(probs)]}")
    else:
        print("⚠ Regime detection failed, but that's okay - will fallback to original")
    
    print("\nTesting enhanced network...")
    network = EnhancedTradingNetwork(
        lookback_long=100,  # Smaller for quick test
        lookback_short=20,
        enable_adaptive_periods=True,
        enable_regime_detection=success
    )
    
    # Quick training
    losses = train_enhanced_network(
        network,
        prices[:1500],
        regime_detector=detector if success else None,
        epochs=100,  # Just 100 epochs for quick test
        verbose=True
    )
    
    print(f"✓ Training completed, final loss: {losses[-1]:.4f}")
    
    # Show adaptive periods
    adaptive_periods = network.get_adaptive_periods()
    print(f"✓ Adaptive periods: {adaptive_periods[0]}/{adaptive_periods[1]}")
    
    interpretation = network.interpret_weights()
    print(f"✓ Strategy learned: {interpretation.get('base_strategy', 'Unknown')}")
    
    print("\n✅ Integration test successful! Ready for full experiments.")


def run_single_enhanced_experiment(data_type="switching"):
    """
    Run a single enhanced experiment (good for debugging)
    """
    print(f"SINGLE ENHANCED EXPERIMENT: {data_type.upper()}")
    print("=" * 40)
    
    # Generate data based on type
    if data_type == "switching":
        ou = OrnsteinUhlenbeckGenerator(theta=7.5, mu=50, sigma=10)
        prices = ou.generate_switching_trend(10000)
        init_func = lambda net: net._initialize_momentum_strategy()
    elif data_type == "uptrend":
        ou = OrnsteinUhlenbeckGenerator(theta=2, mu=50, sigma=20)
        prices = ou.generate_uptrend(10000)
        init_func = lambda net: net._initialize_buy_and_hold()
    else:  # reversion
        ou = OrnsteinUhlenbeckGenerator(theta=20, mu=50, sigma=50)
        prices = ou.generate_reversion(10000)
        init_func = lambda net: net._initialize_reversion_strategy()
    
    # Run enhanced experiment
    results = run_enhanced_experiment(data_type, prices, init_func)
    
    return results


if __name__ == "__main__":
    print("ENHANCED KRAUSE-CALLIESS INTEGRATION")
    print("=" * 50)
    
    if not ORIGINAL_AVAILABLE:
        print("⚠️  Original TradingNetwork not found.")
        print("Running enhanced experiments only (this is fine for testing!)")
        print("\nAvailable options:")
        print("1. Quick integration test")
        print("3. Enhanced experiment only")
        print("4. Full comprehensive enhanced experiments")
        
        choice = input("\nEnter choice (1, 3, or 4): ").strip()
        
        if choice == "1":
            quick_test_on_your_data()
        elif choice == "3":
            data_type = input("Data type (switching/uptrend/reversion): ").strip().lower()
            if data_type not in ["switching", "uptrend", "reversion"]:
                data_type = "switching"
            results = run_single_enhanced_experiment(data_type)
        elif choice == "4":
            from enhanced_neural_policy_trading import run_comprehensive_enhanced_experiments
            results = run_comprehensive_enhanced_experiments()
        else:
            print("Running quick test by default...")
            quick_test_on_your_data()
    else:
        print("Choose what to run:")
        print("1. Quick integration test (recommended first)")
        print("2. Original vs Enhanced comparison") 
        print("3. Single enhanced experiment")
        print("4. Full comprehensive experiments")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            quick_test_on_your_data()
        elif choice == "2":
            results = compare_original_vs_enhanced()
        elif choice == "3":
            data_type = input("Data type (switching/uptrend/reversion): ").strip().lower()
            if data_type not in ["switching", "uptrend", "reversion"]:
                data_type = "switching"
            results = run_single_enhanced_experiment(data_type)
        elif choice == "4":
            from enhanced_neural_policy_trading import run_comprehensive_enhanced_experiments
            results = run_comprehensive_enhanced_experiments()
        else:
            print("Running quick test by default...")
            quick_test_on_your_data()
    
    print("\n" + "=" * 50)
    print("Integration complete! 🚀")
    print("Next steps:")
    print("- Try different data types")
    print("- Experiment with train_layers='enhanced'")
    print("- Analyze the adaptive periods learned")
    print("- Compare regime detection insights")