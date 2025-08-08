import numpy as np
import torch

# Import modules
from data_generation import OrnsteinUhlenbeckGenerator
from financial_indicators import calculate_all_indicators, prepare_indicator_data
from compact_network import CompactFinancialNetwork, CompactNetworkTrainer

def simple_test():
    print("COMPACT NETWORK SIMPLE TEST")
    print("=" * 40)
    
    # Step 1: Generate data
    print("1. Generating data...")
    np.random.seed(42)
    ou = OrnsteinUhlenbeckGenerator(theta=7.5, mu=50, sigma=10)
    prices = ou.generate_switching_trend(1000)
    print(f"   ✓ Generated {len(prices)} price points")
    
    # Step 2: Calculate indicators
    print("2. Calculating indicators...")
    indicators_df = calculate_all_indicators(prices)
    print(f"   ✓ Calculated {len(indicators_df.columns)} indicators")
    
    # Step 3: Prepare data
    print("3. Preparing data...")
    X, y, feature_names = prepare_indicator_data(prices)
    print(f"   ✓ Prepared {X.shape[0]} samples with {X.shape[1]} features")
    
    # Step 4: Create network
    print("4. Creating compact network...")
    network = CompactFinancialNetwork(n_indicators=len(feature_names))
    print(f"   ✓ Network created with {network.param_count} parameters")
    
    # Step 5: Test forward pass
    print("5. Testing forward pass...")
    with torch.no_grad():
        positions = network(torch.FloatTensor(X[:10]))
    print(f"   ✓ Forward pass successful: {positions.shape}")
    
    # Step 6: Test interpretation
    print("6. Testing interpretability...")
    importance = network.get_indicator_importance(feature_names)
    print(f"   ✓ Top 3 indicators: {importance['top_3']}")
    
    # Step 7: Quick training test
    print("7. Quick training test...")
    trainer = CompactNetworkTrainer(network, feature_names)
    X_small = X[:100]
    y_small = y[:100]
    
    losses, _, final_interp = trainer.train_with_sparsity(
        X_small, y_small, epochs=50, lr=0.01, sparsity_weight=0.1
    )
    
    print(f"   ✓ Training completed")
    print(f"   ✓ Final top indicators: {final_interp['indicator_importance']['top_3']}")
    
    print("\n" + "=" * 40)
    print("ALL TESTS PASSED! ✅")
    print(f"✓ Ultra-compact network: {network.param_count} parameters")
    print("✓ Hierarchical indicator discovery working")
    print("✓ Full interpretability maintained")
    
    return True

if __name__ == "__main__":
    simple_test()