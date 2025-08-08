# optimized_enhanced_trading.py
"""
Optimized version to beat the 2.45 SMA-MOM benchmark
Key optimizations:
1. More aggressive adaptive period ranges
2. Enhanced regime-specific strategy selection
3. Better loss function targeting Sharpe ratio directly
4. Momentum-focused initialization for switching data
"""

import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler


class OptimizedTradingNetwork(nn.Module):
    """
    Optimized version designed to beat benchmarks while maintaining interpretability
    """
    
    def __init__(self, lookback_long=200, lookback_short=50, beta=10.0, 
                 enable_adaptive_periods=True, enable_regime_detection=True,
                 aggressive_adaptation=True):
        super(OptimizedTradingNetwork, self).__init__()
        
        self.lookback_long = lookback_long
        self.lookback_short = lookback_short
        self.beta = beta  # Increased beta for sharper decisions
        self.enable_adaptive_periods = enable_adaptive_periods
        self.enable_regime_detection = enable_regime_detection
        self.aggressive_adaptation = aggressive_adaptation
        
        # Core architecture (same as before)
        self.w_0_1 = nn.Parameter(torch.zeros(lookback_long))
        self.w_0_2 = nn.Parameter(torch.zeros(lookback_long))
        self.feature_layer = nn.Linear(2, 2, bias=True)
        self.logic_layer = nn.Linear(2, 3, bias=True)
        
        # OPTIMIZED: More aggressive adaptive periods
        if enable_adaptive_periods:
            if aggressive_adaptation:
                # Allow wider range: 10-80 for short, 80-350 for long
                self.short_period_param = nn.Parameter(torch.tensor(-0.5))  # Start at ~37
                self.long_period_param = nn.Parameter(torch.tensor(-0.3))   # Start at ~150
                self.period_learning_rate = 0.1  # Faster period adaptation
            else:
                self.short_period_param = nn.Parameter(torch.tensor(0.0))
                self.long_period_param = nn.Parameter(torch.tensor(0.0))
        
        # OPTIMIZED: Enhanced regime detection with momentum bias
        if enable_regime_detection:
            # Regime processor with better capacity
            self.regime_processor = nn.Linear(4, 4, bias=True)  # More capacity
            
            # Separate momentum strategies for different regimes
            self.strong_momentum = nn.Linear(2, 3, bias=True)   # For trending regimes
            self.weak_momentum = nn.Linear(2, 3, bias=True)    # For ranging regimes
            self.reversion_strategy = nn.Linear(2, 3, bias=True)
            
            # Smarter regime-based mixing
            self.regime_selector = nn.Linear(4, 4, bias=True)  # Select best strategy per regime
        
        # Initialize everything
        self._initialize_optimized_network()
    
    def _initialize_optimized_network(self):
        """Initialize with momentum bias for switching data"""
        # Standard initialization first
        self._initialize_ma_weights()
        self._initialize_momentum_strategy()
        
        # OPTIMIZED: Momentum-biased regime initialization
        if self.enable_regime_detection:
            with torch.no_grad():
                # Regime processor: Small but non-zero weights
                self.regime_processor.weight.data.normal_(0, 0.05)
                self.regime_processor.bias.data.zero_()
                
                # Strong momentum: Even more aggressive than standard
                self.strong_momentum.weight[0, 0] = 2.0   # Long bias
                self.strong_momentum.weight[0, 1] = -2.0
                self.strong_momentum.bias[0] = 0.5
                
                self.strong_momentum.weight[1, 0] = -2.0  # Short bias  
                self.strong_momentum.weight[1, 1] = 2.0
                self.strong_momentum.bias[1] = 0.5
                
                self.strong_momentum.weight[2, 0] = -2.0  # Discourage neutral
                self.strong_momentum.weight[2, 1] = -2.0
                self.strong_momentum.bias[2] = -1.0
                
                # Weak momentum: Standard momentum
                self.weak_momentum.weight.data.copy_(self.logic_layer.weight.data)
                self.weak_momentum.bias.data.copy_(self.logic_layer.bias.data)
                
                # Reversion: Keep as before but discourage it initially
                self.reversion_strategy.weight[0, 0] = -1.0
                self.reversion_strategy.weight[0, 1] = 1.0
                self.reversion_strategy.bias[0] = -0.5  # Slight bias against
                
                self.reversion_strategy.weight[1, 0] = 1.0
                self.reversion_strategy.weight[1, 1] = -1.0
                self.reversion_strategy.bias[1] = -0.5
                
                self.reversion_strategy.weight[2, 0] = -1.0
                self.reversion_strategy.weight[2, 1] = -1.0
                self.reversion_strategy.bias[2] = 1.5
                
                # Regime selector: Bias toward momentum strategies
                self.regime_selector.weight.data.normal_(0, 0.1)
                self.regime_selector.bias.data = torch.tensor([0.2, 0.2, -0.2, -0.2])  # Favor momentum
    
    def _initialize_ma_weights(self):
        """Standard MA initialization"""
        with torch.no_grad():
            self.w_0_1[:self.lookback_short] = 1.0 / self.lookback_short
            self.w_0_1[self.lookback_short:] = 0.0
            self.w_0_2[:] = 1.0 / self.lookback_long
    
    def _initialize_momentum_strategy(self):
        """Standard momentum initialization"""
        with torch.no_grad():
            self.feature_layer.weight[0, 0] = 1.0
            self.feature_layer.weight[0, 1] = -1.0
            self.feature_layer.bias[0] = 0.0
            
            self.feature_layer.weight[1, 0] = -1.0
            self.feature_layer.weight[1, 1] = 1.0
            self.feature_layer.bias[1] = 0.0
            
            self.logic_layer.weight[0, 0] = 1.0
            self.logic_layer.weight[0, 1] = -1.0
            self.logic_layer.bias[0] = 0.0
            
            self.logic_layer.weight[1, 0] = -1.0
            self.logic_layer.weight[1, 1] = 1.0
            self.logic_layer.bias[1] = 0.0
            
            self.logic_layer.weight[2, 0] = -1.0
            self.logic_layer.weight[2, 1] = -1.0
            self.logic_layer.bias[2] = 1.0
    
    def get_adaptive_periods(self):
        """Get adaptive periods with wider ranges"""
        if not self.enable_adaptive_periods:
            return self.lookback_short, self.lookback_long
        
        if self.aggressive_adaptation:
            # Wider ranges: 10-80 and 80-350
            short_base = 10 + 70 * torch.sigmoid(self.short_period_param).item()  # 10-80
            long_base = 80 + 270 * torch.sigmoid(self.long_period_param).item()   # 80-350
        else:
            short_base = self.lookback_short * (0.5 + 0.5 * torch.sigmoid(self.short_period_param).item())
            long_base = self.lookback_long * (0.5 + 0.5 * torch.sigmoid(self.long_period_param).item())
        
        short_period = max(5, min(90, int(short_base)))
        long_period = max(50, min(400, int(long_base)))
        
        return short_period, long_period
    
    def forward(self, x, regime_probs=None):
        """Optimized forward pass with better regime utilization"""
        # Input normalization
        x_norm = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
        
        # Adaptive moving averages
        if self.enable_adaptive_periods:
            short_p, long_p = self.get_adaptive_periods()
            
            # Dynamic weight computation
            w_short = torch.zeros_like(self.w_0_1)
            w_short[-short_p:] = 1.0 / short_p
            
            w_long = torch.full_like(self.w_0_2, 1.0 / long_p)
            
            ma_short = torch.sum(x_norm * w_short, dim=1)
            ma_long = torch.sum(x_norm * w_long, dim=1)
        else:
            ma_short = torch.sum(x_norm * self.w_0_1, dim=1)
            ma_long = torch.sum(x_norm * self.w_0_2, dim=1)
        
        # Feature processing
        ma_features = torch.stack([ma_short, ma_long], dim=1)
        feature_output = torch.sigmoid(self.feature_layer(ma_features))
        
        # OPTIMIZED: Enhanced regime-based strategy selection
        if self.enable_regime_detection and regime_probs is not None:
            regime_tensor = torch.FloatTensor(regime_probs).unsqueeze(0)
            if regime_tensor.shape[0] != x.shape[0]:
                regime_tensor = regime_tensor.repeat(x.shape[0], 1)
            
            # Enhanced regime processing
            regime_features = torch.sigmoid(self.regime_processor(regime_tensor))
            
            # Multiple strategy outputs
            strong_momentum_out = torch.sigmoid(self.strong_momentum(feature_output))
            weak_momentum_out = torch.sigmoid(self.weak_momentum(feature_output))
            reversion_out = torch.sigmoid(self.reversion_strategy(feature_output))
            base_logic_out = torch.sigmoid(self.logic_layer(feature_output))
            
            # Smart regime-based selection
            regime_weights = torch.softmax(self.regime_selector(regime_tensor), dim=1)
            
            # Combine strategies based on regime detection
            # Regime weights: [strong_mom, weak_mom, reversion, base]
            combined_logic = (
                regime_weights[:, 0:1] * strong_momentum_out +
                regime_weights[:, 1:2] * weak_momentum_out +
                regime_weights[:, 2:3] * reversion_out +
                regime_weights[:, 3:4] * base_logic_out
            )
            
            logic_output = combined_logic
        else:
            logic_output = torch.sigmoid(self.logic_layer(feature_output))
        
        # Enhanced softargmax with higher temperature for sharper decisions
        positions = torch.softmax(self.beta * logic_output, dim=1)
        
        return positions
    
    def interpret_weights(self):
        """Enhanced interpretation"""
        base_interp = self._get_base_interpretation()
        
        enhanced_interp = {
            'base_strategy': base_interp,
            'adaptive_periods': self.get_adaptive_periods() if self.enable_adaptive_periods else None,
            'regime_enabled': self.enable_regime_detection,
            'aggressive_adaptation': self.aggressive_adaptation,
        }
        
        if self.enable_regime_detection:
            enhanced_interp['regime_strategies'] = self._interpret_regime_strategies()
        
        return enhanced_interp
    
    def _get_base_interpretation(self):
        """Same as before"""
        w1 = self.logic_layer.weight[0, 0].item()
        w2 = self.logic_layer.weight[0, 1].item()
        
        if w1 > 0.5 and w2 < -0.5:
            return "Momentum: Long when short_MA > long_MA"
        elif w1 < -0.5 and w2 > 0.5:
            return "Reversion: Long when long_MA > short_MA"
        else:
            return f"Custom rule: {w1:.2f}*z1 + {w2:.2f}*z2"
    
    def _interpret_regime_strategies(self):
        """Interpret the different regime strategies"""
        strong_mom_bias = self.strong_momentum.bias[0].item()
        weak_mom_bias = self.weak_momentum.bias[0].item()
        reversion_bias = self.reversion_strategy.bias[0].item()
        
        return {
            'strong_momentum_bias': strong_mom_bias,
            'weak_momentum_bias': weak_mom_bias,
            'reversion_bias': reversion_bias,
            'regime_selection': 'Active' if torch.std(self.regime_selector.weight) > 0.05 else 'Minimal'
        }


def optimized_sharpe_loss(strategy_returns, target_sharpe=2.5):
    """
    Loss function that directly targets a specific Sharpe ratio
    Encourages the network to beat benchmarks
    """
    mean_return = torch.mean(strategy_returns)
    std_return = torch.std(strategy_returns) + 1e-8
    
    # Current Sharpe ratio (annualized)
    current_sharpe = (mean_return * 252) / (std_return * torch.sqrt(torch.tensor(252.0)))
    
    # Loss that encourages beating target Sharpe
    sharpe_loss = -current_sharpe  # Maximize Sharpe
    
    # Additional penalty for not reaching target
    if current_sharpe < target_sharpe:
        penalty = (target_sharpe - current_sharpe) ** 2
        sharpe_loss = sharpe_loss + penalty
    
    return sharpe_loss


def train_optimized_network(network, prices, regime_detector=None, epochs=1000, 
                           lr=0.001, target_sharpe=2.5, verbose=True):
    """
    Optimized training to beat benchmarks
    """
    from evaluation import prepare_data, calculate_sharpe_ratio
    
    # Data preparation
    X, y, current_prices = prepare_data(prices, network.lookback_long)
    X_tensor = torch.FloatTensor(X)
    
    # All parameters trainable for maximum optimization
    params_to_train = list(network.parameters())
    optimizer = torch.optim.Adam(params_to_train, lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, patience=100
    )
    
    # Get regime information
    regime_probs = None
    if regime_detector and regime_detector.is_fitted:
        regime_probs = regime_detector.predict_regime_probabilities(prices)
        if verbose:
            regime_names = ['Low Vol', 'High Vol', 'Trending', 'Mean Rev']
            regime_str = ', '.join([f"{name}: {prob:.2f}" 
                                  for name, prob in zip(regime_names, regime_probs)])
            print(f"    Target regime: {regime_str}")
    
    # Track performance
    initial_sharpe = _calculate_performance(network, X_tensor, y, current_prices, regime_probs)
    best_sharpe = initial_sharpe
    best_state = network.state_dict().copy()
    
    if verbose:
        adaptive_periods = network.get_adaptive_periods()
        print(f"    Initial: Sharpe {initial_sharpe:.3f}, Periods: {adaptive_periods}, Target: {target_sharpe:.2f}")
    
    # Optimized training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        if regime_probs is not None and network.enable_regime_detection:
            positions = network(X_tensor, regime_probs)
        else:
            positions = network(X_tensor)
        
        # Calculate returns
        returns = (y - current_prices) / current_prices
        returns_tensor = torch.FloatTensor(returns)
        position_weights = positions[:, 0] - positions[:, 1]
        strategy_returns = position_weights * returns_tensor
        
        # Optimized loss targeting specific Sharpe ratio
        loss = optimized_sharpe_loss(strategy_returns, target_sharpe)
        
        # Regularization for adaptive periods (keep them reasonable)
        if network.enable_adaptive_periods:
            short_p, long_p = network.get_adaptive_periods()
            period_reg = 0.001 * (
                max(0, 5 - short_p) + max(0, short_p - 90) +
                max(0, 50 - long_p) + max(0, long_p - 400)
            )
            loss = loss + period_reg
        
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
        optimizer.step()
        
        # Track best performance
        current_sharpe = calculate_sharpe_ratio(strategy_returns.detach().numpy())
        if current_sharpe > best_sharpe:
            best_sharpe = current_sharpe
            best_state = network.state_dict().copy()
        
        scheduler.step(current_sharpe)
        
        # Progress reporting
        if verbose and epoch % 200 == 0:
            adaptive_periods = network.get_adaptive_periods()
            print(f"    Epoch {epoch}: Sharpe {current_sharpe:.3f} (Best: {best_sharpe:.3f}), Periods: {adaptive_periods}")
    
    # Load best state
    network.load_state_dict(best_state)
    
    if verbose:
        final_sharpe = calculate_sharpe_ratio(
            _calculate_strategy_returns(network, X_tensor, y, current_prices, regime_probs)
        )
        improvement = final_sharpe - initial_sharpe
        target_achieved = "✓" if final_sharpe >= target_sharpe else "✗"
        print(f"    Final: Sharpe {final_sharpe:.3f} ({improvement:+.3f}) Target: {target_achieved}")
    
    return network


def _calculate_performance(network, X_tensor, y, current_prices, regime_probs):
    """Calculate current Sharpe performance"""
    from evaluation import calculate_sharpe_ratio
    
    with torch.no_grad():
        if regime_probs is not None and network.enable_regime_detection:
            positions = network(X_tensor, regime_probs)
        else:
            positions = network(X_tensor)
        returns = (y - current_prices) / current_prices
        position_weights = positions[:, 0] - positions[:, 1]
        strategy_returns = position_weights * torch.FloatTensor(returns)
        return calculate_sharpe_ratio(strategy_returns.numpy())


def _calculate_strategy_returns(network, X_tensor, y, current_prices, regime_probs):
    """Calculate strategy returns"""
    with torch.no_grad():
        if regime_probs is not None and network.enable_regime_detection:
            positions = network(X_tensor, regime_probs)
        else:
            positions = network(X_tensor)
        returns = (y - current_prices) / current_prices
        position_weights = positions[:, 0] - positions[:, 1]
        strategy_returns = position_weights * torch.FloatTensor(returns)
        return strategy_returns.numpy()


def run_optimized_experiment(data_name, prices, initialization_func, target_sharpe=2.5):
    """
    Run optimized experiment targeting specific benchmark performance
    """
    from enhanced_neural_policy_trading import RegimeDetector
    from evaluation import prepare_data, calculate_returns, calculate_sharpe_ratio
    from benchmark_strategies import sma_momentum_strategy, sma_reversion_strategy, buy_and_hold_strategy
    
    print(f"\n🎯 OPTIMIZED EXPERIMENT: {data_name.upper()}")
    print(f"Target: Beat {target_sharpe:.2f} Sharpe benchmark")
    print("-" * 50)
    
    train_prices = prices[:8000]
    test_prices = prices[8000:]
    
    # Step 1: Enhanced regime detection
    regime_detector = RegimeDetector()
    regime_trained = regime_detector.train_regime_detector(train_prices)
    
    if regime_trained:
        regime_probs = regime_detector.predict_regime_probabilities(train_prices)
        regime_names = ['Low Vol', 'High Vol', 'Trending', 'Mean Rev']
        dominant_regime = regime_names[np.argmax(regime_probs)]
        print(f"Training regime: {dominant_regime} ({regime_probs[np.argmax(regime_probs)]:.2f})")
    else:
        regime_probs = None
        print("Regime detection failed, using aggressive adaptation only")
    
    # Step 2: Initialize optimized network
    network = OptimizedTradingNetwork(
        enable_adaptive_periods=True,
        enable_regime_detection=regime_trained,
        aggressive_adaptation=True,
        beta=15.0  # Sharper decisions
    )
    initialization_func(network)
    print("✓ Optimized network initialized")
    
    # Step 3: Optimized training
    print("🚀 Training optimized network...")
    network = train_optimized_network(
        network,
        train_prices,
        regime_detector=regime_detector if regime_trained else None,
        epochs=1200,  # More epochs
        lr=0.0005,    # Lower learning rate for stability
        target_sharpe=target_sharpe,
        verbose=True
    )
    
    # Step 4: Evaluation
    print("📊 Evaluating optimized performance...")
    test_regime_probs = None
    if regime_trained:
        test_regime_probs = regime_detector.predict_regime_probabilities(test_prices)
    
    X_test, y_test, current_test_prices = prepare_data(test_prices, network.lookback_long)
    X_test_tensor = torch.FloatTensor(X_test)
    
    with torch.no_grad():
        if test_regime_probs is not None and network.enable_regime_detection:
            test_positions = network(X_test_tensor, test_regime_probs)
        else:
            test_positions = network(X_test_tensor)
    
    test_returns = calculate_returns(test_prices[network.lookback_long:], test_positions.numpy())
    optimized_sharpe = calculate_sharpe_ratio(test_returns)
    
    # Benchmarks
    pos_mom = sma_momentum_strategy(test_prices)
    ret_mom = calculate_returns(test_prices[200:], pos_mom)
    sharpe_mom = calculate_sharpe_ratio(ret_mom)
    
    pos_rev = sma_reversion_strategy(test_prices)  
    ret_rev = calculate_returns(test_prices[200:], pos_rev)
    sharpe_rev = calculate_sharpe_ratio(ret_rev)
    
    pos_bh = buy_and_hold_strategy(test_prices)
    ret_bh = calculate_returns(test_prices[200:], pos_bh)
    sharpe_bh = calculate_sharpe_ratio(ret_bh)
    
    # Results
    target_achieved = optimized_sharpe >= target_sharpe
    improvement_vs_target = optimized_sharpe - target_sharpe
    
    print(f"\n🏆 OPTIMIZED RESULTS:")
    print(f"  Optimized Network: {optimized_sharpe:.2f} {'✓' if target_achieved else '✗'}")
    print(f"  Target (SMA-MOM):  {target_sharpe:.2f}")
    print(f"  Gap to target:     {improvement_vs_target:+.2f}")
    print(f"  SMA-REV:          {sharpe_rev:.2f}")
    print(f"  Buy & Hold:       {sharpe_bh:.2f}")
    
    # Interpretation
    interpretation = network.interpret_weights()
    print(f"\n🧠 LEARNED OPTIMIZATIONS:")
    print(f"  Strategy: {interpretation.get('base_strategy', 'Unknown')}")
    if interpretation.get('adaptive_periods'):
        short_p, long_p = interpretation['adaptive_periods']
        print(f"  Optimized periods: {short_p}/{long_p} (vs 50/200 standard)")
    
    if interpretation.get('regime_strategies'):
        rs = interpretation['regime_strategies']
        print(f"  Strong momentum bias: {rs['strong_momentum_bias']:.2f}")
        print(f"  Regime selection: {rs['regime_selection']}")
    
    return {
        'optimized_sharpe': optimized_sharpe,
        'target_achieved': target_achieved,
        'improvement_vs_target': improvement_vs_target,
        'benchmarks': {
            'SMA-MOM': sharpe_mom,
            'SMA-REV': sharpe_rev,
            'Buy & Hold': sharpe_bh
        },
        'interpretation': interpretation
    }


if __name__ == "__main__":
    # Test the optimized system
    from data_generation import OrnsteinUhlenbeckGenerator
    
    print("🎯 OPTIMIZED KRAUSE-CALLIESS - BEAT THE BENCHMARKS")
    print("=" * 60)
    
    # Generate switching trend data
    np.random.seed(42)
    torch.manual_seed(42)
    
    ou_switch = OrnsteinUhlenbeckGenerator(theta=7.5, mu=50, sigma=10)
    prices = ou_switch.generate_switching_trend(10000)
    
    # Run optimized experiment targeting 2.45 Sharpe
    results = run_optimized_experiment(
        "Switching", 
        prices, 
        lambda net: net._initialize_momentum_strategy(),
        target_sharpe=2.45
    )
    
    if results['target_achieved']:
        print(f"\n🎉 SUCCESS! Achieved {results['optimized_sharpe']:.2f} ≥ {2.45:.2f}")
    else:
        print(f"\n📈 PROGRESS! Gap reduced to {-results['improvement_vs_target']:.2f}")
        print("Try running again or increase training epochs for better results!")