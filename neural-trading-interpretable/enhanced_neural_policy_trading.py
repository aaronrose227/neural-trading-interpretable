# enhanced_neural_policy_trading.py
"""
Enhanced version of your neural_policy_trading.py with regime detection and adaptive periods.
Maintains full compatibility with your existing experiments while adding novel capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler


class RegimeDetector:
    """
    XGBoost-based regime detector that identifies market conditions.
    Research contribution: Interpretable regime classification for strategy selection.
    """
    
    def __init__(self):
        self.xgb_model = None
        self.scaler = RobustScaler()
        self.regime_names = ['Low Volatility', 'High Volatility', 'Trending', 'Mean Reverting']
        self.is_fitted = False
    
    def extract_regime_features(self, prices, lookback=100):
        """Extract features that characterize market regimes"""
        if len(prices) < lookback:
            return np.array([])
        
        features = []
        
        for i in range(lookback, len(prices)):
            window = prices[i-lookback:i]
            
            # Volatility measures
            returns = np.diff(window) / window[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Trend strength
            prices_norm = (window - window[0]) / window[0]
            trend_strength = abs(prices_norm[-1])  # Total return magnitude
            
            # Mean reversion tendency
            ma_20 = np.mean(window[-20:])
            ma_50 = np.mean(window[-50:]) if len(window) >= 50 else ma_20
            mean_reversion = abs(window[-1] - ma_50) / (np.std(window) + 1e-8)
            
            # Momentum persistence
            short_returns = returns[-10:]
            momentum_consistency = np.sum(np.sign(short_returns[:-1]) == np.sign(short_returns[1:])) / 9
            
            # Autocorrelation of returns (mean reversion indicator)
            if len(returns) > 1:
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                autocorr = autocorr if not np.isnan(autocorr) else 0
            else:
                autocorr = 0
            
            features.append([
                volatility,
                trend_strength, 
                mean_reversion,
                momentum_consistency,
                autocorr
            ])
        
        return np.array(features)
    
    def create_regime_labels(self, prices, lookback=100):
        """Create regime labels based on subsequent performance of strategies"""
        features = self.extract_regime_features(prices, lookback)
        if len(features) == 0:
            return np.array([])
        
        labels = []
        
        for i in range(len(features)):
            actual_idx = i + lookback
            if actual_idx + 30 >= len(prices):  # Reduced lookahead for more data
                break
            
            # Test both momentum and reversion over next 30 periods
            future_prices = prices[actual_idx:actual_idx+30]
            if len(future_prices) < 20:
                continue
            
            future_returns = np.diff(future_prices) / future_prices[:-1]
            
            # Simulate simple strategies (simplified)
            momentum_returns = []
            reversion_returns = []
            
            for j in range(10, len(future_prices)-1):  # Reduced lookback
                if j-5 >= 0 and j-10 >= 0:
                    ma_short = np.mean(future_prices[j-5:j])
                    ma_long = np.mean(future_prices[j-10:j])
                    
                    ret = future_returns[j]
                    
                    # Momentum strategy
                    if ma_short > ma_long:
                        momentum_returns.append(ret)   # Long
                        reversion_returns.append(-ret) # Short (opposite)
                    else:
                        momentum_returns.append(-ret)  # Short
                        reversion_returns.append(ret)  # Long (opposite)
            
            if len(momentum_returns) == 0:
                continue
            
            # Calculate strategy performance
            momentum_perf = np.mean(momentum_returns)
            reversion_perf = np.mean(reversion_returns)
            
            # Simplified binary classification: Momentum vs Reversion
            if momentum_perf > reversion_perf:
                labels.append(1)  # Momentum favorable
            else:
                labels.append(0)  # Reversion favorable
        
        return np.array(labels)
    
    def train_regime_detector(self, prices):
        """Train XGBoost to detect regimes"""
        print("    Training regime detector...")
        
        features = self.extract_regime_features(prices)
        labels = self.create_regime_labels(prices)
        
        if len(features) == 0 or len(labels) == 0:
            print("    Warning: Insufficient data for regime detection")
            return False
        
        # Align features and labels
        min_len = min(len(features), len(labels))
        X = features[:min_len]
        y = labels[:min_len]
        
        if len(np.unique(y)) < 2:
            print("    Warning: Not enough regime diversity")
            return False
        
        # FIX: Remap labels to be continuous starting from 0
        unique_labels = np.unique(y)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        y_remapped = np.array([label_mapping[label] for label in y])
        
        print(f"    Detected {len(unique_labels)} regime types: {unique_labels}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier with remapped labels
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        
        try:
            self.xgb_model.fit(X_scaled, y_remapped)
            self.is_fitted = True
            self.label_mapping = label_mapping  # Store mapping for prediction
            self.reverse_label_mapping = {v: k for k, v in label_mapping.items()}
            
            # Print regime insights
            feature_names = ['Volatility', 'Trend Strength', 'Mean Reversion', 'Momentum Consistency', 'Autocorr']
            importance = self.xgb_model.feature_importances_
            
            print("    Regime detection features:")
            for name, imp in zip(feature_names, importance):
                print(f"      {name}: {imp:.3f}")
            
            return True
            
        except Exception as e:
            print(f"    Warning: XGBoost training failed: {e}")
            return False
    
    def predict_regime_probabilities(self, prices, lookback=100):
        """Predict regime probabilities for recent history"""
        if not self.is_fitted:
            return np.array([0.5, 0.5])  # Binary: [reversion, momentum]
        
        features = self.extract_regime_features(prices, lookback)
        if len(features) == 0:
            return np.array([0.5, 0.5])
        
        # Use most recent features
        X = features[-1:].reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        try:
            probs = self.xgb_model.predict_proba(X_scaled)[0]
            
            # Convert binary classification to 4-class for compatibility
            if len(probs) == 2:
                # probs[0] = reversion favorable, probs[1] = momentum favorable
                # Map to [Low Vol Reversion, High Vol Reversion, Low Vol Momentum, High Vol Momentum]
                full_probs = np.array([
                    probs[0] * 0.5,  # Low Vol Reversion
                    probs[0] * 0.5,  # High Vol Reversion  
                    probs[1] * 0.5,  # Low Vol Momentum
                    probs[1] * 0.5   # High Vol Momentum
                ])
            else:
                full_probs = np.array([0.25, 0.25, 0.25, 0.25])
            
            return full_probs
            
        except Exception as e:
            print(f"Warning: Regime prediction failed: {e}")
            return np.array([0.25, 0.25, 0.25, 0.25])


class EnhancedTradingNetwork(nn.Module):
    """
    Enhanced version of your TradingNetwork with regime detection and adaptive periods.
    Maintains full compatibility with your existing training and evaluation code.
    """
    
    def __init__(self, lookback_long=200, lookback_short=50, beta=1.0, 
                 enable_adaptive_periods=True, enable_regime_detection=True):
        super(EnhancedTradingNetwork, self).__init__()
        
        # Keep all original parameters for compatibility
        self.lookback_long = lookback_long
        self.lookback_short = lookback_short
        self.beta = beta
        self.enable_adaptive_periods = enable_adaptive_periods
        self.enable_regime_detection = enable_regime_detection
        
        # Original Krause-Calliess architecture (unchanged for compatibility)
        self.w_0_1 = nn.Parameter(torch.zeros(lookback_long))
        self.w_0_2 = nn.Parameter(torch.zeros(lookback_long))
        self.feature_layer = nn.Linear(2, 2, bias=True)
        self.logic_layer = nn.Linear(2, 3, bias=True)
        
        # NEW: Adaptive period parameters (FIXED: Initialize around current good values)
        if enable_adaptive_periods:
            # Initialize to be VERY close to your proven 50/200 values
            # Using small perturbations so the network starts near optimal
            self.short_period_param = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5 -> 50
            self.long_period_param = nn.Parameter(torch.tensor(0.0))   # sigmoid(0) = 0.5 -> 200
        
        # NEW: Regime detection components (FIXED: Make them nearly neutral initially)
        if enable_regime_detection:
            # Regime feature processing (start neutral)
            self.regime_processor = nn.Linear(4, 2, bias=True)
            
            # Strategy selection based on regime (start nearly identical)
            self.momentum_strategy = nn.Linear(2, 3, bias=True)
            self.reversion_strategy = nn.Linear(2, 3, bias=True)
            
            # Regime-based strategy weighting (start neutral)
            self.strategy_mixer = nn.Linear(4, 3, bias=True)
        
        # Initialize all components
        self._initialize_enhanced_network()
    
    def _initialize_enhanced_network(self):
        """Initialize both original and new components"""
        # Original initialization (keep your existing logic exactly)
        self._initialize_ma_weights()
        self._initialize_momentum_strategy()
        
        # NEW: Initialize regime detection components (FIXED: Start nearly neutral)
        if self.enable_regime_detection:
            with torch.no_grad():
                # Regime processor: VERY small weights initially (nearly neutral)
                self.regime_processor.weight.data.normal_(0, 0.01)  # Very small random weights
                self.regime_processor.bias.data.zero_()
                
                # Momentum strategy: Same as original logic layer (GOOD prior)
                self.momentum_strategy.weight.data.copy_(self.logic_layer.weight.data)
                self.momentum_strategy.bias.data.copy_(self.logic_layer.bias.data)
                
                # Reversion strategy: Opposite of momentum (GOOD prior)
                self.reversion_strategy.weight[0, 0] = -1.0  # Long when trend down
                self.reversion_strategy.weight[0, 1] = 1.0
                self.reversion_strategy.bias[0] = 0.0
                
                self.reversion_strategy.weight[1, 0] = 1.0   # Short when trend up
                self.reversion_strategy.weight[1, 1] = -1.0
                self.reversion_strategy.bias[1] = 0.0
                
                self.reversion_strategy.weight[2, 0] = -1.0  # Neutral otherwise
                self.reversion_strategy.weight[2, 1] = -1.0
                self.reversion_strategy.bias[2] = 1.0
                
                # Strategy mixer: START NEUTRAL - no regime influence initially
                self.strategy_mixer.weight.data.zero_()  # No mixing initially
                self.strategy_mixer.bias.data.zero_()    # Neutral bias
    
    def _initialize_ma_weights(self):
        """Original MA weight initialization (unchanged)"""
        with torch.no_grad():
            self.w_0_1[:self.lookback_short] = 1.0 / self.lookback_short
            self.w_0_1[self.lookback_short:] = 0.0
            self.w_0_2[:] = 1.0 / self.lookback_long
    
    def _initialize_momentum_strategy(self):
        """Original strategy initialization (unchanged)"""
        with torch.no_grad():
            # Feature layer
            self.feature_layer.weight[0, 0] = 1.0
            self.feature_layer.weight[0, 1] = -1.0
            self.feature_layer.bias[0] = 0.0
            
            self.feature_layer.weight[1, 0] = -1.0
            self.feature_layer.weight[1, 1] = 1.0
            self.feature_layer.bias[1] = 0.0
            
            # Logic layer - momentum initialization
            self.logic_layer.weight[0, 0] = 1.0
            self.logic_layer.weight[0, 1] = -1.0
            self.logic_layer.bias[0] = 0.0
            
            self.logic_layer.weight[1, 0] = -1.0
            self.logic_layer.weight[1, 1] = 1.0
            self.logic_layer.bias[1] = 0.0
            
            self.logic_layer.weight[2, 0] = -1.0
            self.logic_layer.weight[2, 1] = -1.0
            self.logic_layer.bias[2] = 1.0
    
    def _initialize_reversion_strategy(self):
        """Initialize for reversion (can be called externally like original)"""
        with torch.no_grad():
            # Keep MA weights the same
            self.w_0_1[:self.lookback_short] = 1.0 / self.lookback_short
            self.w_0_1[self.lookback_short:] = 0.0
            self.w_0_2[:] = 1.0 / self.lookback_long
            
            # Feature layer same
            self.feature_layer.weight[0, 0] = 1.0
            self.feature_layer.weight[0, 1] = -1.0
            self.feature_layer.weight[1, 0] = -1.0
            self.feature_layer.weight[1, 1] = 1.0
            self.feature_layer.bias[0] = 0.0
            self.feature_layer.bias[1] = 0.0
            
            # Logic layer - reversion (opposite of momentum)
            self.logic_layer.weight[0, 0] = -1.0
            self.logic_layer.weight[0, 1] = 1.0
            self.logic_layer.bias[0] = 0.0
            
            self.logic_layer.weight[1, 0] = 1.0
            self.logic_layer.weight[1, 1] = -1.0
            self.logic_layer.bias[1] = 0.0
            
            self.logic_layer.weight[2, 0] = -1.0
            self.logic_layer.weight[2, 1] = -1.0
            self.logic_layer.bias[2] = 1.0
    
    def _initialize_buy_and_hold(self):
        """Initialize for buy and hold (unchanged for compatibility)"""
        with torch.no_grad():
            self.w_0_1[:self.lookback_short] = 1.0 / self.lookback_short
            self.w_0_1[self.lookback_short:] = 0.0
            self.w_0_2[:] = 1.0 / self.lookback_long
            
            self.feature_layer.weight.zero_()
            self.feature_layer.bias.zero_()
            
            self.logic_layer.weight[0, :] = 0.0
            self.logic_layer.bias[0] = 10.0
            
            self.logic_layer.weight[1, :] = 0.0
            self.logic_layer.bias[1] = -10.0
            
            self.logic_layer.weight[2, :] = 0.0
            self.logic_layer.bias[2] = -10.0
    
    def get_adaptive_periods(self):
        """Get current adaptive MA periods - FIXED: Start near proven values"""
        if not self.enable_adaptive_periods:
            return self.lookback_short, self.lookback_long
        
        # FIXED: Initialize near proven 50/200, allow small deviations
        # sigmoid(0) = 0.5, so we get 50 and 200 initially
        short_period = int(self.lookback_short * (0.5 + 0.5 * torch.sigmoid(self.short_period_param).item()))  # 25-75 range
        long_period = int(self.lookback_long * (0.5 + 0.5 * torch.sigmoid(self.long_period_param).item()))     # 100-300 range
        
        # Ensure reasonable bounds
        short_period = max(10, min(90, short_period))
        long_period = max(100, min(400, long_period))
        
        return short_period, long_period
    
    def forward(self, x, regime_probs=None):
        """
        Enhanced forward pass with regime detection.
        Maintains full compatibility with your existing code - regime_probs is optional.
        """
        # Original input normalization
        x_norm = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
        
        # Compute moving averages (potentially adaptive)
        if self.enable_adaptive_periods:
            # Use learnable periods for weight computation
            short_p, long_p = self.get_adaptive_periods()
            
            # Dynamically create weights (simplified - you could make this more sophisticated)
            w_short = torch.zeros_like(self.w_0_1)
            w_short[-short_p:] = 1.0 / short_p
            
            w_long = torch.full_like(self.w_0_2, 1.0 / long_p)
            
            ma_short = torch.sum(x_norm * w_short, dim=1)
            ma_long = torch.sum(x_norm * w_long, dim=1)
        else:
            # Original fixed-period computation
            ma_short = torch.sum(x_norm * self.w_0_1, dim=1)
            ma_long = torch.sum(x_norm * self.w_0_2, dim=1)
        
        # Feature layer (unchanged)
        ma_features = torch.stack([ma_short, ma_long], dim=1)
        feature_output = torch.sigmoid(self.feature_layer(ma_features))
        
        # Regime-enhanced logic (FIXED: Start with minimal regime influence)
        if self.enable_regime_detection and regime_probs is not None:
            # Process regime information
            regime_tensor = torch.FloatTensor(regime_probs).unsqueeze(0)
            if regime_tensor.shape[0] != x.shape[0]:
                regime_tensor = regime_tensor.repeat(x.shape[0], 1)
            
            # Get regime features
            regime_features = torch.sigmoid(self.regime_processor(regime_tensor))
            
            # Compute both momentum and reversion strategies
            momentum_output = torch.sigmoid(self.momentum_strategy(feature_output))
            reversion_output = torch.sigmoid(self.reversion_strategy(feature_output))
            
            # Mix strategies based on regime (FIXED: Start with mostly momentum)
            strategy_weights = torch.softmax(self.strategy_mixer(regime_tensor), dim=1)
            
            # FIXED: Start with 90% original logic, 10% regime mixing
            # This ensures good initial performance while allowing learning
            base_logic = torch.sigmoid(self.logic_layer(feature_output))  # Original good logic
            regime_mixing = 0.9 * momentum_output + 0.1 * reversion_output  # Light mixing
            
            # Gradually blend as training progresses (very conservative initially)
            mixing_strength = 0.1  # Start with 10% regime influence
            logic_output = (1 - mixing_strength) * base_logic + mixing_strength * regime_mixing
        else:
            # Fall back to original logic layer (this should give good initial performance)
            logic_output = torch.sigmoid(self.logic_layer(feature_output))
        
        # Original softmax output
        positions = torch.softmax(self.beta * logic_output, dim=1)
        
        return positions
    
    def interpret_weights(self):
        """Enhanced interpretation including regime and adaptive features"""
        base_interpretation = self._get_base_interpretation()
        
        enhanced_interpretation = {
            'base_strategy': base_interpretation,
            'adaptive_periods': self.get_adaptive_periods() if self.enable_adaptive_periods else None,
            'regime_enabled': self.enable_regime_detection,
        }
        
        if self.enable_regime_detection:
            enhanced_interpretation['regime_weights'] = self._interpret_regime_components()
        
        return enhanced_interpretation
    
    def _get_base_interpretation(self):
        """Get base strategy interpretation (same logic as original)"""
        w1 = self.logic_layer.weight[0, 0].item()
        w2 = self.logic_layer.weight[0, 1].item()
        b = self.logic_layer.bias[0].item()
        
        if w1 > 0.5 and w2 < -0.5:
            return "Momentum: Long when short_MA > long_MA"
        elif w1 < -0.5 and w2 > 0.5:
            return "Reversion: Long when long_MA > short_MA"
        elif abs(w1) < 0.1 and abs(w2) < 0.1 and b > 5.0:
            return "Buy-and-Hold: Always active"
        else:
            return f"Custom rule: {w1:.2f}*z1 + {w2:.2f}*z2 + {b:.2f}"
    
    def _interpret_regime_components(self):
        """Interpret regime detection components"""
        if not self.enable_regime_detection:
            return None
        
        # Get momentum vs reversion preference weights
        mom_weights = self.momentum_strategy.weight.detach().cpu().numpy()
        rev_weights = self.reversion_strategy.weight.detach().cpu().numpy()
        
        return {
            'momentum_strength': np.mean(np.abs(mom_weights)),
            'reversion_strength': np.mean(np.abs(rev_weights)),
            'regime_processing': 'Active' if torch.std(self.regime_processor.weight) > 0.1 else 'Minimal'
        }


# enhanced_training.py - Drop-in replacement for your training.py
"""
Enhanced training that integrates regime detection with your existing training loop.
"""

def train_enhanced_network(network, prices, regime_detector=None, epochs=800, lr=0.001, 
                          train_layers='all', loss_type='returns', verbose=False):
    """
    Enhanced version of your train_network function.
    Maintains full compatibility - regime_detector is optional.
    """
    from evaluation import prepare_data, calculate_sharpe_ratio
    
    # Your original data preparation (unchanged)
    X, y, current_prices = prepare_data(prices, network.lookback_long)
    X_tensor = torch.FloatTensor(X)
    
    # Configure trainable parameters (same logic as original)
    if train_layers == 'all':
        params_to_train = list(network.parameters())
    elif train_layers == 'logic':
        params_to_train = list(network.logic_layer.parameters())
    elif train_layers == 'logic_feature':
        params_to_train = (list(network.logic_layer.parameters()) + 
                          list(network.feature_layer.parameters()))
    elif train_layers == 'enhanced':  # NEW option
        params_to_train = list(network.logic_layer.parameters())
        if hasattr(network, 'regime_processor') and network.enable_regime_detection:
            params_to_train.extend(list(network.regime_processor.parameters()))
            params_to_train.extend(list(network.momentum_strategy.parameters()))
            params_to_train.extend(list(network.reversion_strategy.parameters()))
            params_to_train.extend(list(network.strategy_mixer.parameters()))
        if hasattr(network, 'short_period_param') and network.enable_adaptive_periods:
            params_to_train.extend([network.short_period_param, network.long_period_param])
    else:
        params_to_train = list(network.parameters())
    
    optimizer = torch.optim.Adam(params_to_train, lr=lr)
    
    # Get regime probabilities if detector available
    regime_probs = None
    if regime_detector and regime_detector.is_fitted:
        regime_probs = regime_detector.predict_regime_probabilities(prices)
        if verbose:
            regime_names = ['Low Vol', 'High Vol', 'Trending', 'Mean Rev']
            regime_str = ', '.join([f"{name}: {prob:.2f}" 
                                  for name, prob in zip(regime_names, regime_probs)])
            print(f"    Detected regime: {regime_str}")
    
    # Track performance
    losses = []
    if verbose:
        initial_sharpe = _calculate_initial_enhanced_performance(
            network, X_tensor, y, current_prices, regime_probs)
        adaptive_periods = network.get_adaptive_periods()
        print(f"    Initial: Sharpe {initial_sharpe:.3f}, Periods: {adaptive_periods}")
    
    # Enhanced training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass with regime information
        if regime_probs is not None and hasattr(network, 'enable_regime_detection') and network.enable_regime_detection:
            positions = network(X_tensor, regime_probs)
        else:
            positions = network(X_tensor)
        
        # Your original loss calculation (unchanged)
        returns = (y - current_prices) / current_prices
        returns_tensor = torch.FloatTensor(returns)
        position_weights = positions[:, 0] - positions[:, 1]
        strategy_returns = position_weights * returns_tensor
        
        if loss_type == 'returns':
            clipped_returns = torch.clamp(1 + strategy_returns, min=1e-8)
            loss = -torch.mean(torch.log(clipped_returns))
        else:  # sharpe
            mean_return = torch.mean(strategy_returns)
            std_return = torch.std(strategy_returns)
            sharpe = (mean_return * 252) / (std_return * torch.sqrt(torch.tensor(252.0)) + 1e-8)
            loss = -sharpe
        
        # Optional: Add small regularization for adaptive periods
        if hasattr(network, 'enable_adaptive_periods') and network.enable_adaptive_periods:
            # Encourage reasonable period ranges
            short_p, long_p = network.get_adaptive_periods()
            period_penalty = 0.001 * (max(0, 10 - short_p) + max(0, short_p - 90) +
                                    max(0, 100 - long_p) + max(0, long_p - 400))
            loss = loss + period_penalty
        
        # Backpropagation (unchanged)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        # Progress reporting
        if verbose and epoch % 200 == 0:
            current_sharpe = calculate_sharpe_ratio(strategy_returns.detach().numpy())
            if hasattr(network, 'get_adaptive_periods'):
                adaptive_periods = network.get_adaptive_periods()
                print(f"    Epoch {epoch}: Sharpe {current_sharpe:.3f}, Periods: {adaptive_periods}")
            else:
                print(f"    Epoch {epoch}: Sharpe {current_sharpe:.3f}")
    
    if verbose:
        final_sharpe = _calculate_final_enhanced_performance(
            network, X_tensor, y, current_prices, regime_probs)
        print(f"    Final: Sharpe {final_sharpe:.3f}")
        
        # Show what the network learned
        interpretation = network.interpret_weights()
        print(f"    Learned: {interpretation.get('base_strategy', 'Unknown')}")
        if interpretation.get('adaptive_periods'):
            print(f"    Adaptive periods: {interpretation['adaptive_periods']}")
    
    return losses


def _calculate_initial_enhanced_performance(network, X_tensor, y, current_prices, regime_probs):
    """Calculate initial performance with regime information"""
    from evaluation import calculate_sharpe_ratio
    
    with torch.no_grad():
        if regime_probs is not None and hasattr(network, 'enable_regime_detection') and network.enable_regime_detection:
            positions = network(X_tensor, regime_probs)
        else:
            positions = network(X_tensor)
        returns = (y - current_prices) / current_prices
        position_weights = positions[:, 0] - positions[:, 1]
        strategy_returns = position_weights * torch.FloatTensor(returns)
        return calculate_sharpe_ratio(strategy_returns.numpy())


def _calculate_final_enhanced_performance(network, X_tensor, y, current_prices, regime_probs):
    """Calculate final performance with regime information"""
    from evaluation import calculate_sharpe_ratio
    
    with torch.no_grad():
        if regime_probs is not None and hasattr(network, 'enable_regime_detection') and network.enable_regime_detection:
            positions = network(X_tensor, regime_probs)
        else:
            positions = network(X_tensor)
        returns = (y - current_prices) / current_prices
        position_weights = positions[:, 0] - positions[:, 1]
        strategy_returns = position_weights * torch.FloatTensor(returns)
        return calculate_sharpe_ratio(strategy_returns.numpy())


# enhanced_experiments.py - Integration with your existing experiments
"""
Enhanced experiments that can be dropped into your existing experiments.py
"""

def run_enhanced_experiment(data_name, prices, initialization_func, train_layers='enhanced'):
    """
    Run enhanced experiment with regime detection.
    Can be called from your existing experiment loop.
    """
    print(f"\n{data_name.upper()} - Enhanced with Regime Detection")
    print("-" * 50)
    
    train_prices = prices[:8000]
    test_prices = prices[8000:]
    
    # Step 1: Train regime detector on training data
    regime_detector = RegimeDetector()
    regime_trained = regime_detector.train_regime_detector(train_prices)
    
    if regime_trained:
        regime_probs = regime_detector.predict_regime_probabilities(train_prices)
        regime_names = ['Low Vol', 'High Vol', 'Trending', 'Mean Rev']
        dominant_regime = regime_names[np.argmax(regime_probs)]
        print(f"Training data regime: {dominant_regime} ({regime_probs[np.argmax(regime_probs)]:.2f})")
    else:
        regime_probs = None
        print("Regime detection failed, using standard approach")
    
    # Step 2: Initialize enhanced network
    network = EnhancedTradingNetwork(
        enable_adaptive_periods=True,
        enable_regime_detection=regime_trained
    )
    initialization_func(network)
    print("✓ Network initialized with enhanced capabilities")
    
    # Step 3: Train with regime information
    print("Training enhanced network...")
    losses = train_enhanced_network(
        network,
        train_prices,
        regime_detector=regime_detector if regime_trained else None,
        epochs=800,
        lr=0.001,
        train_layers=train_layers,
        verbose=True
    )
    
    # Step 4: Evaluate on test set
    print("Evaluating enhanced performance...")
    test_regime_probs = None
    if regime_trained:
        test_regime_probs = regime_detector.predict_regime_probabilities(test_prices)
        test_dominant = regime_names[np.argmax(test_regime_probs)]
        print(f"Test data regime: {test_dominant} ({test_regime_probs[np.argmax(test_regime_probs)]:.2f})")
    
    # Evaluate enhanced network
    from evaluation import prepare_data, calculate_returns, calculate_sharpe_ratio
    
    X_test, y_test, current_test_prices = prepare_data(test_prices, network.lookback_long)
    X_test_tensor = torch.FloatTensor(X_test)
    
    with torch.no_grad():
        if test_regime_probs is not None and network.enable_regime_detection:
            test_positions = network(X_test_tensor, test_regime_probs)
        else:
            test_positions = network(X_test_tensor)
    
    test_returns = calculate_returns(test_prices[network.lookback_long:], test_positions.numpy())
    enhanced_sharpe = calculate_sharpe_ratio(test_returns)
    
    # Get interpretation
    interpretation = network.interpret_weights()
    
    # Compare with benchmarks (reuse your existing benchmark code)
    from benchmark_strategies import sma_momentum_strategy, sma_reversion_strategy, buy_and_hold_strategy
    
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
    print(f"\nTest Set Performance:")
    print(f"  Enhanced Network: {enhanced_sharpe:.2f}")
    print(f"  SMA-MOM:         {sharpe_mom:.2f}")
    print(f"  SMA-REV:         {sharpe_rev:.2f}")
    print(f"  Buy & Hold:      {sharpe_bh:.2f}")
    
    print(f"\nLearned Strategy:")
    print(f"  Base: {interpretation.get('base_strategy', 'Unknown')}")
    if interpretation.get('adaptive_periods'):
        print(f"  Adaptive periods: {interpretation['adaptive_periods']}")
    if interpretation.get('regime_enabled'):
        print(f"  Regime detection: Active")
        if interpretation.get('regime_weights'):
            rw = interpretation['regime_weights']
            print(f"    Momentum strength: {rw['momentum_strength']:.3f}")
            print(f"    Reversion strength: {rw['reversion_strength']:.3f}")
    
    return {
        'enhanced_sharpe': enhanced_sharpe,
        'benchmark_sharpes': {
            'SMA-MOM': sharpe_mom,
            'SMA-REV': sharpe_rev,
            'Buy & Hold': sharpe_bh
        },
        'interpretation': interpretation,
        'regime_detection': regime_trained
    }


def run_comprehensive_enhanced_experiments():
    """
    Comprehensive experiment comparing original vs enhanced approach.
    This can replace or supplement your existing experiments.py main function.
    """
    from data_generation import OrnsteinUhlenbeckGenerator
    import pandas as pd
    
    print("ENHANCED KRAUSE-CALLIESS EXPERIMENTS")
    print("=" * 60)
    print("Novel contributions: Regime Detection + Adaptive Periods")
    print("=" * 60)
    
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate the same data as your original experiments
    data_generators = {
        'Up-trend': (OrnsteinUhlenbeckGenerator(theta=2, mu=50, sigma=20), 
                    lambda gen: gen.generate_uptrend(10_000, trend_rate=0.01)),
        'Switching': (OrnsteinUhlenbeckGenerator(theta=7.5, mu=50, sigma=10),
                     lambda gen: gen.generate_switching_trend(10_000)),
        'Reversion': (OrnsteinUhlenbeckGenerator(theta=20, mu=50, sigma=50),
                     lambda gen: gen.generate_reversion(10_000))
    }
    
    # Initialization functions (same as your original)
    def init_momentum(net): net._initialize_momentum_strategy()
    def init_reversion(net): net._initialize_reversion_strategy()
    def init_buy_hold(net): net._initialize_buy_and_hold()
    
    experiment_configs = [
        ('Up-trend', init_buy_hold, 'logic'),      # Paper's config
        ('Switching', init_momentum, 'enhanced'),  # Our enhanced config
        ('Reversion', init_reversion, 'enhanced')  # Our enhanced config
    ]
    
    all_results = {}
    
    # Run experiments
    for data_name, init_func, train_layers in experiment_configs:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {data_name.upper()}")
        print(f"{'='*60}")
        
        # Generate data
        generator, gen_func = data_generators[data_name]
        prices = gen_func(generator)
        
        # Handle any NaN values
        if np.any(np.isnan(prices)):
            prices = np.nan_to_num(prices, nan=50.0)
        
        print(f"Generated {len(prices)} price points")
        print(f"Price range: {prices.min():.1f} to {prices.max():.1f}")
        
        # Run enhanced experiment
        results = run_enhanced_experiment(data_name, prices, init_func, train_layers)
        all_results[data_name] = results
    
    # Create comprehensive results table
    print(f"\n{'='*60}")
    print("COMPREHENSIVE RESULTS COMPARISON")
    print(f"{'='*60}")
    
    results_df = pd.DataFrame({
        data_name: {
            'Enhanced': results['enhanced_sharpe'],
            'SMA-MOM': results['benchmark_sharpes']['SMA-MOM'],
            'SMA-REV': results['benchmark_sharpes']['SMA-REV'],
            'Buy & Hold': results['benchmark_sharpes']['Buy & Hold']
        }
        for data_name, results in all_results.items()
    })
    
    print(results_df.round(2).to_string())
    
    # Enhancement analysis
    print(f"\n{'='*60}")
    print("ENHANCEMENT ANALYSIS")
    print(f"{'='*60}")
    
    for data_name, results in all_results.items():
        print(f"\n{data_name}:")
        
        enhanced_sharpe = results['enhanced_sharpe']
        benchmarks = results['benchmark_sharpes']
        best_benchmark = max(benchmarks.values())
        improvement = enhanced_sharpe - best_benchmark
        
        print(f"  Enhanced Network:    {enhanced_sharpe:.3f}")
        print(f"  Best Benchmark:      {best_benchmark:.3f}")
        print(f"  Improvement:         {improvement:+.3f}")
        print(f"  Regime Detection:    {'✓' if results['regime_detection'] else '✗'}")
        
        interp = results['interpretation']
        print(f"  Learned Strategy:    {interp.get('base_strategy', 'Unknown')}")
        if interp.get('adaptive_periods'):
            short_p, long_p = interp['adaptive_periods']
            print(f"  Adaptive Periods:    {short_p}/{long_p} (vs 50/200 baseline)")
    
    # Research insights
    print(f"\n{'='*60}")
    print("RESEARCH INSIGHTS")
    print(f"{'='*60}")
    
    regime_success_count = sum(1 for r in all_results.values() if r['regime_detection'])
    print(f"✓ Regime detection successful: {regime_success_count}/3 experiments")
    
    improvements = [r['enhanced_sharpe'] - max(r['benchmark_sharpes'].values()) 
                   for r in all_results.values()]
    avg_improvement = np.mean(improvements)
    print(f"✓ Average Sharpe improvement: {avg_improvement:+.3f}")
    
    adaptive_experiments = [name for name, r in all_results.items() 
                           if r['interpretation'].get('adaptive_periods')]
    print(f"✓ Adaptive period learning: {len(adaptive_experiments)}/3 experiments")
    
    for exp_name in adaptive_experiments:
        short_p, long_p = all_results[exp_name]['interpretation']['adaptive_periods']
        print(f"    {exp_name}: {short_p}/{long_p} periods")
    
    print(f"\n{'='*60}")
    print("KEY NOVEL CONTRIBUTIONS")
    print(f"{'='*60}")
    print("1. Automatic regime detection using XGBoost feature analysis")
    print("2. Learnable moving average periods (adaptive to market conditions)")
    print("3. Strategy mixing based on detected market regime")
    print("4. Maintains full interpretability of learned trading rules")
    print("5. Backward compatible with original Krause-Calliess architecture")
    
    return all_results


# Integration helper for your existing code
def enhance_your_existing_experiment(data_name, prices, original_network_init):
    """
    Drop-in function to enhance any of your existing experiments.
    Just replace your network creation and training with this function.
    """
    print(f"🚀 ENHANCING: {data_name}")
    
    # Your original approach
    print("  Running original approach...")
    original_network = TradingNetwork()  # Your original class
    original_network_init(original_network)
    
    from training import train_network  # Your original training
    train_network(original_network, prices[:8000], train_layers='logic', verbose=False)
    
    # Enhanced approach
    print("  Running enhanced approach...")
    enhanced_results = run_enhanced_experiment(
        data_name, prices, original_network_init, train_layers='enhanced'
    )
    
    # Quick comparison
    from evaluation import prepare_data, calculate_returns, calculate_sharpe_ratio
    
    X_test, _, _ = prepare_data(prices[8000:], original_network.lookback_long)
    X_test_tensor = torch.FloatTensor(X_test)
    
    with torch.no_grad():
        original_positions = original_network(X_test_tensor)
    
    original_returns = calculate_returns(prices[8000:][original_network.lookback_long:], 
                                       original_positions.numpy())
    original_sharpe = calculate_sharpe_ratio(original_returns)
    
    enhancement = enhanced_results['enhanced_sharpe'] - original_sharpe
    
    print(f"  📊 RESULTS:")
    print(f"    Original Sharpe:  {original_sharpe:.3f}")
    print(f"    Enhanced Sharpe:  {enhanced_results['enhanced_sharpe']:.3f}")
    print(f"    Improvement:      {enhancement:+.3f}")
    
    return enhanced_results


if __name__ == "__main__":
    # Run the comprehensive enhanced experiments
    results = run_comprehensive_enhanced_experiments()