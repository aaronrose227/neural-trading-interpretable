# neural_policy_trading.py
"""
Neural network architecture for interpretable trading strategies.
Implements the architecture described in Section 3 of the paper with inductive priors.
"""

import torch
import torch.nn as nn
import numpy as np


class TradingNetwork(nn.Module):
    """
    Neural network architecture for interpretable trading strategies.
    Follows the exact specifications from Section 3 of the paper.
    
    Architecture:
    - Input Feature Subnetwork: Moving average computation (Section 3.1)
    - Feature Subnetwork: Threshold comparison neurons (Section 3.2) 
    - Logic Subnetwork: Decision neurons implementing logical rules (Section 3.3)
    - Output: Softargmax for position allocation (Section 3.4)
    """
    
    def __init__(self, lookback_long=200, lookback_short=50, beta=10.0):
        super(TradingNetwork, self).__init__()
        
        self.lookback_long = lookback_long
        self.lookback_short = lookback_short
        self.beta = beta  # Softargmax temperature parameter
        
        # Input Feature Subnetwork - Moving average weights (Section 3.1)
        self.w_0_1 = nn.Parameter(torch.zeros(lookback_long))  # Short-term MA
        self.w_0_2 = nn.Parameter(torch.zeros(lookback_long))  # Long-term MA
        
        # Feature Subnetwork - Threshold comparisons (Section 3.2)
        self.feature_layer = nn.Linear(2, 2, bias=True)
        
        # Logic Subnetwork - Trading decisions (Section 3.3)
        self.logic_layer = nn.Linear(2, 3, bias=True)
        
        # Initialize with domain knowledge
        self._initialize_ma_weights()
        self._initialize_momentum_strategy()
        
    def _initialize_ma_weights(self):
        """Initialize moving average weights as specified in Equation 5."""
        with torch.no_grad():
            # Short-term MA: equal weights for recent prices
            self.w_0_1[:self.lookback_short] = 1.0 / self.lookback_short
            self.w_0_1[self.lookback_short:] = 0.0
            
            # Long-term MA: equal weights for all prices
            self.w_0_2[:] = 1.0 / self.lookback_long
    
    def _initialize_momentum_strategy(self):
        """
        Initialize weights to implement momentum crossover strategy.
        Encodes domain knowledge as inductive priors (Section 3).
        """
        with torch.no_grad():
            # Feature layer: Detect MA crossovers
            # Neuron 1: fires when short_MA > long_MA
            self.feature_layer.weight[0, 0] = 1.0   # Positive short MA
            self.feature_layer.weight[0, 1] = -1.0  # Negative long MA
            self.feature_layer.bias[0] = 0.0
            
            # Neuron 2: fires when long_MA > short_MA  
            self.feature_layer.weight[1, 0] = -1.0  # Negative short MA
            self.feature_layer.weight[1, 1] = 1.0   # Positive long MA
            self.feature_layer.bias[1] = 0.0
            
            # Logic layer: Implement trading rules
            # Long position: when first feature fires (short > long)
            self.logic_layer.weight[0, 0] = 10.0    # Strong response to uptrend
            self.logic_layer.weight[0, 1] = -10.0   # Suppress when downtrend
            self.logic_layer.bias[0] = -5.0
            
            # Short position: when second feature fires (long > short)
            self.logic_layer.weight[1, 0] = -10.0   # Suppress when uptrend
            self.logic_layer.weight[1, 1] = 10.0    # Strong response to downtrend
            self.logic_layer.bias[1] = -5.0
            
            # Neutral position: generally discouraged
            self.logic_layer.weight[2, 0] = 0.0
            self.logic_layer.weight[2, 1] = 0.0
            self.logic_layer.bias[2] = -5.0
    
    def _initialize_reversion_strategy(self):
        """
        Initialize weights for mean reversion strategy.
        Implements opposite logic to momentum strategy.
        """
        with torch.no_grad():
            # Feature layer: Same MA crossing detection
            self.feature_layer.weight[0, 0] = 1.0
            self.feature_layer.weight[0, 1] = -1.0
            self.feature_layer.weight[1, 0] = -1.0
            self.feature_layer.weight[1, 1] = 1.0
            self.feature_layer.bias[0] = 0.0
            self.feature_layer.bias[1] = 0.0
            
            # Logic layer: Reversed trading rules (reversion logic)
            # Long when long_MA > short_MA (fade the trend)
            self.logic_layer.weight[0, 0] = -10.0
            self.logic_layer.weight[0, 1] = 10.0
            self.logic_layer.bias[0] = -5.0
            
            # Short when short_MA > long_MA (fade the trend)
            self.logic_layer.weight[1, 0] = 10.0
            self.logic_layer.weight[1, 1] = -10.0
            self.logic_layer.bias[1] = -5.0
            
            # Neutral position
            self.logic_layer.weight[2, 0] = 0.0
            self.logic_layer.weight[2, 1] = 0.0
            self.logic_layer.bias[2] = -5.0
    
    def _initialize_buy_and_hold(self):
        """Initialize network for buy-and-hold strategy."""
        with torch.no_grad():
            # Moving averages initialized normally
            self.w_0_1[:self.lookback_short] = 1.0 / self.lookback_short
            self.w_0_1[self.lookback_short:] = 0.0
            self.w_0_2[:] = 1.0 / self.lookback_long
            
            # Feature layer outputs don't affect decision
            self.feature_layer.weight.zero_()
            self.feature_layer.bias.zero_()
            
            # Logic layer: Always prefer long position
            self.logic_layer.weight[0, :] = 0.0
            self.logic_layer.bias[0] = 10.0      # Strong bias toward long
            
            self.logic_layer.weight[1, :] = 0.0
            self.logic_layer.bias[1] = -10.0     # Strong bias against short
            
            self.logic_layer.weight[2, :] = 0.0
            self.logic_layer.bias[2] = -10.0     # Strong bias against neutral
    
    def forward(self, x):
        """
        Forward pass through the network using raw prices (no input normalization).
        
        Args:
            x: Input tensor of shape (batch_size, lookback_long)
            
        Returns:
            Position probabilities [long, short, neutral] for each sample
        """
        # Input Feature Subnetwork: Compute moving averages
        ma_short = torch.sum(x * self.w_0_1, dim=1)  # Short-term MA
        ma_long = torch.sum(x * self.w_0_2, dim=1)   # Long-term MA
        
        # Stack MAs for feature processing
        ma_features = torch.stack([ma_short, ma_long], dim=1)
        
        # Feature Subnetwork: Threshold comparisons with sigmoid activation
        feature_output = torch.sigmoid(self.feature_layer(ma_features))
        
        # Logic Subnetwork: Trading decisions with sigmoid activation
        logic_output = torch.sigmoid(self.logic_layer(feature_output))
        
        # Output layer: Softargmax for position allocation (Section 3.4)
        positions = self.softargmax(logic_output)
        
        return positions
    
    def softargmax(self, x):
        """
        Softargmax function as defined in Section 3.4 of the paper.
        Uses temperature parameter beta to control decision sharpness.
        """
        exp_x = torch.exp(self.beta * x)
        return exp_x / exp_x.sum(dim=1, keepdim=True)
    
    def interpret_weights(self):
        """
        Interpret the learned weights as logical trading rules.
        Provides human-readable explanation of network decisions.
        
        Returns:
            Dictionary containing interpretations of feature and logic layers
        """
        interpretation = {
            'feature_layer': {
                'neuron_1': self._interpret_feature_neuron(0),
                'neuron_2': self._interpret_feature_neuron(1)
            },
            'logic_layer': {
                'long':    self._interpret_logic_neuron(0),
                'short':   self._interpret_logic_neuron(1),
                'neutral': self._interpret_logic_neuron(2)
            },
            'moving_averages': {
                'short_ma_active_weights': torch.sum(self.w_0_1 > 1e-6).item(),
                'long_ma_uniform': torch.allclose(self.w_0_2, torch.full_like(self.w_0_2, 1.0/self.lookback_long))
            }
        }
        return interpretation

    def _interpret_feature_neuron(self, idx):
        """Interpret a feature detection neuron as a logical condition."""
        w1 = self.feature_layer.weight[idx, 0].item()
        w2 = self.feature_layer.weight[idx, 1].item()
        b = self.feature_layer.bias[idx].item()
        
        return (f"Fires when: {w1:.2f}*short_MA + {w2:.2f}*long_MA + {b:.2f} > 0")

    def _interpret_logic_neuron(self, idx):
        """Interpret a logic neuron as a trading rule."""
        w1 = self.logic_layer.weight[idx, 0].item()
        w2 = self.logic_layer.weight[idx, 1].item()
        b = self.logic_layer.bias[idx].item()

        # Interpret common patterns
        if w1 > 5.0 and w2 < -5.0:
            return "Momentum: Long when short_MA > long_MA"
        elif w1 < -5.0 and w2 > 5.0:
            return "Reversion: Long when long_MA > short_MA"
        elif abs(w1) < 1.0 and abs(w2) < 1.0 and b > 5.0:
            return "Buy-and-Hold: Always active"
        elif abs(w1) < 1.0 and abs(w2) < 1.0 and b < -5.0:
            return "Inactive: Generally discouraged"
        else:
            return f"Custom rule: {w1:.2f}*z1 + {w2:.2f}*z2 + {b:.2f}"
