# neural_policy_trading.py
"""
Main implementation exactly as described in Section 3 of the paper
"""

import torch
import torch.nn as nn
import numpy as np


class TradingNetwork(nn.Module):
    """
    Neural network architecture for interpretable trading strategies
    Exactly as described in Section 3 of the paper.
    """
    
    def __init__(self, lookback_long=200, lookback_short=50, beta=10.0):
        super(TradingNetwork, self).__init__()
        
        self.lookback_long = lookback_long
        self.lookback_short = lookback_short
        self.beta = beta
        
        # Input Feature Subnetwork - moving average neurons (Section 3.1)
        # w^0,1 for short-term MA, w^0,2 for long-term MA
        self.w_0_1 = nn.Parameter(torch.zeros(lookback_long))
        self.w_0_2 = nn.Parameter(torch.zeros(lookback_long))
        
        # Initialize as simple moving averages (Equation 5)
        self._initialize_ma_weights()
        
        # Feature Subnetwork - threshold comparison neurons (Section 3.2)
        self.feature_layer = nn.Linear(2, 2, bias=True)
        
        # Logic Subnetwork - decision neurons (Section 3.3)
        self.logic_layer = nn.Linear(2, 3, bias=True)
        
        # Initialize with momentum strategy
        self._initialize_momentum_strategy()
        
    def _initialize_ma_weights(self):
        """Initialize moving average weights as in Equation 5"""
        with torch.no_grad():
            # w^0,1: Short-term MA - equal weights for first h~ elements
            self.w_0_1[:self.lookback_short] = 1.0 / self.lookback_short
            self.w_0_1[self.lookback_short:] = 0.0
            
            # w^0,2: Long-term MA - equal weights for all h elements
            self.w_0_2[:] = 1.0 / self.lookback_long
    
    def _initialize_momentum_strategy(self):
        """Initialize weights for momentum crossover strategy (Section 3)"""
        with torch.no_grad():
            # Feature layer weights as specified in Section 3.2
            # w^1,1 = (1, -1) for m~ - m > τ
            self.feature_layer.weight[0, 0] = 1.0
            self.feature_layer.weight[0, 1] = -1.0
            # w^1,2 = (-1, 1) for m~ - m < -τ
            self.feature_layer.weight[1, 0] = -1.0
            self.feature_layer.weight[1, 1] = 1.0
            
            # τ1 = τ2 = τ (threshold parameters)
            self.feature_layer.bias[0] = 0.0  # -τ1
            self.feature_layer.bias[1] = 0.0  # -τ2
            
            # Logic layer weights as specified in Section 3.3
            # Long position: z AND NOT z~
            self.logic_layer.weight[0, 0] = 1.0   # w^2,1_1 = 1
            self.logic_layer.weight[0, 1] = -1.0  # w^2,1_2 = -1
            self.logic_layer.bias[0] = 0.0        # b1 = 0
            
            # Short position: z~ AND NOT z
            self.logic_layer.weight[1, 0] = -1.0  # w^2,2_1 = -1
            self.logic_layer.weight[1, 1] = 1.0   # w^2,2_2 = 1
            self.logic_layer.bias[1] = 0.0        # b2 = 0
            
            # Neutral position: NOR gate
            self.logic_layer.weight[2, 0] = -1.0  # w^2,3_1 = -1
            self.logic_layer.weight[2, 1] = -1.0  # w^2,3_2 = -1
            self.logic_layer.bias[2] = 1.0        # b3 = 1
    
    def _initialize_reversion_strategy(self):
        """Initialize weights for mean reversion strategy"""
        with torch.no_grad():
            # Feature layer - same as momentum
            self.feature_layer.weight[0, 0] = 1.0
            self.feature_layer.weight[0, 1] = -1.0
            self.feature_layer.weight[1, 0] = -1.0
            self.feature_layer.weight[1, 1] = 1.0
            self.feature_layer.bias[0] = 0.0
            self.feature_layer.bias[1] = 0.0
            
            # Logic layer - reversed from momentum
            # Long when m~ < m (opposite of momentum)
            self.logic_layer.weight[0, 0] = -1.0
            self.logic_layer.weight[0, 1] = 1.0
            self.logic_layer.bias[0] = 0.0
            
            # Short when m~ > m (opposite of momentum)
            self.logic_layer.weight[1, 0] = 1.0
            self.logic_layer.weight[1, 1] = -1.0
            self.logic_layer.bias[1] = 0.0
            
            # Neutral
            self.logic_layer.weight[2, 0] = -1.0
            self.logic_layer.weight[2, 1] = -1.0
            self.logic_layer.bias[2] = 1.0
    
    def forward(self, x):
        """
        Forward pass through the network
        x: tensor of shape (batch_size, lookback_long)
        """
        # Input normalization (Section 3.5) - per sample to avoid look-ahead bias
        x_normalized = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
        
        # Input Feature Subnetwork - compute moving averages
        ma_short = torch.sum(x_normalized * self.w_0_1, dim=1)  # o^0,1
        ma_long = torch.sum(x_normalized * self.w_0_2, dim=1)   # o^0,2
        
        # Stack MAs for feature layer
        ma_features = torch.stack([ma_short, ma_long], dim=1)
        
        # Feature Subnetwork - threshold comparisons with sigmoid
        feature_output = torch.sigmoid(self.feature_layer(ma_features))
        
        # Logic Subnetwork - trading decisions with sigmoid
        logic_output = torch.sigmoid(self.logic_layer(feature_output))
        
        # Softargmax for final decision (Section 3.4)
        positions = self.softargmax(logic_output)
        
        return positions
    
    def softargmax(self, x):
        """Softargmax function as defined in Section 3.4"""
        exp_x = torch.exp(self.beta * x)
        return exp_x / exp_x.sum(dim=1, keepdim=True)
    

    def interpret_weights(self):
        """
        Interpret the learned weights as logical rules
        Returns a dictionary with interpretations for feature and logic layers
        """
        interpretation = {
            'feature_layer': {
                'neuron_1': (
                    f"fires when: "
                    f"{self.feature_layer.weight[0,0]:.2f}*MA_short + "
                    f"{self.feature_layer.weight[0,1]:.2f}*MA_long + "
                    f"{self.feature_layer.bias[0]:.2f} > 0"
                ),
                'neuron_2': (
                    f"fires when: "
                    f"{self.feature_layer.weight[1,0]:.2f}*MA_short + "
                    f"{self.feature_layer.weight[1,1]:.2f}*MA_long + "
                    f"{self.feature_layer.bias[1]:.2f} > 0"
                )
            },
            'logic_layer': {
                'long':    self._interpret_logic_neuron(0),
                'short':   self._interpret_logic_neuron(1),
                'neutral': self._interpret_logic_neuron(2)
            }
        }
        return interpretation

    def _interpret_logic_neuron(self, idx):
        """
        Interpret a single logic neuron as a logical formula
        Based on Section 3.3 of the paper:
          - Long:    z AND NOT z~
          - Short:   z~ AND NOT z
          - Neutral: NOT z AND NOT z~ (NOR)
        """
        w1 = self.logic_layer.weight[idx, 0].item()
        w2 = self.logic_layer.weight[idx, 1].item()
        b  = self.logic_layer.bias[idx].item()

        if w1 >  0.5 and w2 < -0.5:
            return "z AND NOT z~ (uptrend detected)"
        elif w1 < -0.5 and w2 >  0.5:
            return "z~ AND NOT z (downtrend detected)"
        elif w1 < -0.5 and w2 < -0.5 and b > 0.5:
            return "NOT z AND NOT z~ (no clear trend)"
        else:
            return f"Custom: {w1:.2f}*z + {w2:.2f}*z~ + {b:.2f}"
