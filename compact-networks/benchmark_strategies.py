import numpy as np

def sma_momentum_strategy(prices, lookback_short=50, lookback_long=200, threshold=0):
    positions = []
    
    for i in range(lookback_long, len(prices)-1):
        sma_short = np.mean(prices[i-lookback_short:i])
        sma_long = np.mean(prices[i-lookback_long:i])
        
        if sma_short - sma_long > threshold:
            positions.append([1, 0, 0])
        elif sma_short - sma_long < -threshold:
            positions.append([0, 1, 0])
        else:
            positions.append([0, 0, 1])
    
    return np.array(positions)

def sma_reversion_strategy(prices, lookback_short=50, lookback_long=200, threshold=0):
    positions = []
    
    for i in range(lookback_long, len(prices)-1):
        sma_short = np.mean(prices[i-lookback_short:i])
        sma_long = np.mean(prices[i-lookback_long:i])
        
        if sma_short - sma_long < -threshold:
            positions.append([1, 0, 0])
        elif sma_short - sma_long > threshold:
            positions.append([0, 1, 0])
        else:
            positions.append([0, 0, 1])
    
    return np.array(positions)