import numpy as np
import pandas as pd

def calculate_all_indicators(prices, lookback_long=200, lookback_short=50):
    prices_series = pd.Series(prices)
    indicators = {}
    
    # Moving averages
    indicators['sma_short'] = prices_series.rolling(lookback_short).mean()
    indicators['sma_long'] = prices_series.rolling(lookback_long).mean()
    
    # Momentum indicators
    indicators['momentum_5'] = prices_series.pct_change(5)
    indicators['momentum_20'] = prices_series.pct_change(20)
    indicators['price_ma_ratio'] = prices_series / indicators['sma_long']
    
    # Volatility indicators
    returns = prices_series.pct_change()
    indicators['volatility_10'] = returns.rolling(10).std()
    indicators['volatility_30'] = returns.rolling(30).std()
    
    # Mean reversion indicators
    indicators['mean_reversion'] = (prices_series - indicators['sma_long']) / (indicators['sma_long'] * indicators['volatility_30'])
    indicators['bollinger_position'] = (prices_series - indicators['sma_short']) / (2 * indicators['volatility_10'] * indicators['sma_short'])
    
    # Trend strength indicators
    indicators['ma_spread'] = (indicators['sma_short'] - indicators['sma_long']) / indicators['sma_long']
    indicators['trend_consistency'] = indicators['momentum_5'].rolling(10).apply(lambda x: np.sum(x > 0) / len(x))
    
    # Synthetic market indicators
    synthetic_market = create_synthetic_market_indicators(prices_series)
    indicators.update(synthetic_market)
    
    # Convert to DataFrame and handle NaN values
    indicator_df = pd.DataFrame(indicators)
    indicator_df = indicator_df.fillna(method='ffill').fillna(method='bfill')
    
    return indicator_df

def create_synthetic_market_indicators(price_series):
    returns = price_series.pct_change()
    
    # Synthetic VIX
    base_vix = 20
    vix_component = -10 * returns.rolling(5).mean() + 5 * returns.rolling(20).std()
    synthetic_vix = base_vix + vix_component + np.random.normal(0, 1, len(price_series))
    synthetic_vix = np.clip(synthetic_vix, 5, 80)
    
    # Synthetic Interest Rate
    base_rate = 2.0
    rate_trend = 0.001 * np.arange(len(price_series)) / 100
    rate_noise = np.random.normal(0, 0.05, len(price_series))
    synthetic_rate = base_rate + rate_trend + rate_noise
    synthetic_rate = np.clip(synthetic_rate, 0.1, 8.0)
    
    # Synthetic Dollar Index
    base_dxy = 100
    dxy_trend = -0.1 * returns.cumsum()
    dxy_noise = np.random.normal(0, 0.5, len(price_series))
    synthetic_dxy = base_dxy + dxy_trend + dxy_noise
    
    # Market regime indicator
    regime_strength = abs(returns.rolling(20).mean()) / (returns.rolling(20).std() + 1e-8)
    
    return {
        'synthetic_vix': pd.Series(synthetic_vix, index=price_series.index),
        'synthetic_rate': pd.Series(synthetic_rate, index=price_series.index),
        'synthetic_dxy': pd.Series(synthetic_dxy, index=price_series.index),
        'regime_strength': regime_strength
    }

def prepare_indicator_data(prices, lookback=50, min_history=250):
    indicators_df = calculate_all_indicators(prices)
    
    selected_indicators = [
        'sma_short', 'sma_long', 'momentum_5', 'momentum_20',
        'volatility_10', 'mean_reversion', 'ma_spread',
        'synthetic_vix', 'regime_strength'
    ]
    
    indicator_subset = indicators_df[selected_indicators]
    
    X = []
    y = []
    
    for i in range(min_history, len(prices) - 1):
        current_indicators = indicator_subset.iloc[i].values
        target_return = (prices[i+1] - prices[i]) / prices[i]
        
        X.append(current_indicators)
        y.append(target_return)
    
    return np.array(X), np.array(y), selected_indicators