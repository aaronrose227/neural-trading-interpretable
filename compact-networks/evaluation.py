import numpy as np

def calculate_sharpe_ratio(returns, periods_per_year=252):
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return * periods_per_year) / (std_return * np.sqrt(periods_per_year))
    return sharpe