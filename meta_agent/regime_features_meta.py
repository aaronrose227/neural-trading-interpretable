"""
regime_features_meta.py
Purpose: Lightweight regime feature extractor + scaler, adapted from your RegimeDetector.
We only use features & scaling; labels/classification aren't required by the allocator.
"""

import numpy as np
from sklearn.preprocessing import RobustScaler
import xgboost as xgb  # kept for parity; not used for classification here


class RegimeFeatureExtractor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.is_fitted = False

    def extract_regime_features(self, prices, lookback=100):
        if len(prices) < lookback:
            return np.array([])
        feats = []
        for i in range(lookback, len(prices)):
            window = prices[i - lookback:i]
            rets = np.diff(window) / window[:-1]
            vol = np.std(rets) * np.sqrt(252)
            prices_norm = (window - window[0]) / window[0]
            trend_strength = abs(prices_norm[-1])
            ma20 = np.mean(window[-20:])
            ma50 = np.mean(window[-50:]) if len(window) >= 50 else ma20
            mean_rev = abs(window[-1] - ma50) / (np.std(window) + 1e-8)
            if len(rets) > 1:
                autocorr = np.corrcoef(rets[:-1], rets[1:])[0, 1]
                autocorr = 0 if np.isnan(autocorr) else autocorr
            else:
                autocorr = 0
            short_rets = rets[-10:] if len(rets) >= 10 else rets
            if len(short_rets) >= 2:
                mom_consistency = np.sum(np.sign(short_rets[:-1]) == np.sign(short_rets[1:])) / (len(short_rets) - 1)
            else:
                mom_consistency = 0
            feats.append([vol, trend_strength, mean_rev, mom_consistency, autocorr])
        return np.array(feats)

    def fit_scaler(self, prices, lookback=100):
        X = self.extract_regime_features(prices, lookback)
        if X.size == 0:
            return False
        self.scaler.fit(X)
        self.is_fitted = True
        return True

    def transform_recent(self, prices, lookback=100):
        X = self.extract_regime_features(prices, lookback)
        if X.size == 0:
            return np.array([])
        if not self.is_fitted:
            # Fit-once default behavior
            self.fit_scaler(prices, lookback)
        return self.scaler.transform(X)
