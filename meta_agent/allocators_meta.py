"""
allocators_meta.py
Purpose: XGBoost-based allocator (meta-agent) that maps regime features to
weights over frozen specialists. Supervised blender trained on validation data.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import xgboost as xgb

from regime_features_meta import RegimeFeatureExtractor


@dataclass
class AllocatorConfig:
    forward_horizon: int = 20
    min_samples: int = 200
    normalize_nonnegative: bool = True


class XGBAllocator:
    def __init__(self, specialist_order: List[str], cfg: AllocatorConfig = AllocatorConfig()):
        self.order = specialist_order
        self.cfg = cfg
        self.fx = RegimeFeatureExtractor()
        self.models: Dict[str, xgb.XGBRegressor] = {}

    def _features_over_series(self, prices: np.ndarray, lookback: int = 100) -> np.ndarray:
        X = self.fx.extract_regime_features(prices, lookback=lookback)
        if X.size == 0:
            return X
        if not self.fx.is_fitted:
            self.fx.fit_scaler(prices, lookback)
        return self.fx.scaler.transform(X)

    def _build_labels(self, prices: np.ndarray, specialist_returns: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        lookback = 100
        X = self._features_over_series(prices, lookback=lookback)
        if X.size == 0:
            return np.array([]), np.array([])
        T = len(X)
        H = self.cfg.forward_horizon
        aligned = {k: v[:T] for k, v in specialist_returns.items()}
        rows, targets = [], []
        for t in range(0, T - H):
            f = np.array([aligned[k][t:t + H].sum() for k in self.order], dtype=float)
            if self.cfg.normalize_nonnegative:
                f = np.maximum(f, 0.0)
                f = f / f.sum() if f.sum() > 0 else np.ones_like(f) / len(f)
            rows.append(X[t])
            targets.append(f)
        return np.array(rows), np.array(targets)

    def fit(self, train_prices: np.ndarray, val_prices: np.ndarray, specialist_returns_val: Dict[str, np.ndarray]) -> bool:
        self.fx.fit_scaler(train_prices, lookback=100)
        X, Y = self._build_labels(val_prices, specialist_returns_val)
        if len(X) < self.cfg.min_samples:
            print("Allocator: insufficient samples on validation window.")
            return False
        self.models = {}
        for j, kind in enumerate(self.order):
            y = Y[:, j]
            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42
            )
            model.fit(X, y)
            self.models[kind] = model
        return True

    def predict_weights(self, recent_prices: np.ndarray) -> np.ndarray:
        feats = self._features_over_series(recent_prices, lookback=100)
        if feats.size == 0:
            return np.ones(len(self.order)) / len(self.order)
        x = feats[-1].reshape(1, -1)
        preds = np.array([self.models[k].predict(x)[0] if k in self.models else 1.0 for k in self.order])
        preds = np.maximum(preds, 0.0)
        return preds / preds.sum() if preds.sum() > 0 else np.ones(len(self.order)) / len(self.order)
