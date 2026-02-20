"""
Isolated unit test for the sklearn feature-name mismatch bug.

Reproduces the exact train→predict pattern used by AIStrategyEngine
without importing auto_trading.py (which pulls in Streamlit, plotly, etc.)

Run with:
    python -m pytest tests/test_signal_generation.py -v
or:
    python tests/test_signal_generation.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# ── Replicate _engineer_features exactly as written in auto_trading.py ───────

def _calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def _calculate_macd(prices):
    return prices.ewm(span=12).mean() - prices.ewm(span=26).mean()

def _calculate_bollinger_bands(prices, window=20):
    ma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return ma + 2 * std, ma - 2 * std

def _calculate_atr(data, window=14):
    hl = data['high'] - data['low']
    hc = np.abs(data['high'] - data['close'].shift())
    lc = np.abs(data['low'] - data['close'].shift())
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(window).mean()

def _calculate_obv(prices, volume):
    return (np.sign(prices.diff()) * volume).fillna(0).cumsum()

def _engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Exact copy of AIStrategyEngine._engineer_features."""
    features = data[['open', 'high', 'low', 'close', 'volume']].copy()
    features['rsi'] = _calculate_rsi(features['close'])
    features['macd'] = _calculate_macd(features['close'])
    features['bb_upper'], features['bb_lower'] = _calculate_bollinger_bands(features['close'])
    features['atr'] = _calculate_atr(features)
    features['obv'] = _calculate_obv(features['close'], features['volume'])
    features['price_change'] = features['close'].pct_change()
    features['volume_change'] = features['volume'].pct_change()
    features['high_low_ratio'] = features['high'] / features['low']
    features['close_open_ratio'] = features['close'] / features['open']
    for window in [5, 10, 20, 50]:
        features[f'ma_{window}'] = features['close'].rolling(window).mean()
        features[f'price_to_ma_{window}'] = features['close'] / features[f'ma_{window}']
    features['volatility_5'] = features['close'].rolling(5).std()
    features['volatility_20'] = features['close'].rolling(20).std()
    features['hour'] = pd.to_datetime(data.index).hour
    features['day_of_week'] = pd.to_datetime(data.index).dayofweek
    return features.fillna(0)


# ── OHLCV data factory ────────────────────────────────────────────────────────

def _make_ohlcv(n=300, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 150 + np.cumsum(rng.normal(0, 1, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high  = np.maximum(close, open_) * (1 + rng.uniform(0, 0.005, n))
    low   = np.minimum(close, open_) * (1 - rng.uniform(0, 0.005, n))
    volume = rng.integers(1_000_000, 50_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ── Simulate the BROKEN training + prediction path (old code) ────────────────

def _broken_train_and_predict(train_df, pred_df):
    """
    Replicates the OLD (broken) code path:
    - scaler fit on DataFrame X that includes open/high/low/close/volume
    - predict drops those same columns → sklearn raises feature-name error
    """
    scaler = StandardScaler()
    model  = RandomForestRegressor(n_estimators=10, random_state=42)

    # -- TRAIN --
    features = _engineer_features(train_df)
    features['target'] = features['close'].pct_change().shift(-1)
    features = features.dropna()
    X = features.drop(columns=['target'])          # includes OHLCV
    y = features['target']
    X_scaled = scaler.fit_transform(X)             # scaler sees OHLCV col names
    model.fit(X_scaled, y)

    # -- PREDICT (old broken way: drop OHLCV before transform) --
    pred_features = _engineer_features(pred_df)
    latest = pred_features.iloc[-1:].drop(
        columns=['open', 'high', 'low', 'close', 'volume'], errors='ignore'
    )
    return scaler.transform(latest)                # ← triggers sklearn error


def _fixed_train_and_predict(train_df, pred_df):
    """
    Replicates the FIXED code path:
    - scaler fit with .values (numpy) → no feature_names_in_ stored
    - predict selects same columns via feature_columns, then .values
    """
    scaler = StandardScaler()
    model  = RandomForestRegressor(n_estimators=10, random_state=42)

    # -- TRAIN --
    features = _engineer_features(train_df)
    features['target'] = features['close'].pct_change().shift(-1)
    features = features.dropna()
    X = features.drop(columns=['target'])          # includes OHLCV
    y = features['target']
    feature_columns = list(X.columns)             # ← stored for prediction

    X_scaled = scaler.fit_transform(X.values)     # ← numpy, no feature names stored
    model.fit(X_scaled, y)

    # -- PREDICT (fixed: use feature_columns + .values) --
    pred_features = _engineer_features(pred_df)
    latest = pred_features.iloc[-1:][feature_columns]
    return scaler.transform(latest.values)         # ← numpy, no name check


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFeatureNameBug:

    def test_broken_path_raises_sklearn_error(self):
        """
        The OLD code MUST reproduce the exact sklearn error the user reported.
        If this test fails it means the environment doesn't trigger the bug
        (old sklearn without feature-name validation) — warning only.
        """
        train_df = _make_ohlcv(300, seed=1)
        pred_df  = _make_ohlcv(100, seed=2)
        try:
            _broken_train_and_predict(train_df, pred_df)
            # sklearn < 1.0 won't raise — just warn
            import warnings
            warnings.warn(
                "sklearn did NOT raise the feature-name error. "
                "This environment may have sklearn < 1.0 where the bug is silent.",
                UserWarning,
            )
        except ValueError as e:
            assert "feature names" in str(e).lower() or "feature_names" in str(e).lower(), (
                f"Unexpected ValueError: {e}"
            )

    def test_fixed_path_does_not_raise(self):
        """The FIXED code must never raise a feature-name error."""
        train_df = _make_ohlcv(300, seed=10)
        pred_df  = _make_ohlcv(100, seed=20)
        # Must not raise
        result = _fixed_train_and_predict(train_df, pred_df)
        assert result.shape[0] == 1, f"Expected 1 row, got shape {result.shape}"

    def test_fixed_output_shape_matches_feature_count(self):
        """Scaled output width must equal the number of engineered features."""
        train_df = _make_ohlcv(300, seed=30)
        pred_df  = _make_ohlcv(100, seed=40)

        features = _engineer_features(train_df)
        features['target'] = features['close'].pct_change().shift(-1)
        features = features.dropna()
        X = features.drop(columns=['target'])
        expected_n_features = len(X.columns)

        result = _fixed_train_and_predict(train_df, pred_df)
        assert result.shape == (1, expected_n_features), (
            f"Shape mismatch: got {result.shape}, expected (1, {expected_n_features})"
        )

    def test_fixed_path_multiple_symbols(self):
        """Same engine retrained on a second symbol must still work."""
        for seed in [10, 20, 30]:
            train_df = _make_ohlcv(300, seed=seed)
            pred_df  = _make_ohlcv(100, seed=seed + 100)
            result = _fixed_train_and_predict(train_df, pred_df)
            assert result is not None

    def test_ohlcv_included_in_feature_columns(self):
        """feature_columns (training column set) must include all 5 OHLCV cols."""
        train_df = _make_ohlcv(300)
        features = _engineer_features(train_df)
        features['target'] = features['close'].pct_change().shift(-1)
        features = features.dropna()
        X = features.drop(columns=['target'])
        feature_columns = list(X.columns)
        for col in ('open', 'high', 'low', 'close', 'volume'):
            assert col in feature_columns, f"'{col}' missing from feature_columns"


# ── Allow running directly ────────────────────────────────────────────────────
if __name__ == "__main__":
    import traceback
    tests = TestFeatureNameBug()
    cases = [
        tests.test_broken_path_raises_sklearn_error,
        tests.test_fixed_path_does_not_raise,
        tests.test_fixed_output_shape_matches_feature_count,
        tests.test_fixed_path_multiple_symbols,
        tests.test_ohlcv_included_in_feature_columns,
    ]
    passed = failed = 0
    for fn in cases:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(failed)
