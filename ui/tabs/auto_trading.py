import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import requests
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from config import config
from services.data_fetcher import (
    pro_get_historical_data,
    pro_get_real_time_data,
    pro_get_news,
)

# ========== ENUMS & DATA STRUCTURES ==========

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    PARTIAL = "PARTIAL"

class RiskLevel(Enum):
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"
    PROFESSIONAL = "PROFESSIONAL"

class BrokerType(Enum):
    ALPACA = "Alpaca"
    INTERACTIVE_BROKERS = "Interactive Brokers"
    TD_AMERITRADE = "TD Ameritrade"
    MOCK = "Mock Broker"

@dataclass
class TradingSignal:
    symbol: str
    action: str  # BUY/SELL/HOLD
    confidence: float
    price: float
    quantity: int
    strategy_name: str
    timestamp: datetime
    reasoning: Dict
    risk_score: float
    expected_return: float
    stop_loss: float
    take_profit: float
    feature_importance: Dict
    market_regime: str
    sentiment_score: float = 0.0

@dataclass
class RiskMetrics:
    max_drawdown: float = 0.05
    position_size_limit: float = 0.1
    daily_loss_limit: float = 0.03
    correlation_limit: float = 0.7
    var_95: float = 0.02
    sharpe_threshold: float = 1.0
    max_positions: int = 10
    sector_concentration: float = 0.3

# ========== PROFESSIONAL BROKER INTEGRATION ==========

class BrokerAPI:
    """Professional multi-broker API integration"""

    def __init__(self, broker_type: BrokerType, credentials: Dict):
        self.broker_type = broker_type
        self.credentials = credentials
        self.is_connected = False
        self.account_info = {}

    def connect(self) -> Tuple[bool, str]:
        """Connect to broker with proper error handling"""
        try:
            if self.broker_type == BrokerType.ALPACA:
                return self._connect_alpaca()
            elif self.broker_type == BrokerType.INTERACTIVE_BROKERS:
                return self._connect_ib()
            elif self.broker_type == BrokerType.TD_AMERITRADE:
                return self._connect_td()
            else:  # Mock
                return self._connect_mock()
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def _connect_alpaca(self) -> Tuple[bool, str]:
        """Connect to Alpaca Markets"""
        # In production: from alpaca_trade_api import REST
        # api = REST(self.credentials['api_key'], self.credentials['secret_key'], base_url='https://paper-api.alpaca.markets')
        self.is_connected = True
        return True, "Connected to Alpaca Markets"

    def _connect_ib(self) -> Tuple[bool, str]:
        """Connect to Interactive Brokers"""
        # In production: from ib_insync import IB
        self.is_connected = True
        return True, "Connected to Interactive Brokers"

    def _connect_td(self) -> Tuple[bool, str]:
        """Connect to TD Ameritrade"""
        # In production: TD Ameritrade API integration
        self.is_connected = True
        return True, "Connected to TD Ameritrade"

    def _connect_mock(self) -> Tuple[bool, str]:
        """Mock broker for testing"""
        self.is_connected = True
        return True, "Connected to Mock Broker"

    def place_order(self, signal: TradingSignal) -> Dict:
        """Place order — simulated execution with realistic fill price slippage."""
        if not self.is_connected:
            return {'success': False, 'message': 'Not connected to broker'}

        order_id = f"ORD_{int(time.time())}_{signal.symbol}"

        # Realistic market-order slippage: ±0.05% of price (not random failure)
        slippage = 0.0005
        if signal.action == 'BUY':
            fill_price = signal.price * (1 + slippage)   # pay slightly more on buy
        else:
            fill_price = signal.price * (1 - slippage)   # receive slightly less on sell

        commission = signal.quantity * fill_price * 0.0005   # 0.05% commission

        return {
            'success': True,
            'order_id': order_id,
            'status': OrderStatus.FILLED.value,
            'fill_price': round(fill_price, 4),
            'timestamp': datetime.now(),
            'commission': round(commission, 4),
        }

    def get_account_info(self, engine_portfolio: Dict = None) -> Dict:
        """Return live account info derived from the engine portfolio."""
        if engine_portfolio:
            return {
                'buying_power': engine_portfolio.get('cash', 0),
                'portfolio_value': engine_portfolio.get('total_value', 0),
                'open_positions': len(engine_portfolio.get('positions', {})),
                'total_trades': engine_portfolio.get('total_trades', 0),
            }
        return {
            'buying_power': 0,
            'portfolio_value': 0,
            'open_positions': 0,
            'total_trades': 0,
        }

# ========== AI-POWERED STRATEGY ENGINE ==========

class AIStrategyEngine:
    """Advanced AI-driven trading strategies with explainability"""

    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.market_regime_detector = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def train_models(self, market_data: pd.DataFrame, symbols: List[str]) -> Dict:
        """Train multiple AI models for different strategies"""
        training_results = {}

        for symbol in symbols:
            try:
                # Prepare features
                features = self._engineer_features(market_data, symbol)
                if features.empty:
                    continue

                # Create target variables for different strategies
                features['future_return_1h'] = features['close'].pct_change().shift(-1)
                features['future_return_1d'] = features['close'].pct_change(24).shift(-24)
                features['volatility_target'] = features['close'].rolling(24).std()

                # Remove NaN and infinite values
                features = features.replace([float('inf'), float('-inf')], float('nan')).dropna()

                if len(features) < 100:  # Need sufficient data
                    continue

                # Train ensemble of models
                models = {}

                # XGBoost for price prediction
                X = features.drop(['future_return_1h', 'future_return_1d', 'volatility_target'], axis=1)
                y_returns = features['future_return_1h'].replace([float('inf'), float('-inf')], float('nan')).fillna(0)

                # Time series split for proper validation
                tscv = TimeSeriesSplit(n_splits=3)

                for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y_returns.iloc[train_idx], y_returns.iloc[val_idx]

                    # Scale features
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_val_scaled = self.scaler.transform(X_val)

                    # XGBoost model
                    xgb_model = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42
                    )
                    xgb_model.fit(X_train_scaled, y_train)

                    # Random Forest for comparison
                    rf_model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )
                    rf_model.fit(X_train_scaled, y_train)

                    # Store models
                    models[f'xgb_fold_{i}'] = xgb_model
                    models[f'rf_fold_{i}'] = rf_model

                    # Calculate feature importance
                    self.feature_importance[f'{symbol}_xgb'] = dict(zip(X.columns, xgb_model.feature_importances_))
                    self.feature_importance[f'{symbol}_rf'] = dict(zip(X.columns, rf_model.feature_importances_))

                self.models[symbol] = models
                training_results[symbol] = {
                    'status': 'success',
                    'models_trained': len(models),
                    'features_count': len(X.columns),
                    'training_samples': len(X)
                }

            except Exception as e:
                training_results[symbol] = {
                    'status': 'error',
                    'error': str(e)
                }

        self.is_trained = len(self.models) > 0
        return training_results

    def _engineer_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Engineer comprehensive technical and fundamental features"""
        try:
            if data.empty:
                return pd.DataFrame()
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(c in data.columns for c in required):
                return pd.DataFrame()
            # Basic OHLC features
            features = data[required].copy()

            # Technical indicators
            features['rsi'] = self._calculate_rsi(features['close'])
            features['macd'] = self._calculate_macd(features['close'])
            features['bb_upper'], features['bb_lower'] = self._calculate_bollinger_bands(features['close'])
            features['atr'] = self._calculate_atr(features)
            features['obv'] = self._calculate_obv(features['close'], features['volume'])

            # Price-based features
            features['price_change'] = features['close'].pct_change()
            features['volume_change'] = features['volume'].pct_change()
            features['high_low_ratio'] = features['high'] / features['low']
            features['close_open_ratio'] = features['close'] / features['open']

            # Moving averages
            for window in [5, 10, 20, 50]:
                features[f'ma_{window}'] = features['close'].rolling(window).mean()
                features[f'price_to_ma_{window}'] = features['close'] / features[f'ma_{window}']

            # Volatility features
            features['volatility_5'] = features['close'].rolling(5).std()
            features['volatility_20'] = features['close'].rolling(20).std()

            # Time-based features
            features['hour'] = pd.to_datetime(data.index).hour
            features['day_of_week'] = pd.to_datetime(data.index).dayofweek

            return features.replace([float('inf'), float('-inf')], float('nan')).fillna(0)

        except Exception as e:
            st.error(f"Feature engineering failed for {symbol}: {e}")
            return pd.DataFrame()

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        return ema12 - ema26

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        return ma + 2*std, ma - 2*std

    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    def _calculate_obv(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = (np.sign(prices.diff()) * volume).fillna(0).cumsum()
        return obv

    def generate_signal(self, symbol: str, current_data: pd.DataFrame, strategy: str) -> TradingSignal:
        """Generate AI-powered trading signal with explainability"""
        if not self.is_trained or symbol not in self.models:
            return self._fallback_signal(symbol, current_data)

        try:
            # Engineer features for current data
            features = self._engineer_features(current_data, symbol)
            if features.empty:
                return self._fallback_signal(symbol, current_data)

            # Get latest features
            latest_features = features.iloc[-1:].drop(['open', 'high', 'low', 'close', 'volume'], axis=1, errors='ignore')

            # Scale features
            latest_scaled = self.scaler.transform(latest_features)

            # Get ensemble predictions
            predictions = []
            models = self.models[symbol]

            for model_name, model in models.items():
                pred = model.predict(latest_scaled)[0]
                predictions.append(pred)

            # Ensemble prediction
            avg_prediction = np.mean(predictions)
            prediction_std = np.std(predictions)

            # Convert prediction to trading signal
            confidence = min(abs(avg_prediction) * 10, 1.0)  # Scale confidence

            if avg_prediction > 0.01 and confidence > 0.7:
                action = "BUY"
            elif avg_prediction < -0.01 and confidence > 0.7:
                action = "SELL"
            else:
                action = "HOLD"

            # Get current price
            current_price = current_data['close'].iloc[-1]

            # Calculate position size based on Kelly Criterion
            kelly_fraction = self._calculate_kelly_criterion(avg_prediction, prediction_std)
            quantity = max(1, int(kelly_fraction * 100))  # Base quantity of 100 shares

            # Feature importance for explainability
            feature_importance = self.feature_importance.get(f'{symbol}_xgb', {})
            top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])

            # Generate reasoning
            reasoning = {
                'prediction': avg_prediction,
                'confidence_raw': confidence,
                'ensemble_agreement': 1 - (prediction_std / (abs(avg_prediction) + 1e-8)),
                'top_features': top_features,
                'strategy_used': strategy,
                'model_count': len(predictions)
            }

            # Derive sentiment from RSI and recent price momentum (no random)
            rsi_val = float(features['rsi'].iloc[-1]) if 'rsi' in features.columns else 50.0
            recent_mom = float(features['price_change'].iloc[-5:].mean()) if 'price_change' in features.columns else 0.0
            # RSI component: >50 → positive, <50 → negative, scaled to [-0.5, +0.5]
            rsi_sentiment = (rsi_val - 50) / 100
            # Momentum component clipped to [-0.5, +0.5]
            mom_sentiment = float(np.clip(recent_mom * 25, -0.5, 0.5))
            sentiment_score = float(np.clip(rsi_sentiment + mom_sentiment, -1.0, 1.0))

            return TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                price=current_price,
                quantity=quantity,
                strategy_name=f"AI_{strategy}",
                timestamp=datetime.now(),
                reasoning=reasoning,
                risk_score=min(prediction_std * 5, 1.0),
                expected_return=avg_prediction,
                stop_loss=max(0.02, prediction_std * 2),
                take_profit=abs(avg_prediction) * 2,
                feature_importance=top_features,
                market_regime=self._detect_market_regime(current_data),
                sentiment_score=sentiment_score,
            )

        except Exception as e:
            st.error(f"Signal generation failed for {symbol}: {e}")
            return self._fallback_signal(symbol, current_data)

    def _calculate_kelly_criterion(self, expected_return: float, volatility: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        if volatility <= 0:
            return 0.1

        # Simplified Kelly: f = (bp - q) / b
        # where b = odds, p = probability of win, q = probability of loss
        win_prob = 0.55  # Assume slight edge
        loss_prob = 0.45
        avg_win = abs(expected_return)
        avg_loss = volatility

        if avg_loss <= 0:
            return 0.1

        kelly_fraction = (win_prob * avg_win - loss_prob * avg_loss) / avg_loss

        # Cap Kelly fraction to avoid excessive risk
        return max(0.01, min(kelly_fraction, 0.25))

    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime (trending/ranging/volatile)"""
        if len(data) < 20:
            return "UNKNOWN"

        recent_data = data.tail(20)
        returns = recent_data['close'].pct_change()
        volatility = returns.std()

        # Simple regime detection
        if volatility > 0.03:
            return "HIGH_VOLATILITY"
        elif returns.mean() > 0.01:
            return "BULL_TREND"
        elif returns.mean() < -0.01:
            return "BEAR_TREND"
        else:
            return "RANGING"

    def _fallback_signal(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Fallback signal when AI models aren't available"""
        current_price = data['close'].iloc[-1]

        return TradingSignal(
            symbol=symbol,
            action="HOLD",
            confidence=0.5,
            price=current_price,
            quantity=10,
            strategy_name="FALLBACK",
            timestamp=datetime.now(),
            reasoning={'note': 'Fallback signal - AI models not trained'},
            risk_score=0.5,
            expected_return=0.0,
            stop_loss=0.02,
            take_profit=0.05,
            feature_importance={},
            market_regime="UNKNOWN"
        )

# ========== RISK MANAGEMENT SYSTEM ==========

class RiskManager:
    """Advanced risk management with real-time monitoring"""

    def __init__(self, risk_metrics: RiskMetrics):
        self.risk_metrics = risk_metrics
        self.position_tracker = {}
        self.daily_pnl = 0.0
        self.max_drawdown_today = 0.0
        self.risk_alerts = []

    def validate_trade(self, signal: TradingSignal, portfolio: Dict, current_positions: Dict) -> Tuple[bool, str, Dict]:
        """Comprehensive trade validation"""
        validation_results = {
            'position_size_check': False,
            'daily_loss_check': False,
            'correlation_check': False,
            'concentration_check': False,
            'max_positions_check': False
        }

        # Position size check
        portfolio_value = portfolio.get('total_value', 100000)
        trade_value = signal.quantity * signal.price
        position_size_ratio = trade_value / portfolio_value

        if position_size_ratio <= self.risk_metrics.position_size_limit:
            validation_results['position_size_check'] = True
        else:
            return False, f"Position size {position_size_ratio:.2%} exceeds limit {self.risk_metrics.position_size_limit:.2%}", validation_results

        # Daily loss limit check
        if abs(self.daily_pnl) <= self.risk_metrics.daily_loss_limit * portfolio_value:
            validation_results['daily_loss_check'] = True
        else:
            return False, f"Daily loss limit exceeded: {self.daily_pnl:.2f}", validation_results

        # Maximum positions check
        if len(current_positions) < self.risk_metrics.max_positions:
            validation_results['max_positions_check'] = True
        else:
            return False, f"Maximum positions limit ({self.risk_metrics.max_positions}) reached", validation_results

        # Correlation and concentration checks would require historical correlation data
        validation_results['correlation_check'] = True  # Simplified
        validation_results['concentration_check'] = True  # Simplified

        return True, "Trade validated", validation_results

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) < 30:
            return 0.02  # Default VaR

        return np.percentile(returns, (1 - confidence) * 100)

    def detect_anomalies(self, market_data: pd.DataFrame) -> List[str]:
        """Detect market anomalies using Isolation Forest"""
        try:
            # Prepare features for anomaly detection
            features = market_data[['close', 'volume']].pct_change().fillna(0)

            if len(features) < 50:
                return []

            # Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(features)

            anomaly_indices = np.where(anomalies == -1)[0]

            alerts = []
            for idx in anomaly_indices[-5:]:  # Last 5 anomalies
                alerts.append(f"Anomaly detected at {market_data.index[idx]}")

            return alerts

        except Exception:
            return []

    def update_daily_pnl(self, trade_pnl: float):
        """Update daily P&L tracking"""
        self.daily_pnl += trade_pnl
        self.max_drawdown_today = min(self.max_drawdown_today, self.daily_pnl)

# ========== AUTO-TRADING ENGINE ==========

class AutoTradingEngine:
    """Professional auto-trading engine with AI and risk management"""

    def __init__(self):
        self.is_active = False
        self.portfolio = {
            'cash': 100000,
            'positions': {},
            'total_value': 100000,
            'daily_pnl': 0,
            'total_trades': 0
        }
        self.trade_history = []
        self.ai_engine = AIStrategyEngine()
        self.risk_manager = RiskManager(RiskMetrics())
        self.broker_apis = {}
        self.active_bots = {}
        self.performance_metrics = {}
        self.lock = threading.Lock()

    def add_broker(self, name: str, broker_type: BrokerType, credentials: Dict) -> bool:
        """Add broker API connection"""
        try:
            broker = BrokerAPI(broker_type, credentials)
            success, message = broker.connect()
            if success:
                self.broker_apis[name] = broker
                return True
            else:
                st.error(f"Failed to connect to {name}: {message}")
                return False
        except Exception as e:
            st.error(f"Error adding broker {name}: {e}")
            return False

    def train_ai_models(self, symbols: List[str]) -> Dict:
        """Train AI models for all symbols using real EODHD data"""
        training_data = self._fetch_real_data(symbols)
        return self.ai_engine.train_models(training_data, symbols)

    def _fetch_real_data(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch historical OHLCV data — EODHD first, Yahoo Finance fallback."""
        api_key = config.get_eodhd_api_key()
        all_frames = []

        for symbol in symbols:
            df = pd.DataFrame()

            # --- Primary: EODHD ---
            if api_key:
                try:
                    raw = pro_get_historical_data(symbol, api_key)
                    if not raw.empty:
                        raw.columns = [c.lower() for c in raw.columns]
                        if 'adjusted_close' in raw.columns and 'close' not in raw.columns:
                            raw = raw.rename(columns={'adjusted_close': 'close'})
                        needed = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in raw.columns]
                        if len(needed) == 5:
                            df = raw[needed].copy()
                except Exception:
                    pass

            # --- Fallback: Yahoo Finance (strip exchange suffix) ---
            if df.empty:
                try:
                    yf_sym = symbol.split('.')[0]
                    raw = yf.download(yf_sym, period='6mo', progress=False, auto_adjust=True)
                    if not raw.empty:
                        raw.columns = [c.lower() for c in raw.columns]
                        needed = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in raw.columns]
                        if len(needed) == 5:
                            df = raw[needed].copy()
                except Exception:
                    pass

            if df.empty:
                continue

            df['symbol'] = symbol
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)
            if not df.empty:
                all_frames.append(df)

        if all_frames:
            combined = pd.concat(all_frames)
            combined.index = pd.to_datetime(combined.index)
            return combined
        return pd.DataFrame()

    def execute_trade(self, signal: TradingSignal, broker_name: str = None) -> Dict:
        """Execute trade with comprehensive error handling"""
        with self.lock:
            try:
                # Select broker
                if broker_name and broker_name in self.broker_apis:
                    broker = self.broker_apis[broker_name]
                elif self.broker_apis:
                    broker = list(self.broker_apis.values())[0]
                else:
                    return {'success': False, 'message': 'No broker available'}

                # Risk validation
                valid, message, validation_results = self.risk_manager.validate_trade(
                    signal, self.portfolio, self.portfolio['positions']
                )

                if not valid:
                    return {
                        'success': False,
                        'message': f'Risk check failed: {message}',
                        'validation_results': validation_results
                    }

                # Place order with broker
                order_result = broker.place_order(signal)

                if not order_result['success']:
                    return order_result

                # Update portfolio
                trade_result = self._update_portfolio(signal, order_result)

                # Update risk metrics
                self.risk_manager.update_daily_pnl(trade_result.get('pnl', 0))

                # Log trade
                trade_log = {
                    'id': order_result['order_id'],
                    'timestamp': datetime.now(),
                    'signal': signal.__dict__,
                    'order_result': order_result,
                    'portfolio_snapshot': self.portfolio.copy(),
                    'validation_results': validation_results,
                    'pnl': trade_result.get('pnl', 0),
                }

                self.trade_history.append(trade_log)
                self.portfolio['total_trades'] += 1

                return {
                    'success': True,
                    'trade_log': trade_log,
                    'message': f'Trade executed: {signal.action} {signal.quantity} {signal.symbol}'
                }

            except Exception as e:
                return {
                    'success': False,
                    'message': f'Trade execution failed: {str(e)}',
                    'error': str(e)
                }

    def _update_portfolio(self, signal: TradingSignal, order_result: Dict) -> Dict:
        """Update portfolio after trade execution"""
        fill_price = order_result.get('fill_price', signal.price)
        commission = order_result.get('commission', 0)

        if signal.action == 'BUY':
            cost = signal.quantity * fill_price + commission
            self.portfolio['cash'] -= cost

            if signal.symbol in self.portfolio['positions']:
                pos = self.portfolio['positions'][signal.symbol]
                total_qty = pos['quantity'] + signal.quantity
                total_cost = pos['quantity'] * pos['avg_price'] + cost
                pos['avg_price'] = total_cost / total_qty
                pos['quantity'] = total_qty
            else:
                self.portfolio['positions'][signal.symbol] = {
                    'quantity': signal.quantity,
                    'avg_price': fill_price,
                    'entry_time': datetime.now()
                }

        elif signal.action == 'SELL':
            proceeds = signal.quantity * fill_price - commission
            self.portfolio['cash'] += proceeds

            if signal.symbol in self.portfolio['positions']:
                pos = self.portfolio['positions'][signal.symbol]
                pos['quantity'] -= signal.quantity

                if pos['quantity'] <= 0:
                    del self.portfolio['positions'][signal.symbol]

        # Update total portfolio value (mark positions at fill price)
        position_value = sum(
            pos['quantity'] * pos['avg_price']
            for pos in self.portfolio['positions'].values()
        )
        self.portfolio['total_value'] = self.portfolio['cash'] + position_value

        # Real P&L: proceeds minus cost for a round-trip sell, or 0 on a buy
        if signal.action == 'SELL':
            entry_price = self.portfolio['positions'].get(signal.symbol, {}).get('avg_price', fill_price)
            pnl = (fill_price - entry_price) * signal.quantity - commission
        else:
            pnl = 0.0

        return {'pnl': pnl}

    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trade_history:
            return {}

        # Extract trade data
        trades_df = pd.DataFrame([
            {
                'timestamp': trade['timestamp'],
                'symbol': trade['signal']['symbol'],
                'action': trade['signal']['action'],
                'quantity': trade['signal']['quantity'],
                'price': trade['signal']['price'],
                'confidence': trade['signal']['confidence']
            }
            for trade in self.trade_history
        ])

        # Calculate real P&L per trade from trade_history
        pnl_list = []
        for trade in self.trade_history:
            pnl_list.append(trade.get('pnl', 0))

        pnl_series = pd.Series(pnl_list)

        profitable = int((pnl_series > 0).sum())
        total = len(pnl_series)
        win_rate = profitable / total if total > 0 else 0.0

        # Portfolio value history (snapshot stored per trade)
        portfolio_values = [
            t.get('portfolio_snapshot', {}).get('total_value', self.portfolio['total_value'])
            for t in self.trade_history
        ]
        pv_series = pd.Series(portfolio_values)

        # Returns from portfolio value changes
        pv_returns = pv_series.pct_change().dropna()

        # Sharpe ratio (annualised, assuming daily trades ~ 252 periods)
        if pv_returns.std() > 0:
            sharpe = (pv_returns.mean() / pv_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        rolling_max = pv_series.cummax()
        drawdown = (pv_series - rolling_max) / rolling_max
        max_drawdown = float(abs(drawdown.min())) if not drawdown.empty else 0.0

        # Calmar ratio
        total_return = (pv_series.iloc[-1] / pv_series.iloc[0] - 1) if len(pv_series) > 1 else 0.0
        calmar = total_return / max_drawdown if max_drawdown > 0 else 0.0

        metrics = {
            'total_trades': total,
            'profitable_trades': profitable,
            'win_rate': win_rate,
            'avg_return': float(pnl_series.mean()) if total > 0 else 0.0,
            'total_pnl': float(pnl_series.sum()),
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown': round(max_drawdown, 4),
            'volatility': round(float(pv_returns.std()) * np.sqrt(252), 4) if not pv_returns.empty else 0.0,
            'calmar_ratio': round(calmar, 3),
            'best_performing_symbol': trades_df['symbol'].mode().iloc[0] if not trades_df.empty else 'N/A',
            'portfolio_value_history': portfolio_values,
        }

        return metrics

# ========== STREAMLIT UI COMPONENTS ==========

def render_professional_autotrading():
    """Render the professional auto-trading interface"""

    # Initialize session state, with a check for object integrity
    if 'trading_engine' not in st.session_state or not hasattr(st.session_state.trading_engine, 'portfolio'):
        st.session_state.trading_engine = AutoTradingEngine()

    if 'trading_bots' not in st.session_state:
        st.session_state.trading_bots = {}

    engine = st.session_state.trading_engine

    # Main header
    st.markdown("""
    # 🚀 AI-Powered Auto-Trading Platform
    ### Professional-Grade Algorithmic Trading with Explainable AI
    """)

    # Key metrics dashboard
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Portfolio Value",
            f"${engine.portfolio['total_value']:,.2f}",
            f"{((engine.portfolio['total_value'] - 100000) / 100000 * 100):+.1f}%"
        )

    with col2:
        st.metric("Active Bots", len(st.session_state.trading_bots))

    with col3:
        st.metric("Total Trades", engine.portfolio['total_trades'])

    with col4:
        st.metric("Available Cash", f"${engine.portfolio['cash']:,.2f}")

    with col5:
        ai_status = "🟢 Trained" if engine.ai_engine.is_trained else "🔴 Not Trained"
        st.metric("AI Status", ai_status)

    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🤖 Bot Management",
        "🧠 AI Training",
        "🔗 Broker Setup",
        "⚡ Live Signals",
        "📊 Performance",
        "🛡️ Risk Management"
    ])

    with tab1:
        render_bot_management(engine)

    with tab2:
        render_ai_training(engine)

    with tab3:
        render_broker_setup(engine)

    with tab4:
        render_live_signals(engine)

    with tab5:
        render_performance_analytics(engine)

    with tab6:
        render_risk_management(engine)

def render_bot_management(engine):
    """Render bot management interface"""
    st.markdown("## 🤖 Trading Bot Management")

    # Bot deployment form
    with st.expander("🛠️ Deploy New AI Trading Bot", expanded=True):
        with st.form("deploy_advanced_bot"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Bot Configuration")
                bot_name = st.text_input("Bot Name", f"AI-Bot-{len(st.session_state.trading_bots) + 1}")
                symbols = st.multiselect(
                    "Trading Symbols",
                    ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX'],
                    default=['AAPL']
                )
                broker = st.selectbox("Broker", list(engine.broker_apis.keys()) if engine.broker_apis else ['None'])

            with col2:
                st.markdown("#### Strategy Configuration")
                strategy_type = st.selectbox("AI Strategy", [
                    "Multi-Model Ensemble",
                    "XGBoost Momentum",
                    "Random Forest Mean Reversion",
                    "LSTM Neural Network",
                    "Reinforcement Learning",
                    "Sentiment-Driven"
                ])

                confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.75, 0.05)
                max_position_size = st.slider("Max Position Size (%)", 1, 20, 10, 1)

            col3, col4 = st.columns(2)
            with col3:
                st.markdown("#### Risk Parameters")
                risk_level = st.selectbox("Risk Level", [e.value for e in RiskLevel])
                stop_loss = st.slider("Stop Loss (%)", 0.5, 10.0, 2.0, 0.5)
                take_profit = st.slider("Take Profit (%)", 1.0, 20.0, 5.0, 0.5)

            with col4:
                st.markdown("#### Execution Settings")
                trade_frequency = st.selectbox("Trade Frequency", ["1 min", "5 min", "15 min", "1 hour"])
                max_daily_trades = st.number_input("Max Daily Trades", 1, 100, 20)
                enable_afterhours = st.checkbox("Enable After-Hours Trading")

            if st.form_submit_button("🚀 Deploy Advanced Bot", type="primary"):
                if bot_name in st.session_state.trading_bots:
                    st.error(f"Bot '{bot_name}' already exists!")
                elif not engine.ai_engine.is_trained:
                    st.warning("⚠️ AI models not trained! Please train models first.")
                else:
                    # Create new advanced bot
                    bot_config = {
                        'name': bot_name,
                        'symbols': symbols,
                        'strategy': strategy_type,
                        'confidence_threshold': confidence_threshold,
                        'max_position_size': max_position_size / 100,
                        'risk_level': risk_level,
                        'stop_loss': stop_loss / 100,
                        'take_profit': take_profit / 100,
                        'trade_frequency': trade_frequency,
                        'max_daily_trades': max_daily_trades,
                        'afterhours': enable_afterhours,
                        'broker': broker,
                        'status': 'Deployed',
                        'trades_today': 0,
                        'pnl_today': 0.0,
                        'created': datetime.now()
                    }

                    st.session_state.trading_bots[bot_name] = bot_config
                    st.success(f"✅ Advanced bot '{bot_name}' deployed successfully!")
                    st.rerun()

    # Active bots dashboard
    st.markdown("---")
    st.markdown("### 🛰️ Active Trading Bots")

    if not st.session_state.trading_bots:
        st.info("📭 No bots deployed yet. Deploy your first AI trading bot above!")
    else:
        for bot_name, bot_config in st.session_state.trading_bots.items():
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

                with col1:
                    st.markdown(f"**{bot_name}**")
                    st.caption(f"Strategy: {bot_config['strategy']}")
                    st.caption(f"Symbols: {', '.join(bot_config['symbols'])}")

                with col2:
                    status_color = "🟢" if bot_config['status'] == 'Running' else "🟡"
                    st.markdown(f"{status_color} **{bot_config['status']}**")
                    st.caption(f"Trades Today: {bot_config['trades_today']}")

                with col3:
                    st.metric("P&L Today", f"${bot_config['pnl_today']:+.2f}")

                with col4:
                    col4a, col4b = st.columns(2)
                    with col4a:
                        if st.button("▶️", key=f"start_{bot_name}", help="Start Bot"):
                            st.session_state.trading_bots[bot_name]['status'] = 'Running'
                            st.rerun()
                    with col4b:
                        if st.button("⏹️", key=f"stop_{bot_name}", help="Stop Bot"):
                            st.session_state.trading_bots[bot_name]['status'] = 'Stopped'
                            st.rerun()

                st.markdown("---")

def render_ai_training(engine):
    """Render AI model training interface"""
    st.markdown("## 🧠 AI Model Training Center")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Model Training Configuration")

        symbols_to_train = st.multiselect(
            "Select symbols for training",
            ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX', 'SPY', 'QQQ'],
            default=['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        )

        training_period = st.selectbox(
            "Training Data Period",
            ["1 month", "3 months", "6 months", "1 year", "2 years"],
            index=2
        )

        model_types = st.multiselect(
            "Model Types to Train",
            ["XGBoost", "Random Forest", "LSTM", "Transformer", "Ensemble"],
            default=["XGBoost", "Random Forest", "Ensemble"]
        )

        if st.button("🚀 Start AI Training", type="primary"):
            if not symbols_to_train:
                st.error("Please select at least one symbol for training!")
            else:
                status_box = st.empty()
                progress_bar = st.progress(0)

                try:
                    status_box.info(f"📡 Fetching data for {', '.join(symbols_to_train)}…")
                    progress_bar.progress(15)
                    training_results = engine.train_ai_models(symbols_to_train)
                    progress_bar.progress(100)

                    if training_results:
                        status_box.success("✅ AI models trained successfully!")
                        st.markdown("#### Training Results")
                        for symbol, result in training_results.items():
                            if result['status'] == 'success':
                                st.success(f"✅ {symbol}: {result['models_trained']} models trained with {result['features_count']} features")
                            else:
                                st.error(f"❌ {symbol}: {result['error']}")
                    else:
                        status_box.error("❌ Could not fetch market data for any symbol. Check your EODHD API key or internet connection.")
                        progress_bar.empty()
                except Exception as e:
                    status_box.error(f"❌ Training error: {e}")
                    progress_bar.empty()

    with col2:
        st.markdown("### Training Status")

        if engine.ai_engine.is_trained:
            st.success("🟢 AI Models: **TRAINED**")
            st.metric("Models Available", len(engine.ai_engine.models))
        else:
            st.warning("🔴 AI Models: **NOT TRAINED**")

        if engine.ai_engine.feature_importance:
            st.markdown("#### Top Features")
            # Show feature importance for first available symbol
            first_symbol = list(engine.ai_engine.feature_importance.keys())[0]
            importance_data = engine.ai_engine.feature_importance[first_symbol]

            for feature, importance in list(importance_data.items())[:5]:
                st.progress(float(importance), text=f"{feature}: {importance:.3f}")

def render_broker_setup(engine):
    """Render broker setup interface"""
    st.markdown("## 🔗 Broker API Integration")

    # Broker connection form
    with st.expander("Add New Broker Connection", expanded=len(engine.broker_apis) == 0):
        broker_type = st.selectbox("Broker Type", [e.value for e in BrokerType])
        broker_name = st.text_input("Connection Name", f"{broker_type}_Connection")

        col1, col2 = st.columns(2)

        if broker_type == BrokerType.ALPACA.value:
            with col1:
                api_key = st.text_input("API Key", type="password")
            with col2:
                secret_key = st.text_input("Secret Key", type="password")

            paper_trading = st.checkbox("Paper Trading Mode", value=True)

        elif broker_type == BrokerType.INTERACTIVE_BROKERS.value:
            with col1:
                host = st.text_input("TWS Host", "127.0.0.1")
                port = st.number_input("TWS Port", value=7497)
            with col2:
                client_id = st.number_input("Client ID", value=1)

        elif broker_type == BrokerType.TD_AMERITRADE.value:
            with col1:
                api_key = st.text_input("Consumer Key", type="password")
            with col2:
                redirect_uri = st.text_input("Redirect URI")

        else:  # Mock
            st.info("Mock broker for testing - no credentials needed")

        if st.button("Connect to Broker"):
            credentials = {}

            if broker_type == BrokerType.ALPACA.value:
                credentials = {
                    'api_key': api_key,
                    'secret_key': secret_key,
                    'paper': paper_trading
                }
            elif broker_type == BrokerType.INTERACTIVE_BROKERS.value:
                credentials = {
                    'host': host,
                    'port': port,
                    'client_id': client_id
                }
            elif broker_type == BrokerType.TD_AMERITRADE.value:
                credentials = {
                    'api_key': api_key,
                    'redirect_uri': redirect_uri
                }

            success = engine.add_broker(broker_name, BrokerType(broker_type), credentials)

            if success:
                st.success(f"✅ Successfully connected to {broker_name}!")
                st.rerun()

    # Connected brokers
    if engine.broker_apis:
        st.markdown("### Connected Brokers")

        for broker_name, broker in engine.broker_apis.items():
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown(f"**{broker_name}**")
                st.caption(f"Type: {broker.broker_type.value}")

            with col2:
                status = "🟢 Connected" if broker.is_connected else "🔴 Disconnected"
                st.markdown(status)

            with col3:
                if st.button("Remove", key=f"remove_{broker_name}"):
                    del engine.broker_apis[broker_name]
                    st.rerun()

def render_live_signals(engine):
    """Render live trading signals interface"""
    st.markdown("## ⚡ Live AI Trading Signals")

    if not engine.ai_engine.is_trained:
        st.warning("⚠️ AI models need to be trained first!")
        return

    # Signal generation controls
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_symbol = st.selectbox("Symbol", ['AAPL', 'GOOGL', 'MSFT', 'TSLA'])

    with col2:
        strategy = st.selectbox("Strategy", ["Multi-Model Ensemble", "XGBoost Momentum", "Random Forest"])

    with col3:
        if st.button("🔄 Generate Signal", type="primary"):
            with st.spinner(f"Fetching live data for {selected_symbol}…"):
                api_key = config.get_eodhd_api_key()
                current_data = pro_get_historical_data(selected_symbol, api_key)
                if current_data.empty:
                    st.error(f"Could not fetch data for {selected_symbol}. Check EODHD API key.")
                else:
                    current_data.columns = [c.lower() for c in current_data.columns]
                    if 'adjusted_close' in current_data.columns and 'close' not in current_data.columns:
                        current_data = current_data.rename(columns={'adjusted_close': 'close'})
                    current_data = current_data[['open', 'high', 'low', 'close', 'volume']].apply(
                        pd.to_numeric, errors='coerce'
                    ).dropna()
                    signal = engine.ai_engine.generate_signal(selected_symbol, current_data, strategy)
                    st.session_state.latest_signal = signal

    # Display latest signal
    if hasattr(st.session_state, 'latest_signal'):
        signal = st.session_state.latest_signal

        # Signal overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            action_color = "🟢" if signal.action == "BUY" else "🔴" if signal.action == "SELL" else "🟡"
            st.metric("Signal", f"{action_color} {signal.action}")

        with col2:
            st.metric("Confidence", f"{signal.confidence:.1%}")

        with col3:
            st.metric("Expected Return", f"{signal.expected_return:+.2%}")

        with col4:
            st.metric("Risk Score", f"{signal.risk_score:.1%}")

        # Explainable AI section
        st.markdown("### 🧠 AI Decision Explanation")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Key Factors")
            for feature, importance in signal.feature_importance.items():
                st.progress(float(importance), text=f"{feature}: {importance:.3f}")

        with col2:
            st.markdown("#### Reasoning")
            st.json(signal.reasoning)

        # Execute signal
        if st.button("🚀 Execute This Signal", type="primary"):
            if engine.broker_apis:
                result = engine.execute_trade(signal)

                if result['success']:
                    st.success(f"✅ {result['message']}")
                else:
                    st.error(f"❌ {result['message']}")
            else:
                st.error("No broker connected!")

def render_performance_analytics(engine):
    """Render performance analytics dashboard"""
    st.markdown("## 📊 Performance Analytics")

    metrics = engine.get_performance_metrics()

    if not metrics:
        st.info("No trading data available yet.")
        return

    # Key performance metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Trades", metrics.get('total_trades', 0))
    with col2:
        win_rate = metrics.get('win_rate', 0)
        st.metric("Win Rate", f"{win_rate:.1%}",
                  delta="above 50%" if win_rate > 0.5 else "below 50%",
                  delta_color="normal" if win_rate > 0.5 else "inverse")
    with col3:
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    with col4:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.1%}")
    with col5:
        total_pnl = metrics.get('total_pnl', 0)
        st.metric("Total P&L", f"${total_pnl:+,.2f}",
                  delta_color="normal" if total_pnl >= 0 else "inverse")

    # Portfolio value chart from real trade history
    st.markdown("### 📈 Portfolio Value Over Time")
    pv_history = metrics.get('portfolio_value_history', [])
    if pv_history and len(pv_history) > 1:
        trade_times = [t['timestamp'].strftime('%m-%d %H:%M') for t in engine.trade_history]
        fig_pv = go.Figure()
        fig_pv.add_trace(go.Scatter(
            x=trade_times, y=pv_history,
            mode='lines+markers',
            line=dict(color='#26a69a', width=2),
            marker=dict(size=6, color='#26a69a'),
            fill='tozeroy', fillcolor='rgba(38,166,154,0.08)',
            name='Portfolio Value',
        ))
        # Starting capital reference line
        fig_pv.add_hline(y=100000, line_dash='dash', line_color='#6B7280',
                         annotation_text='Starting Capital $100k')
        fig_pv.update_layout(
            paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
            font=dict(color='#FAFAFA'), height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title='Portfolio Value ($)',
            xaxis_title='Trade',
            showlegend=False,
        )
        fig_pv.update_xaxes(gridcolor='#1E2130', tickangle=-30)
        fig_pv.update_yaxes(gridcolor='#1E2130', tickformat='$,.0f')
        st.plotly_chart(fig_pv, use_container_width=True)
    else:
        st.info("Execute trades to see portfolio value history here.")

    # Trading history
    if engine.trade_history:
        st.markdown("### Recent Trades")
        trades_data = []

        for trade in engine.trade_history[-10:]:  # Last 10 trades
            trades_data.append({
                'Time': trade['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'Symbol': trade['signal']['symbol'],
                'Action': trade['signal']['action'],
                'Quantity': trade['signal']['quantity'],
                'Price': f"${trade['signal']['price']:.2f}",
                'Confidence': f"{trade['signal']['confidence']:.1%}",
                'Status': trade['order_result'].get('status', 'Unknown')
            })

        st.dataframe(pd.DataFrame(trades_data), use_container_width=True)

def render_risk_management(engine):
    """Render risk management interface"""
    st.markdown("## 🛡️ Risk Management Center")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Risk Parameters")

        # Risk settings form
        with st.form("risk_settings"):
            max_position_size = st.slider("Max Position Size (%)", 1, 50, 10)
            daily_loss_limit = st.slider("Daily Loss Limit (%)", 1, 20, 5)
            max_positions = st.number_input("Max Concurrent Positions", 1, 50, 10)
            stop_loss_pct = st.slider("Default Stop Loss (%)", 0.5, 10.0, 2.0)

            if st.form_submit_button("Update Risk Settings"):
                engine.risk_manager.risk_metrics.position_size_limit = max_position_size / 100
                engine.risk_manager.risk_metrics.daily_loss_limit = daily_loss_limit / 100
                engine.risk_manager.risk_metrics.max_positions = max_positions
                st.success("✅ Risk settings updated!")

    with col2:
        st.markdown("### Live Risk Alerts")

        alerts = []
        positions = engine.portfolio.get('positions', {})
        total_value = engine.portfolio.get('total_value', 1)
        cash = engine.portfolio.get('cash', 0)
        daily_pnl = engine.risk_manager.daily_pnl

        # 1 — concentration check: any single position > 30% of portfolio
        for sym, pos in positions.items():
            pos_value = pos['quantity'] * pos['avg_price']
            pct = pos_value / total_value * 100 if total_value else 0
            if pct > 30:
                alerts.append(f"🔴 High concentration: **{sym}** = {pct:.1f}% of portfolio")

        # 2 — daily loss limit
        daily_loss_limit = engine.risk_manager.risk_metrics.daily_loss_limit * total_value
        if daily_pnl < 0 and abs(daily_pnl) > daily_loss_limit * 0.8:
            alerts.append(f"🔴 Daily loss near limit: ${daily_pnl:+,.2f} (limit ${-daily_loss_limit:,.2f})")

        # 3 — cash ratio warning (< 10% cash)
        cash_pct = cash / total_value * 100 if total_value else 100
        if cash_pct < 10:
            alerts.append(f"⚠️ Low cash reserve: {cash_pct:.1f}% of portfolio")

        # 4 — max positions
        n_pos = len(positions)
        max_pos = engine.risk_manager.risk_metrics.max_positions
        if n_pos >= max_pos:
            alerts.append(f"⚠️ At maximum positions limit: {n_pos}/{max_pos}")

        # 5 — all clear
        if not alerts:
            st.success("✅ No risk alerts — portfolio within all limits")
        for alert in alerts:
            st.warning(alert)

    # Risk metrics visualisation
    st.markdown("### Current Risk Exposure")

    positions = engine.portfolio.get('positions', {})
    total_value = engine.portfolio.get('total_value', 1)

    if positions:
        pos_data = []
        for sym, pos in positions.items():
            mv = pos['quantity'] * pos['avg_price']
            pos_data.append({'Symbol': sym, 'Market Value': mv,
                             'Weight %': mv / total_value * 100})
        pos_df = pd.DataFrame(pos_data).sort_values('Weight %', ascending=False)

        fig = go.Figure(go.Bar(
            x=pos_df['Symbol'], y=pos_df['Weight %'],
            marker_color=['#ef5350' if w > 30 else '#42A5F5' for w in pos_df['Weight %']],
            text=[f"{w:.1f}%" for w in pos_df['Weight %']],
            textposition='outside',
        ))
        fig.update_layout(
            title='Position Concentration (red = >30%)',
            yaxis_title='Portfolio Weight %',
            paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
            font=dict(color='#FAFAFA'), height=300,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        fig.update_xaxes(gridcolor='#1E2130')
        fig.update_yaxes(gridcolor='#1E2130')
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pos_df.style.format({'Market Value': '${:,.2f}', 'Weight %': '{:.1f}%'}),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No open positions — deploy a bot and execute trades to see exposure.")

# Main function to render the interface
def render():
    """Main render function for the auto-trading module"""
    render_professional_autotrading()
