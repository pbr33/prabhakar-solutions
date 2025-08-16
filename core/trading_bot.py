import threading
import time
import requests
import pandas as pd
from typing import Dict

# Assuming AutoTradingEngine is in a separate file
from .trading_engine import AutoTradingEngine

class MockBrokerAPI:
    """A mock broker API to simulate trade execution."""
    def __init__(self, api_key: str, broker_name: str):
        self.api_key = api_key
        self.broker_name = broker_name
        self.is_connected = False

    def connect(self):
        """Simulates connecting to the broker."""
        if self.api_key:
            self.is_connected = True
            return True, f"Successfully connected to {self.broker_name}."
        else:
            self.is_connected = False
            return False, f"Failed to connect to {self.broker_name}: API key is missing."

    def place_order(self, symbol: str, action: str, quantity: int, order_type: str = "MARKET") -> Dict:
        """Simulates placing an order."""
        if not self.is_connected:
            return {'success': False, 'message': 'Not connected to broker.'}
        
        # Simulate a successful order
        return {
            'success': True, 
            'order_id': f"ORD_{int(time.time())}", 
            'status': "FILLED", 
            'message': f"Order to {action} {quantity} shares of {symbol} was successfully filled."
        }


class TradingBot:
    """Represents a single, independent trading agent running in a thread."""
    def __init__(self, bot_id: str, symbol: str, strategy: Dict, trade_qty: int, interval: int, 
                 broker_api: MockBrokerAPI, central_engine: AutoTradingEngine, eodhd_api_key: str):
        self.bot_id = bot_id
        self.symbol = symbol
        self.strategy = strategy
        self.trade_qty = trade_qty
        self.interval = interval
        self.broker_api = broker_api
        self.central_engine = central_engine
        self.eodhd_api_key = eodhd_api_key

        self.is_running = False
        self.stop_event = threading.Event()
        self.thread = None
        self.status = "Stopped"
        self.last_log = "Agent initialized."
        self.position = 0  # 0 for flat, 1 for long

    def _fetch_data(self) -> pd.DataFrame:
        """Fetches the latest intraday data for the bot's symbol."""
        # Note: In a real scenario, use a more robust data fetching module
        api_interval = f"{max(1, self.interval // 60)}m" # Ensure interval is at least 1m
        url = f"https://eodhd.com/api/intraday/{self.symbol}?interval={api_interval}&api_token={self.eodhd_api_key}&fmt=json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            return df
        except Exception as e:
            self.last_log = f"Data fetch failed: {e}"
            return pd.DataFrame()

    def _apply_strategy(self, data: pd.DataFrame) -> str:
        """Applies the moving average crossover strategy."""
        if data.empty or len(data) < self.strategy['long_window']:
            return "HOLD"

        short_window = self.strategy['short_window']
        long_window = self.strategy['long_window']
        
        data['short_mavg'] = data['close'].rolling(window=short_window).mean()
        data['long_mavg'] = data['close'].rolling(window=long_window).mean()
        
        last_row = data.iloc[-1]
        
        if last_row['short_mavg'] > last_row['long_mavg'] and self.position == 0:
            return "BUY"
        elif last_row['short_mavg'] < last_row['long_mavg'] and self.position == 1:
            return "SELL"
        else:
            return "HOLD"

    def _run_loop(self):
        """The main loop for the trading bot thread."""
        self.status = "Running"
        self.broker_api.connect()
        self.last_log = self.broker_api.connect()[1]
        
        while not self.stop_event.is_set():
            self.last_log = f"Fetching data for {self.symbol}..."
            data = self._fetch_data()
            
            if not data.empty:
                self.last_log = "Analyzing data..."
                signal = self._apply_strategy(data)
                self.last_log = f"Signal: {signal}"
                
                current_price = data['close'].iloc[-1]

                if signal != "HOLD":
                    self._execute_trade_signal(signal, current_price)

            self.last_log = f"Sleeping for {self.interval}s..."
            self.stop_event.wait(self.interval)

        self.status = "Stopped"
        self.last_log = "Agent has been stopped."

    def _execute_trade_signal(self, signal: str, price: float):
        """Executes a trade based on the generated signal."""
        action = signal.upper()
        
        # Check if we should act on the signal
        if (action == "BUY" and self.position == 0) or \
           (action == "SELL" and self.position == 1):
            
            self.last_log = f"Executing {action} for {self.trade_qty} {self.symbol}"
            broker_response = self.broker_api.place_order(self.symbol, action, self.trade_qty)
            
            if broker_response['success']:
                trade_result = self.central_engine.execute_trade(
                    self.symbol, action, self.trade_qty, price, agent_name=self.bot_id
                )
                if trade_result['status'] == 'FILLED':
                    self.position = 1 if action == "BUY" else 0
                    self.last_log = f"{action} order filled for {self.symbol}."
                else:
                    self.last_log = f"{action} failed: {trade_result.get('reason', 'Unknown')}"
            else:
                 self.last_log = f"Broker rejected {action}: {broker_response['message']}"

    def start(self):
        """Starts the trading bot in a new thread."""
        if not self.is_running:
            self.is_running = True
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            self.status = "Starting..."

    def stop(self):
        """Stops the trading bot."""
        if self.is_running:
            self.stop_event.set()
            self.is_running = False
            self.status = "Stopping..."
