import threading
from datetime import datetime
from enum import Enum
from typing import Dict, List

# Assuming pro_get_fundamental_data is moved to services.data_fetcher
from services.data_fetcher import pro_get_fundamental_data
from config import config

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

class AutoTradingEngine:
    """
    Manages the core trading logic, portfolio, and order execution.
    """
    def __init__(self):
        self.is_active = False
        self.trades_executed: List[Dict] = []
        self.portfolio_balance: float = 100000  # Starting with $100k
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.max_position_size: float = 0.1  # Max 10% per position
        self.daily_loss_limit: float = 0.05  # Max 5% daily loss
        self.lock = threading.Lock()

    def execute_trade(self, symbol: str, action: str, quantity: int, price: float,
                      order_type: OrderType = OrderType.MARKET, agent_name: str = 'Auto-Trader') -> Dict:
        """Executes a trade order and updates the portfolio."""
        with self.lock:
            trade_id = f"TXN_{len(self.trades_executed) + 1:06d}"
            
            trade = {
                'id': trade_id,
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action.upper(),
                'quantity': quantity,
                'price': price,
                'order_type': order_type.value,
                'status': OrderStatus.FILLED.value,
                'total_value': quantity * price,
                'fees': quantity * price * 0.001,  # 0.1% fees
                'agent': agent_name
            }
            
            # Update portfolio based on action
            if action.upper() == 'BUY':
                self._handle_buy(trade, symbol, quantity, price)
            elif action.upper() == 'SELL':
                self._handle_sell(trade, symbol, quantity, price)
            
            self.trades_executed.append(trade)
            self.trade_history.append(trade)
            return trade

    def _handle_buy(self, trade: Dict, symbol: str, quantity: int, price: float):
        """Logic for handling a BUY order."""
        cost = (quantity * price) + trade['fees']
        if cost <= self.portfolio_balance:
            self.portfolio_balance -= cost
            if symbol in self.positions:
                current_qty = self.positions[symbol]['quantity']
                current_val = self.positions[symbol]['avg_price'] * current_qty
                new_qty = current_qty + quantity
                new_val = current_val + (quantity * price)
                self.positions[symbol]['avg_price'] = new_val / new_qty
                self.positions[symbol]['quantity'] = new_qty
            else:
                api_key = config.get_eodhd_api_key()
                fundamentals = pro_get_fundamental_data(symbol, api_key) if api_key else {}
                sector = fundamentals.get('General', {}).get('Sector', 'Unknown')
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'entry_time': datetime.now(),
                    'sector': sector
                }
        else:
            trade['status'] = OrderStatus.CANCELLED.value
            trade['reason'] = 'Insufficient funds'

    def _handle_sell(self, trade: Dict, symbol: str, quantity: int, price: float):
        """Logic for handling a SELL order."""
        if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
            proceeds = (quantity * price) - trade['fees']
            self.portfolio_balance += proceeds
            self.positions[symbol]['quantity'] -= quantity
            if self.positions[symbol]['quantity'] == 0:
                del self.positions[symbol]
        else:
            trade['status'] = OrderStatus.CANCELLED.value
            trade['reason'] = 'Insufficient shares'

    def get_portfolio_value(self, live_prices: Dict) -> float:
        """Calculates total portfolio value using live prices."""
        with self.lock:
            total_value = self.portfolio_balance
            for symbol, position in self.positions.items():
                current_price = live_prices.get(symbol, position['avg_price'])
                total_value += position['quantity'] * current_price
            return total_value
