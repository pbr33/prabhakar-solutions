# ui/tabs/ai_intelligence/agents/base_agent.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd

@dataclass
class AgentSignal:
    """Data class for agent trading signals."""
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0-100
    reasoning: str
    key_levels: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None

class BaseTradingAgent(ABC):
    """Abstract base class for trading agents."""
    
    def __init__(self, name: str, specialty: str, emoji: str):
        self.name = name
        self.specialty = specialty
        self.emoji = emoji
        self._cache = {}
    
    @abstractmethod
    def analyze(self, symbol: str, data: pd.DataFrame) -> AgentSignal:
        """Perform analysis and return trading signal."""
        pass
    
    @abstractmethod
    def get_confidence_factors(self) -> Dict[str, float]:
        """Return factors contributing to confidence score."""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data quality."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in data.columns for col in required_columns)
    
    def format_report(self, signal: AgentSignal) -> Dict[str, Any]:
        """Format agent report for UI display."""
        return {
            "agent": self.name,
            "emoji": self.emoji,
            "specialty": self.specialty,
            "signal": signal.signal,
            "confidence": signal.confidence,
            "reasoning": signal.reasoning,
            "key_levels": signal.key_levels,
            "metadata": signal.metadata or {}
        }