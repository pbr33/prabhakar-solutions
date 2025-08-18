# ui/tabs/ai_intelligence/agents/multi_agent_system.py
from typing import Dict, List, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from .technical_agent import TechnicalAgent
from .macro_agent import MacroAgent
from .sentiment_agent import SentimentAgent
from .quant_agent import QuantAgent

class MultiAgentSystem:
    """Orchestrates multiple trading agents with parallel processing."""
    
    def __init__(self, enable_parallel: bool = True):
        self.enable_parallel = enable_parallel
        self.agents = [
            TechnicalAgent(),
            MacroAgent(),
            SentimentAgent(),
            QuantAgent()
        ]
    
    def analyze_symbol(
        self, 
        symbol: str, 
        data: pd.DataFrame,
        llm_service=None
    ) -> Dict[str, Any]:
        """Run analysis from all agents."""
        if self.enable_parallel:
            return self._parallel_analysis(symbol, data, llm_service)
        else:
            return self._sequential_analysis(symbol, data, llm_service)
    
    def _parallel_analysis(
        self,
        symbol: str,
        data: pd.DataFrame,
        llm_service=None
    ) -> Dict[str, Any]:
        """Run agent analysis in parallel for better performance."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            future_to_agent = {
                executor.submit(
                    agent.analyze, 
                    symbol, 
                    data
                ): agent for agent in self.agents
            }
            
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    signal = future.result()
                    results[agent.name] = agent.format_report(signal)
                except Exception as e:
                    results[agent.name] = {
                        "error": str(e),
                        "signal": "ERROR",
                        "confidence": 0
                    }
        
        # Calculate consensus
        results["consensus"] = self._calculate_consensus(results)
        
        return results
    
    def _sequential_analysis(
        self,
        symbol: str,
        data: pd.DataFrame,
        llm_service=None
    ) -> Dict[str, Any]:
        """Run agent analysis sequentially."""
        results = {}
        
        for agent in self.agents:
            try:
                signal = agent.analyze(symbol, data)
                results[agent.name] = agent.format_report(signal)
            except Exception as e:
                results[agent.name] = {
                    "error": str(e),
                    "signal": "ERROR",
                    "confidence": 0
                }
        
        results["consensus"] = self._calculate_consensus(results)
        
        return results
    
    def _calculate_consensus(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate consensus from agent signals."""
        valid_results = [
            r for r in results.values() 
            if isinstance(r, dict) and r.get("signal") != "ERROR"
        ]
        
        if not valid_results:
            return {
                "signal": "HOLD",
                "confidence": 0,
                "reasoning": "No valid agent signals available"
            }
        
        # Count signals
        signals = [r["signal"] for r in valid_results]
        confidences = [r["confidence"] for r in valid_results]
        
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        hold_count = signals.count("HOLD")
        
        # Determine consensus
        if buy_count > sell_count and buy_count > hold_count:
            consensus_signal = "BUY"
        elif sell_count > buy_count and sell_count > hold_count:
            consensus_signal = "SELL"
        else:
            consensus_signal = "HOLD"
        
        # Calculate weighted confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Adjust confidence based on agreement
        agreement_factor = max(buy_count, sell_count, hold_count) / len(valid_results)
        adjusted_confidence = avg_confidence * agreement_factor
        
        return {
            "signal": consensus_signal,
            "confidence": round(adjusted_confidence, 1),
            "buy_votes": buy_count,
            "sell_votes": sell_count,
            "hold_votes": hold_count,
            "total_agents": len(valid_results),
            "reasoning": self._generate_consensus_reasoning(
                consensus_signal, 
                agreement_factor
            )
        }
    
    def _generate_consensus_reasoning(
        self, 
        signal: str, 
        agreement: float
    ) -> str:
        """Generate reasoning for consensus signal."""
        strength = "Strong" if agreement > 0.75 else "Moderate" if agreement > 0.5 else "Weak"
        
        reasons = {
            "BUY": f"{strength} buy signal from multiple agents. Favorable technical and fundamental conditions detected.",
            "SELL": f"{strength} sell signal from multiple agents. Risk factors outweigh opportunities.",
            "HOLD": f"{strength} hold signal. Mixed signals suggest waiting for clearer direction."
        }
        
        return reasons.get(signal, "Consensus analysis complete.")