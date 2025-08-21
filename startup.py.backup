#!/usr/bin/env python3
"""
Startup script to resolve import issues and ensure all dependencies are properly loaded.
Run this before starting the main application.
"""

import sys
import os
import importlib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def check_and_install_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'requests',
        'yfinance',
        'scikit-learn',
        'statsmodels',
        'langchain_openai'
    ]
    
    optional_packages = [
        'pandas_ta',
        'talib'
    ]
    
    print("🔍 Checking required dependencies...")
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - REQUIRED")
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"⚠️  {package} - OPTIONAL (enhanced features disabled)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n🚨 Missing required packages: {', '.join(missing_required)}")
        print("Please install them using: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("For enhanced features, install: pip install " + " ".join(missing_optional))
        print("Note: TA-Lib requires special installation - see installation guide")
    
    print("\n✅ All required dependencies are available!")
    return True

def setup_python_path():
    """Ensure the current directory is in Python path to avoid import issues."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"📁 Added {current_dir} to Python path")

def test_imports():
    """Test critical imports to identify circular import issues."""
    print("\n🧪 Testing critical imports...")
    
    try:
        from config import get_config
        print("✅ config.get_config")
    except Exception as e:
        print(f"❌ config.get_config: {e}")
        return False
    
    try:
        from services.data_fetcher import get_market_data_yfinance
        print("✅ services.data_fetcher.get_market_data_yfinance")
    except Exception as e:
        print(f"❌ services.data_fetcher.get_market_data_yfinance: {e}")
        return False
    
    try:
        from analysis.technical import detect_real_candlestick_patterns
        print("✅ analysis.technical.detect_real_candlestick_patterns")
    except Exception as e:
        print(f"❌ analysis.technical.detect_real_candlestick_patterns: {e}")
        return False
    
    try:
        from analysis.predictive import detect_anomalies, generate_forecast
        print("✅ analysis.predictive functions")
    except Exception as e:
        print(f"❌ analysis.predictive functions: {e}")
        return False
    
    try:
        from analysis.reporting import generate_investment_memo
        print("✅ analysis.reporting.generate_investment_memo")
    except Exception as e:
        print(f"❌ analysis.reporting.generate_investment_memo: {e}")
        return False
    
    try:
        from core.trading_engine import AutoTradingEngine
        print("✅ core.trading_engine.AutoTradingEngine")
    except Exception as e:
        print(f"❌ core.trading_engine.AutoTradingEngine: {e}")
        return False
    
    print("✅ All critical imports successful!")
    return True

def check_streamlit_config():
    """Check Streamlit configuration."""
    try:
        import streamlit as st
        print("✅ Streamlit is available")
        
        # Check if running in Streamlit environment
        if hasattr(st, 'session_state'):
            print("✅ Streamlit session state available")
        else:
            print("⚠️  Not running in Streamlit environment")
        
        return True
    except Exception as e:
        print(f"❌ Streamlit configuration issue: {e}")
        return False

def main():
    """Main startup function."""
    print("🚀 AI Trading Platform - Startup Diagnostics")
    print("=" * 50)
    
    # Setup Python path
    setup_python_path()
    
    # Check dependencies
    if not check_and_install_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        return False
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Check for circular imports or missing modules.")
        return False
    
    # Check Streamlit
    if not check_streamlit_config():
        print("\n❌ Streamlit configuration issue.")
        return False
    
    print("\n🎉 All checks passed! You can now run the application.")
    print("\nTo start the application:")
    print("streamlit run main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)