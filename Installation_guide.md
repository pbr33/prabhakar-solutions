# Enhanced Trading Platform - Installation Guide

## Overview
This guide will help you set up the enhanced trading platform with real pattern detection and sophisticated technical analysis capabilities.

## Prerequisites
- Python 3.8 or higher
- pip package manager

## Installation Steps

### 1. Install Basic Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install TA-Lib (Critical for Real Pattern Detection)

TA-Lib requires special installation depending on your operating system:

#### Windows:
```bash
# Option 1: Using conda (recommended)
conda install -c conda-forge ta-lib

# Option 2: Using pip with pre-compiled wheels
pip install TA-Lib
```

#### macOS:
```bash
# Install homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install ta-lib using homebrew
brew install ta-lib

# Then install Python wrapper
pip install TA-Lib
```

#### Linux (Ubuntu/Debian):
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install build-essential wget

# Download and compile ta-lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Install Python wrapper
pip install TA-Lib
```

### 3. Verify Installation
Run this Python script to verify TA-Lib is installed correctly:

```python
import talib
import numpy as np

# Test data
high = np.array([82.15, 81.89, 83.03, 83.30, 83.85])
low = np.array([81.29, 80.64, 81.31, 82.65, 83.07])
close = np.array([81.59, 81.06, 82.87, 83.00, 83.61])
open = np.array([81.85, 81.20, 81.55, 82.91, 83.10])

# Test pattern detection
doji = talib.CDLDOJI(open, high, low, close)
print("TA-Lib installation successful!")
print("Doji pattern detection:", doji)
```

## Key Features Enabled

### 1. Real Pattern Detection
- **TA-Lib Integration**: Detects 12+ real candlestick patterns
- **Confidence Scoring**: Each pattern includes reliability metrics
- **Historical Analysis**: Patterns detected across entire dataset

### 2. Advanced Technical Indicators
- **Volatility**: Bollinger Bands, ATR, Keltner Channels
- **Volume**: OBV, A/D Line, Chaikin Money Flow
- **Trend**: ADX, Multiple Moving Averages
- **Momentum**: Enhanced RSI, Stochastic, Williams %R

### 3. Multi-Factor Analysis
- **Technical Analysis**: Pattern + indicator confluence
- **Fundamental Data**: P/E, Market Cap, Financial Metrics (via EODHD)
- **Sentiment Analysis**: News headline analysis and scoring

### 4. Enhanced Visualizations
- **Interactive Charts**: Fibonacci levels, Support/Resistance
- **Pattern Annotations**: Visual pattern markers on charts
- **Multi-timeframe**: Various technical overlay options

## API Keys Required

### EODHD API (Recommended)
- Sign up at: https://eodhd.com/
- Provides: Real-time data, fundamentals, news
- Free tier: 20 API calls/day
- Paid tiers: Start at $19.99/month

### Azure OpenAI (For AI Analysis)
- Sign up at: https://azure.microsoft.com/en-us/services/cognitive-services/openai-service/
- Provides: GPT-4 powered analysis and insights
- Required for: Comprehensive analysis reports

## Troubleshooting

### TA-Lib Installation Issues
1. **Windows**: Use conda instead of pip
2. **macOS**: Ensure Xcode command line tools are installed
3. **Linux**: Make sure build-essential is installed

### Import Errors
If you get import errors, the system will gracefully fallback to enhanced rule-based pattern detection.

### Performance Optimization
- **Caching**: All expensive operations are cached
- **Data Limits**: Analysis limited to most recent data for speed
- **Progressive Loading**: Features load incrementally

## Running the Application

```bash
streamlit run main.py
```

## Configuration Tips

1. **Set API Keys**: Enter EODHD and Azure keys in sidebar
2. **Symbol Selection**: Use the enhanced ticker search
3. **Pattern Detection**: Click "Detect Real Patterns" for TA-Lib analysis
4. **Comprehensive Analysis**: Use "Generate Complete Analysis" for AI insights

## Performance Notes

- **First Run**: May take longer due to library initialization
- **Pattern Detection**: Real pattern detection is faster and more accurate than simulation
- **Memory Usage**: Enhanced indicators require more memory but provide better insights
- **API Limits**: Be mindful of EODHD API rate limits

## Support

For technical issues:
1. Check that all dependencies are installed correctly
2. Verify API keys are valid and have sufficient credits
3. Ensure internet connection for real-time data
4. Review Streamlit logs for specific error messages