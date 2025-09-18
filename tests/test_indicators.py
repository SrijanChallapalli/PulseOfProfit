import pandas as pd
import numpy as np
from indicators import rsi, macd


def test_rsi_constant_up():
    # constant upward price should produce RSI near 100 after enough periods
    s = pd.Series(np.linspace(100, 200, 30))
    r = rsi(s, period=14)
    assert r.iloc[-1] > 70


def test_macd_basic():
    s = pd.Series(np.linspace(1, 100, 100))
    macd_line, signal_line, hist = macd(s)
    # macd_line should be numeric and same length
    assert len(macd_line) == len(s)
    assert not np.isnan(macd_line).all()
