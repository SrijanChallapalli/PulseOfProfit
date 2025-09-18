import pandas as pd
import numpy as np

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()

def ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    high = df['high']
    low = df['low']
    tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2
    kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_span_b = ((high.rolling(window=senkou_b).max() + low.rolling(window=senkou_b).min()) / 2).shift(kijun)
    chikou_span = df['close'].shift(-kijun)
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['rsi'] = rsi(out['close'])
    macd_line, signal_line, hist = macd(out['close'])
    out['macd'] = macd_line
    out['macd_signal'] = signal_line
    out['macd_hist'] = hist
    out['obv'] = obv(out['close'], out['volume'])
    tenkan, kijun, span_a, span_b, chikou = ichimoku(out)
    out['tenkan'] = tenkan
    out['kijun'] = kijun
    out['senkou_span_a'] = span_a
    out['senkou_span_b'] = span_b
    out['chikou'] = chikou
    return out

def latest_snapshot(df: pd.DataFrame, weights: dict = None) -> dict:
    last = df.iloc[-1]
    out = {
        "close": float(last.get("close", float('nan'))),
        "rsi": float(last.get("rsi", float('nan'))),
        "macd": float(last.get("macd", float('nan'))),
        "macd_signal": float(last.get("macd_signal", float('nan'))),
        "obv": float(last.get("obv", float('nan'))),
        "tenkan": float(last.get("tenkan", float('nan'))),
        "kijun": float(last.get("kijun", float('nan')))
    }
    # include data source if present on the DataFrame
    src = None
    try:
        src = getattr(df, 'attrs', {}).get('source')
    except Exception:
        src = None
    if src:
        out['source'] = src
    # per-indicator signals and combined recommendation
    try:
        signals = {}
        score = 0.0
        # normalize weights
        if weights is None:
            weights = {}

        # RSI signal: oversold -> Buy, overbought -> Sell
        r = out.get('rsi')
        if r is None or (isinstance(r, float) and (r != r)):
            signals['rsi'] = 'N/A'
        else:
            w = float(weights.get('rsi', 1))
            if r < 30:
                signals['rsi'] = 'Buy'
                score += 1.0 * w
            elif r > 70:
                signals['rsi'] = 'Sell'
                score -= 1.0 * w
            else:
                signals['rsi'] = 'Hold'

        # MACD signal: compare macd line to signal line
        macd_v = out.get('macd')
        sig = out.get('macd_signal')
        if macd_v is None or sig is None or (isinstance(macd_v, float) and (macd_v != macd_v)):
            signals['macd'] = 'N/A'
        else:
            w = float(weights.get('macd', 1))
            if macd_v > sig:
                signals['macd'] = 'Buy'
                score += 1.0 * w
            elif macd_v < sig:
                signals['macd'] = 'Sell'
                score -= 1.0 * w
            else:
                signals['macd'] = 'Hold'

        # OBV: compare latest change direction (simple)
        try:
            if len(df['obv']) >= 2:
                obv_now = df['obv'].iloc[-1]
                obv_prev = df['obv'].iloc[-2]
                w = float(weights.get('obv', 1))
                if obv_now > obv_prev:
                    signals['obv'] = 'Buy'
                    score += 1.0 * w
                elif obv_now < obv_prev:
                    signals['obv'] = 'Sell'
                    score -= 1.0 * w
                else:
                    signals['obv'] = 'Hold'
            else:
                signals['obv'] = 'N/A'
        except Exception:
            signals['obv'] = 'N/A'

        # Ichimoku Tenkan/Kijun: bullish if tenkan > kijun
        try:
            tenkan_v = out.get('tenkan')
            kijun_v = out.get('kijun')
            if tenkan_v is None or kijun_v is None or (isinstance(tenkan_v, float) and (tenkan_v != tenkan_v)):
                signals['ichimoku'] = 'N/A'
            else:
                w = float(weights.get('ichimoku', 1))
                if tenkan_v > kijun_v:
                    signals['ichimoku'] = 'Buy'
                    score += 1.0 * w
                elif tenkan_v < kijun_v:
                    signals['ichimoku'] = 'Sell'
                    score -= 1.0 * w
                else:
                    signals['ichimoku'] = 'Hold'
        except Exception:
            signals['ichimoku'] = 'N/A'

        # Moving averages (50/200)
        try:
            ma50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
            ma200 = df['close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
            if ma50 is None or ma200 is None or (isinstance(ma50, float) and (ma50 != ma50)):
                signals['moving_average'] = 'N/A'
            else:
                w = float(weights.get('moving_average', 1))
                if ma50 > ma200:
                    signals['moving_average'] = 'Buy'
                    score += 1.0 * w
                elif ma50 < ma200:
                    signals['moving_average'] = 'Sell'
                    score -= 1.0 * w
                else:
                    signals['moving_average'] = 'Hold'
        except Exception:
            signals['moving_average'] = 'N/A'

        # Build a combined recommendation from score
        # Normalize: bias thresholds based on number of valid signals
        # count valid (non-N/A) signals and compute buy/sell counts
        valid_count = sum(1 for v in signals.values() if v in ('Buy', 'Sell', 'Hold'))
        buy_count = sum(1 for v in signals.values() if v == 'Buy')
        sell_count = sum(1 for v in signals.values() if v == 'Sell')

        # compute threshold based on sum of absolute weights for valid signals
        import math
        sum_weights = 0.0
        for k, v in signals.items():
            if v in ('Buy', 'Sell', 'Hold'):
                sum_weights += abs(float(weights.get(k, 1)))
        if sum_weights == 0:
            combined = 'Hold'
        else:
            thresh = math.ceil(sum_weights / 2.0)
            if score >= thresh:
                combined = 'Buy'
            elif score <= -thresh:
                combined = 'Sell'
            else:
                combined = 'Hold'

        # human-friendly explanations per indicator (short)
        explanations = {}
        try:
            # RSI explanation
            if isinstance(r, float):
                if r < 30:
                    explanations['rsi'] = f"RSI={r:.1f}: oversold (Buy)"
                elif r > 70:
                    explanations['rsi'] = f"RSI={r:.1f}: overbought (Sell)"
                else:
                    explanations['rsi'] = f"RSI={r:.1f}: neutral"
            else:
                explanations['rsi'] = 'RSI: N/A'

            # MACD explanation
            if isinstance(macd_v, float) and isinstance(sig, float):
                explanations['macd'] = f"MACD={macd_v:.3f} vs Signal={sig:.3f}"
            else:
                explanations['macd'] = 'MACD: N/A'

            # OBV explanation
            try:
                if len(df['obv']) >= 2:
                    if obv_now > obv_prev:
                        explanations['obv'] = f"OBV rising ({int(obv_prev)} → {int(obv_now)})"
                    elif obv_now < obv_prev:
                        explanations['obv'] = f"OBV falling ({int(obv_prev)} → {int(obv_now)})"
                    else:
                        explanations['obv'] = 'OBV flat'
                else:
                    explanations['obv'] = 'OBV: N/A'
            except Exception:
                explanations['obv'] = 'OBV: N/A'

            # Ichimoku explanation
            if isinstance(tenkan_v, float) and isinstance(kijun_v, float):
                explanations['ichimoku'] = f"Tenkan={tenkan_v:.3f} vs Kijun={kijun_v:.3f}"
            else:
                explanations['ichimoku'] = 'Ichimoku: N/A'

            # Moving average explanation
            if ma50 is not None and ma200 is not None and isinstance(ma50, float):
                explanations['moving_average'] = f"MA50={ma50:.3f} vs MA200={ma200:.3f}"
            else:
                explanations['moving_average'] = 'MA50/MA200: N/A'
        except Exception:
            explanations = {k: '' for k in signals.keys()}

        out['signals'] = signals
        out['recommendation'] = combined
        out['score'] = float(score)
        out['valid_signals'] = int(valid_count)
        out['buy_count'] = int(buy_count)
        out['sell_count'] = int(sell_count)
        out['summary'] = f"{buy_count} bullish, {sell_count} bearish of {valid_count} indicators => {combined}"
        out['explanations'] = explanations
    except Exception:
        out['signals'] = { 'rsi': 'N/A', 'macd': 'N/A', 'obv': 'N/A', 'ichimoku': 'N/A', 'moving_average': 'N/A' }
        out['recommendation'] = 'Hold'
        out['score'] = 0
        out['valid_signals'] = 0
        out['explanations'] = {k: '' for k in out['signals'].keys()}
    return out

