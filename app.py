import io
import requests
import re
import os
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from indicators import compute_all_indicators, latest_snapshot
from xml.etree import ElementTree as ET

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-key")  # needed for flash()

def fetch_price_data(ticker, start, end):
    """Try Yahoo (yfinance) first; on rate-limit/empty, fall back to Stooq daily data."""
    # --- Try Yahoo first ---
    try:
        df = yf.download(
            ticker, start=start, end=end,
            progress=False, auto_adjust=True, threads=False, timeout=20
        )
        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index)
            df.rename(columns=str.lower, inplace=True)
            # record data source
            df.attrs['source'] = 'yfinance'
            return df
    except Exception as e:
        print("Yahoo fetch error:", e)

    # --- Fallback: Stooq (daily OHLC, no intraday) ---
    # Stooq uses lowercase and market suffixes: US tickers -> .us
    stooq_symbol = f"{ticker.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        if resp.text.strip() == "" or "404 Not Found" in resp.text:
            return pd.DataFrame()
        sdf = pd.read_csv(io.StringIO(resp.text))
        # Expected cols: Date, Open, High, Low, Close, Volume
        sdf.rename(columns=str.lower, inplace=True)
        sdf["date"] = pd.to_datetime(sdf["date"])
        sdf.set_index("date", inplace=True)
        # Clip to requested window
        sdf = sdf.loc[(sdf.index >= pd.to_datetime(start)) & (sdf.index <= pd.to_datetime(end))]
        # Align to yfinance-style column names
        out = sdf[["open", "high", "low", "close", "volume"]]
        out.attrs['source'] = 'stooq'
        return out
    except Exception as e:
        print("Stooq fallback error:", e)
        return pd.DataFrame()


def make_figure(df: pd.DataFrame, add_macd=False, add_ichimoku=False, add_rsi=False, add_obv=False):
    """Build a stacked subplot figure with Price (candlestick) + optional RSI, MACD, OBV rows.
    This keeps each indicator visible on its own axis and preserves legends.
    """
    from plotly.subplots import make_subplots

    rows = 1
    if add_rsi:
        rows += 1
    if add_macd:
        rows += 1
    if add_obv:
        rows += 1
    if add_ichimoku:
        rows += 1

    # allocate heights (price gets larger share)
    heights = [0.5] + [0.5 / max(1, rows - 1)] * (rows - 1)

    titles = ["Price"]
    if add_rsi:
        titles.append('RSI')
    if add_macd:
        titles.append('MACD')
    if add_obv:
        titles.append('OBV')
    if add_ichimoku:
        titles.append('Ichimoku')

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=heights,
                        subplot_titles=titles)

    row = 1
    # Price
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Price', showlegend=True
    ), row=row, col=1)

    # moving averages if present
    for ma in ('ma50', 'ma200'):
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma.upper(), mode='lines', showlegend=True), row=row, col=1)

    # Ichimoku will be drawn on its own subplot row below (not overlaid on price)

    # RSI
    if add_rsi and 'rsi' in df.columns:
        row += 1
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', mode='lines', showlegend=True), row=row, col=1)
        fig.add_hline(y=70, line_dash='dot', line_color='lightcoral', row=row, col=1)
        fig.add_hline(y=30, line_dash='dot', line_color='lightgreen', row=row, col=1)

    # MACD
    if add_macd and {'macd','macd_signal'}.issubset(df.columns):
        row += 1
        fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD', mode='lines', showlegend=True), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', mode='lines', showlegend=True), row=row, col=1)

    # OBV
    if add_obv and 'obv' in df.columns:
        row += 1
        fig.add_trace(go.Scatter(x=df.index, y=df['obv'], name='OBV', mode='lines', showlegend=True), row=row, col=1)

    # Ichimoku (dedicated subplot)
    if add_ichimoku and {'tenkan','kijun','senkou_span_a','senkou_span_b'}.issubset(df.columns):
        row += 1
        fig.add_trace(go.Scatter(x=df.index, y=df['tenkan'], name='Tenkan', mode='lines', showlegend=True), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['kijun'], name='Kijun', mode='lines', showlegend=True), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['senkou_span_a'], name='Span A', mode='lines', line={'dash':'dash'}, showlegend=True), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['senkou_span_b'], name='Span B', mode='lines', line={'dash':'dash'}, showlegend=True), row=row, col=1)

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )

    # axis titles
    cur = 1
    fig.update_yaxes(title_text="Price", row=cur, col=1)
    if add_rsi and 'rsi' in df.columns:
        cur += 1
        fig.update_yaxes(title_text="RSI", row=cur, col=1)
    if add_macd and {'macd','macd_signal'}.issubset(df.columns):
        cur += 1
        fig.update_yaxes(title_text="MACD", row=cur, col=1)
    if add_obv and 'obv' in df.columns:
        cur += 1
        fig.update_yaxes(title_text="OBV", row=cur, col=1)
    if add_ichimoku and {'tenkan','kijun','senkou_span_a','senkou_span_b'}.issubset(df.columns):
        cur += 1
        fig.update_yaxes(title_text="Ichimoku", row=cur, col=1)

    return fig

@app.route("/", methods=["GET"])
def index():
    default_end = datetime.utcnow().date()
    default_start = default_end - timedelta(days=365)
    recent = session.get('recent', [])
    movers = market_movers()
    # fetch a small set of headlines for the homepage marquee
    try:
        news = fetch_trending_news('')
    except Exception:
        news = []

    return render_template("index.html",
                           default_ticker="AAPL",
                           default_start=default_start.isoformat(),
                           default_end=default_end.isoformat(),
                           recent=recent,
                           movers=movers,
                           news=news)

@app.route("/analyze", methods=["POST"])
def analyze():
    print("→ /analyze hit")

    # 1) Read + SANITIZE ticker (letters/numbers/.- only)
    raw = (request.form.get("ticker") or "").strip().upper()
    ticker = re.sub(r"[^A-Z0-9\.\-]", "", raw)
    print(f"ticker raw={raw!r} sanitized={ticker!r}")

    start = request.form.get("start")
    end   = request.form.get("end")
    add_rsi      = bool(request.form.get("rsi"))
    add_macd     = bool(request.form.get("macd"))
    add_obv      = bool(request.form.get("obv"))
    add_ichimoku = bool(request.form.get("ichimoku"))

    # If the user didn't check any indicators, default to showing all of them
    if not any([add_rsi, add_macd, add_obv, add_ichimoku]):
        add_rsi = add_macd = add_obv = add_ichimoku = True

    if not ticker:
        flash("Please enter a valid ticker (letters/numbers only).")
        return redirect(url_for("index"))

    # 2) Parse dates
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt   = datetime.fromisoformat(end)
    except Exception as e:
        print("Date parse error:", e)
        flash("Invalid date format.")
        return redirect(url_for("index"))

    # 3) Fetch price data via helper which tries yfinance then falls back to Stooq
    df = fetch_price_data(ticker, start_dt, end_dt + timedelta(days=1))
    if df is None or df.empty:
        flash(f"No data found for {ticker}. Try a different symbol or date range.")
        return redirect(url_for("index"))

    # 4) Indicators + chart
    df.index = pd.to_datetime(df.index)
    df.rename(columns=str.lower, inplace=True)
    df_ind = compute_all_indicators(df)
    fig = make_figure(df_ind, add_macd=add_macd, add_ichimoku=add_ichimoku, add_rsi=add_rsi, add_obv=add_obv)
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    # map posted weights to the expected keys
    weights = {
        'rsi': float(request.form.get('w_rsi') or 1),
        'macd': float(request.form.get('w_macd') or 1),
        'obv': float(request.form.get('w_obv') or 1),
        'ichimoku': float(request.form.get('w_ichimoku') or 1),
        'moving_average': float(request.form.get('w_moving_average') or 1)
    }
    snap = latest_snapshot(df_ind, weights=weights)
    news = fetch_trending_news(ticker)

    # store recent in session
    try:
        recent = session.get('recent', [])
        entry = {'ticker': ticker, 'start': start_dt.date().isoformat(), 'end': end_dt.date().isoformat(), 'recommendation': snap.get('recommendation'), 'score': snap.get('score')}
        if len(recent) == 0 or recent[0].get('ticker') != entry['ticker'] or recent[0].get('start') != entry['start']:
            recent.insert(0, entry)
        session['recent'] = recent[:8]
    except Exception as e:
        print('session recent store error:', e)

    return render_template(
        "result.html",
        ticker=ticker,
        chart_html=chart_html,
        snapshot=snap,
        news=news,
        start=start_dt.date().isoformat(),
        end=end_dt.date().isoformat(),
        add_rsi=add_rsi,
        add_macd=add_macd,
        add_obv=add_obv,
        add_ichimoku=add_ichimoku,
    )



@app.route("/api/indicators/<ticker>")
def api_indicators(ticker):
    days = int(request.args.get("days", 200))
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    df = fetch_price_data(ticker.upper(), start, end + timedelta(days=1))
    if df.empty:
        return jsonify({"error": "No data"}), 404
    df_ind = compute_all_indicators(df)
    return jsonify({"ticker": ticker.upper(),
                    "as_of": df_ind.index[-1].isoformat(),
                    "indicators": latest_snapshot(df_ind)})


def fetch_trending_news(ticker, max_items=5):
    """Fetch simple news headlines via Google News RSS for the ticker query.
    Falls back to empty list on errors.
    """
    try:
        q = f"{ticker} stock"
        url = f"https://news.google.com/rss/search?q={requests.utils.requote_uri(q)}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = []
        for item in root.findall('.//item')[:max_items]:
            title = item.find('title').text if item.find('title') is not None else ''
            link = item.find('link').text if item.find('link') is not None else ''
            items.append({'title': title, 'link': link})
        return items
    except Exception as e:
        print('news fetch error:', e)
        return []


def market_movers(sample_tickers=None, days=3, max_items=5):
    """Return a small list of tickers and their recent percent change.
    Best-effort: uses fetch_price_data and falls back on failures.
    """
    if sample_tickers is None:
        sample_tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "GOOG", "META"]
    movers = []
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    for t in sample_tickers:
        try:
            df = fetch_price_data(t, start, end + timedelta(days=1))
            if df is None or df.empty or 'close' not in df.columns:
                continue
            if len(df['close']) < 2:
                continue
            last = df['close'].iloc[-1]
            prev = df['close'].iloc[-2]
            pct = (last - prev) / prev * 100.0 if prev != 0 else 0.0
            movers.append({'ticker': t, 'pct': float(pct), 'last': float(last)})
        except Exception as e:
            print('market mover fetch error for', t, e)
            continue
    movers.sort(key=lambda x: abs(x['pct']), reverse=True)
    return movers[:max_items]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  # 5001 avoids AirPlay conflict
    app.run(host="0.0.0.0", port=port, debug=True)

