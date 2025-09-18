# Pulse of Profit

Simple Flask app to compute technical indicators (RSI, MACD, OBV, Ichimoku) and show a Plotly chart.

Run locally (from the project root):

1. Use the project's virtualenv Python (recommended):

2. Open http://127.0.0.1:5001 in your browser.

Troubleshooting:
- If you get ModuleNotFoundError for `requests` or other packages, make sure you installed the requirements into the project venv. Use the venv python executable above.
- The app will try yfinance first. If yfinance fails, the app falls back to Stooq CSV data (this may be used intermittently depending on remote responses).

Design notes:
- Modern dark theme with glassy cards and improved typography.
- Charting uses Plotly (loaded via CDN) for interactive charts.

Deploy notes:
- For production, use a WSGI server such as gunicorn and put it behind a reverse proxy (nginx). Example:

   gunicorn -w 4 -b 0.0.0.0:8000 app:app

+- Keep SECRET_KEY set in the environment for production.

