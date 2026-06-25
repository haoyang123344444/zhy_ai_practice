import time
import requests
import yfinance as yf


def _retry(func, *args, max_retries=3, **kwargs):
    """Simple exponential backoff for yfinance rate limiting."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "Too Many Requests" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            raise


def get_stock_price(symbol: str) -> dict:
    symbol = symbol.strip().upper()

    try:
        ticker = _retry(yf.Ticker, symbol)
        data = _retry(ticker.history, period="1d")

        if data.empty:
            return {
                "tool": "get_stock_price",
                "symbol": symbol,
                "error": "找不到该股票代码",
            }

        latest_price = data["Close"].iloc[-1]

        return {
            "tool": "get_stock_price",
            "symbol": symbol,
            "price": round(float(latest_price), 2),
            "currency": "USD",
        }

    except Exception as e:
        return {
            "tool": "get_stock_price",
            "symbol": symbol,
            "error": str(e),
        }