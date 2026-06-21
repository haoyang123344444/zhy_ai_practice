import requests
import yfinance as yf

def get_stock_price(symbol: str) -> dict:
    symbol = symbol.strip().upper()

    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")

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