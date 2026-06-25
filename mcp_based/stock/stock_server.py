import asyncio
import time
import yfinance as yf
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("stock")


def _retry_sync(func, *args, max_retries=3, **kwargs):
    """Simple exponential backoff for yfinance rate limiting."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "Too Many Requests" in str(e) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


# -----------------------------
# 1. Stock Price Tool
# -----------------------------
@mcp.tool()
async def get_stock_price(symbol: str) -> str:
    """Get current stock price for a given ticker symbol (e.g. AAPL, TSLA)."""
    try:
        ticker = await asyncio.to_thread(_retry_sync, yf.Ticker, symbol.strip().upper())
        data = await asyncio.to_thread(_retry_sync, ticker.history, period="1d")

        if data.empty:
            return f"No data found for symbol: {symbol}"

        price = data["Close"].iloc[-1]
        return f"Stock: {symbol.upper()} | Price: ${price:.2f} USD"

    except Exception as e:
        return f"Stock price error: {str(e)}"


# -----------------------------
# 2. Company Info Tool
# -----------------------------
@mcp.tool()
async def get_company_info(symbol: str) -> str:
    """Get company name, sector, industry, and market cap for a ticker."""
    try:
        ticker = await asyncio.to_thread(_retry_sync, yf.Ticker, symbol.strip().upper())
        info = await asyncio.to_thread(_retry_sync, lambda: ticker.info)

        if not info or "symbol" not in info:
            return f"No company info found for: {symbol}"

        lines = [
            f"Company: {info.get('longName', info.get('shortName', 'N/A'))}",
            f"Symbol: {info.get('symbol', symbol.upper())}",
            f"Sector: {info.get('sector', 'N/A')}",
            f"Industry: {info.get('industry', 'N/A')}",
            f"Market Cap: ${info.get('marketCap', 'N/A')}",
            f"Country: {info.get('country', 'N/A')}",
        ]
        return "\n".join(lines)

    except Exception as e:
        return f"Company info error: {str(e)}"


# -----------------------------
# 3. Historical Data Tool
# -----------------------------
@mcp.tool()
async def get_historical(symbol: str, period: str = "1mo") -> str:
    """Get historical closing prices. period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y."""
    try:
        ticker = await asyncio.to_thread(_retry_sync, yf.Ticker, symbol.strip().upper())
        data = await asyncio.to_thread(_retry_sync, ticker.history, period=period)

        if data.empty:
            return f"No historical data for: {symbol}"

        closes = [f"{d.date()}: ${p:.2f}" for d, p in data["Close"].items()]
        return f"Historical Close ({symbol.upper()}, {period}):\n" + "\n".join(closes)

    except Exception as e:
        return f"Historical data error: {str(e)}"


# -----------------------------
# Entry
# -----------------------------
def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
