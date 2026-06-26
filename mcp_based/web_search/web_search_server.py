import asyncio
from tavily import TavilyClient
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("web-search")

TAVILY_API_KEY = "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"  # 换成你的 key

@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily. Returns title, URL, and snippet for each result."""
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = await asyncio.to_thread(
            lambda: client.search(query, max_results=max_results)
        )

        results = response.get("results", [])

        if not results:
            return f"No results found for: {query}"

        lines = []
        for i, r in enumerate(results):
            title = r.get("title", "N/A")
            href = r.get("url", "N/A")
            body = r.get("content", "N/A")
            lines.append(f"[{i + 1}] {title}\n    URL: {href}\n    {body}\n")

        return "\n".join(lines)

    except Exception as e:
        return f"Web search error: {str(e)}"


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()