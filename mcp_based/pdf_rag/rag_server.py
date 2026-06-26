from mcp.server.fastmcp import FastMCP
from rag_core import search_pdf_async
import sys
print("PYTHON USED:", sys.executable)

mcp = FastMCP("pdf-rag")


@mcp.tool(
    description="Search inside thesis PDF using RAG vector search"
)
async def search_pdf_tool(question: str) -> str:
    try:
        return await search_pdf_async(question)
    except Exception as e:
        return f"PDF RAG error: {str(e)}"


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()