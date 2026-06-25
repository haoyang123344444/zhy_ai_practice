# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-tool AI agent that supports weather lookup, stock prices, calculator, and PDF RAG (question answering). The repo has two architectural layers:

- **Root level** — Original monolithic agent (`practice.py`) using OpenAI Responses API with tool-calling schemas defined in `schema.py`.
- **`mcp_based/`** — Refactored architecture where each tool is an independent MCP (Model Context Protocol) server using `FastMCP`.

## Commands

Each MCP server lives in its own directory with its own venv and `pyproject.toml`:

```bash
# Weather MCP server
cd mcp_based/weather
source .venv/bin/activate
python weather_server.py

# PDF RAG MCP server
cd mcp_based/pdf_rag
source .venv/bin/activate
python rag_server.py

# Stock MCP server
cd mcp_based/stock
source .venv/bin/activate
python stock_server.py

# Orchestrator Agent (root level, multi-tool router)
cd .
source mcp_based/pdf_rag/.venv/bin/activate  # has openai + requests + yfinance
python orchestrator.py
```

MCP servers are run via `stdio` transport. They are designed to be registered in Claude Code's MCP configuration, not run standalone.

## Architecture

### Root-level (original agent)

- `practice.py` — Original single-step agent using OpenAI Responses API with streaming.
- `orchestrator.py` — Orchestrator using direct function imports from `weather.py`/`stock.py`.
- **`orchestrator_mcp.py`** — MCP-native orchestrator. Connects to MCP servers as subprocesses via `stdio_client`, dynamically discovers tools from each server, and routes LLM tool calls through MCP protocol. Uses `AsyncExitStack` for connection lifecycle.
- `schema.py` — OpenAI function-calling schema definitions for all tools.
- `weather.py` / `stock.py` / `calculator.py` — Individual tool implementations used by both `practice.py` and `orchestrator.py`.

### `mcp_based/` (MCP architecture, current week1 branch)

Each subdirectory is an independent MCP server:

- **`mcp_based/weather/weather_server.py`** — FastMCP server exposing `get_weather`, `get_alerts`, and `get_weather_report` tools. Uses Open-Meteo free APIs (no API key needed). Geocodes city name to lat/lon before fetching weather.

- **`mcp_based/pdf_rag/`** — FastMCP server with hybrid PDF RAG:
  - `rag_server.py` — MCP server wrapper, exposes `search_pdf_tool` via FastMCP.
  - `rag_core.py` — Core RAG logic: uses OpenAI vector stores for semantic search + BM25 keyword search (full hybrid), then CrossEncoder (`ms-marco-MiniLM-L-6-v2`) reranking. Requires `OPENAI_API_KEY` env var. Caches results by question+vector_store_id. Reads PDF from `files/Master_thesis.pdf` and persists the vector store ID in `vector_store_id.txt`.

- **`mcp_based/stock/stock_server.py`** — FastMCP server exposing `get_stock_price`, `get_company_info`, and `get_historical` tools. Uses yfinance (wrapped in `asyncio.to_thread`). No API key required.

### Key patterns

- All MCP servers use `FastMCP` with `@mcp.tool()` decorators for tool registration.
- Synchronous libraries (OpenAI, yfinance) are wrapped in `asyncio.to_thread()` for MCP async compatibility.
- Async-native HTTP libraries (httpx) are used directly with `async/await`.
- The `OPENAI_API_KEY` environment variable is required for the PDF RAG server.
- Weather and Stock MCPs require no API keys (free Open-Meteo / yfinance).
