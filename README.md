# Multi-Tool RAG Agent

A lightweight AI agent built with the OpenAI Responses API.

## Features

* Multi-tool function calling
* Weather lookup
* Stock price lookup
* Calculator
* PDF question answering (RAG)
* Conversation memory
* Streaming responses

## Project Structure

```text
.
├── practice.py
├── schema.py
├── weather.py
├── stock.py
├── calculator.py
├── rag_files/
│   ├── __init__.py
│   ├── rag.py
│   ├── files/
│   │   └── thesis.pdf
│   └── vector_store_id.txt
└── README.md
```

## Supported Tools

### Weather Tool

Get real-time weather information for a city.

Example:

```text
What's the weather in Shenzhen?
```

### Stock Tool

Get stock prices by ticker symbol.

Example:

```text
What is the stock price of NVDA?
```

### Calculator Tool

Perform arithmetic operations.

Example:

```text
Calculate 15 * 20
```

### PDF RAG Tool

Search uploaded PDF documents and answer questions based on their contents.

Example:

```text
What is the degree title in my thesis document?
```

## Conversation Memory

The agent maintains a conversation history to support multi-turn interactions.

Example:

```text
User: What is the degree title in my PDF?

Assistant: ...

User: Who is the author?

Assistant: ...
```

## Streaming Output

Responses are streamed token-by-token for a more interactive experience.

## Installation

```bash
pip install openai requests yfinance
```

Set your API key:

```bash
export OPENAI_API_KEY="your_api_key"
```

## Run

```bash
python practice.py
```

## Example

```text
You: What is the degree title in my PDF and what's the weather in Shenzhen?

AI: Degree of Master of Science (120 credits) with a major in Electrical Engineering...

AI: Shenzhen weather: 30.4°C...
```

## Future Improvements

* Tool routing optimization
* Automatic tool retry
* Memory summarization
* Web search integration
* Agent planning
* Multi-agent architecture
* MCP support
* GUI / Web application

```
```
