cal_tools = [
    {
        "type": "function",
        "name": "simple_calculator",
        "description": "Calculate the result of applying an arithmetic operator to two numbers.",
        "strict": True,
        "parameters":{
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "first operated digit",
                },
                "b": {
                    "type": "number",
                    "description" : "second operated digit",
                },
                "op": {
                    "type": "string",
                    "description": "operator",
                    "enum": ["+", "-", "*", "/"]
                }
            },
            "required": ["a", "b", "op"],
            "additionalProperties": False,
        },
    },
]

weather_tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": (
            "Get the current weather for a specified city. "
            "Use this function whenever the user asks about weather."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city, e.g. 北京, 上海, Shenzhen.",
                }
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
]

stock_tool = [
    {
        "type": "function",
        "name": "get_stock_price",
        "description": "Get the current price of a stock by its ticker symbol.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL, NVDA, TSLA."
                }
            },
            "required": ["symbol"],
            "additionalProperties": False,
        },
    }
]

pdf_tool = [
    {
        "type": "function",
        "name": "search_pdf",
        "description": "Search the uploaded PDF document and answer questions based on its content.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user's question about the PDF document."
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    }
]

tools = cal_tools + weather_tools + stock_tool + pdf_tool