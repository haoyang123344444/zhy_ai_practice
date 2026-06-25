"""
Orchestrator Agent — Multi-Tool Router

设计思路：
  1. 用户输入 → Orchestrator 分析意图
  2. Orchestrator 决定调用哪些工具（可以同时调用多个）
  3. 收集工具结果 → 汇总生成最终回答

这是 multi-agent 系统的基础模式：
  未来可以把每个工具替换成独立的 agent（有独立 memory、推理能力），
  Orchestrator 变成真正的 "agent 调度器"。
"""

from openai import OpenAI
import json

# ------------------------------------------------------------
# Tool Implementations（与 MCP server 相同的逻辑）
# ------------------------------------------------------------
from weather import get_weather
from stock import get_stock_price
from calculator import simple_calculator

client = OpenAI()

# ------------------------------------------------------------
# Tool Schema（告诉 OpenAI 有哪些工具可用）
# ------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气。当用户问天气时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 北京、Shenzhen、Tokyo"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "获取股票实时价格。当用户问股价时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码，如 AAPL、TSLA、NVDA"}
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simple_calculator",
            "description": "执行加减乘除运算。当用户需要数学计算时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个操作数"},
                    "b": {"type": "number", "description": "第二个操作数"},
                    "op": {
                        "type": "string",
                        "enum": ["+", "-", "*", "/"],
                        "description": "运算符",
                    },
                },
                "required": ["a", "b", "op"],
            },
        },
    },
]

# ------------------------------------------------------------
# Tool 映射表
# ------------------------------------------------------------
TOOL_MAP = {
    "get_weather": get_weather,
    "get_stock_price": get_stock_price,
    "simple_calculator": simple_calculator,
}

# ------------------------------------------------------------
# Orchestrator 核心逻辑
# ------------------------------------------------------------
SYSTEM_PROMPT = """你是一个智能助手，可以同时调用多个工具来回答用户问题。

规则：
- 如果用户问题涉及多个领域（如天气+股票），一次性调用所有需要的工具
- 拿到所有工具结果后，用中文汇总回答
- 不要编造数据，只使用工具返回的结果
- 回答简洁明了"""


def orchestrate(user_input: str) -> str:
    """
    Orchestrator 主流程:
      Step 1: 把用户消息 + 工具定义发给 LLM
      Step 2: LLM 返回要调用的工具列表
      Step 3: 执行工具，收集结果
      Step 4: 把结果交还给 LLM 生成最终回答
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    # ----- Step 1: 第一次调用 — LLM 决定调用哪些工具 -----
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    choice = response.choices[0].message

    # ----- Step 2: 如果没有需要工具，直接返回文本 -----
    if not choice.tool_calls:
        return choice.content or "（无回复）"

    # ----- Step 3: 并行执行所有工具 -----
    messages.append(choice)  # 把 tool_calls 加入历史

    print(f"\n[Orchestrator] 检测到 {len(choice.tool_calls)} 个工具调用:")

    for tc in choice.tool_calls:
        tool_name = tc.function.name
        args = json.loads(tc.function.arguments)

        print(f"  -> 调用 {tool_name}({args})")

        func = TOOL_MAP.get(tool_name)
        try:
            result = func(**args)
        except Exception as e:
            result = {"error": str(e)}

        # 把工具结果加入对话
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result, ensure_ascii=False),
        })

    # ----- Step 4: 第二次调用 — LLM 汇总结果生成回答 -----
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    return final_response.choices[0].message.content or "（无回复）"


# ------------------------------------------------------------
# 交互入口
# ------------------------------------------------------------
def main():
    print("=" * 50)
    print("Orchestrator Agent (Multi-Tool Router)")
    print("可用工具: get_weather | get_stock_price | simple_calculator")
    print("输入 'quit' 退出")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Bye!")
            break

        if not user_input:
            continue

        answer = orchestrate(user_input)
        print(f"\nAI: {answer}")


if __name__ == "__main__":
    main()
