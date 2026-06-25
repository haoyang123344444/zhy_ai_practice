"""
Orchestrator Agent — MCP Native Version

区别于 orchestrator.py：
  - 不再直接 import 工具函数
  - 通过 MCP 协议（stdio transport）连接 MCP server 子进程
  - 动态发现工具列表，调用时通过 ClientSession.call_tool()
  - 这才是真正的 "agent 调度 MCP" 架构

启动时：
  ┌──────────────┐
  │ Orchestrator │  ← 本文件
  └────┬────┬────┘
       │    │
  MCP  │    │  MCP (stdio 子进程)
       ▼    ▼
  weather   stock

每个 MCP server 运行在独立子进程中，Orchestrator 通过 stdin/stdout 通信。
"""

import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ------------------------------------------------------------
# 配置：你的 MCP server
# ------------------------------------------------------------
BASE_DIR = Path(__file__).parent

# MCP_SERVERS = {
#     "weather": {
#         "command": str(BASE_DIR / "mcp_based/weather/.venv/bin/python"),
#         "args": [str(BASE_DIR / "mcp_based/weather/weather_server.py")],
#     },
#     "stock": {
#         "command": str(BASE_DIR / "mcp_based/stock/.venv/bin/python"),
#         "args": [str(BASE_DIR / "mcp_based/stock/stock_server.py")],
#     },
# }

MCP_SERVERS = {
    "weather": {
        "command": str(BASE_DIR / "mcp_based/weather/.venv/bin/python"),
        "args": [str(BASE_DIR / "mcp_based/weather/weather_server.py")],
    },
    "stock": {
        "command": str(BASE_DIR / "mcp_based/stock/.venv/bin/python"),
        "args": [str(BASE_DIR / "mcp_based/stock/stock_server.py")],
    },
}

client = OpenAI()
SYSTEM_PROMPT = (
    "你是一个智能助手，可以使用多个工具来回答用户问题。"
    "如果用户问题涉及多个领域，同时调用所有需要的工具。"
    "拿到工具结果后，用中文简洁汇总回答。"
    "不要编造数据。"
)


# ============================================================
# 1. 将 MCP Tool Schema 转换为 OpenAI Function Schema
# ============================================================
def mcp_tool_to_openai(tool) -> dict:
    """MCP Tool.inputSchema (JSON Schema) → OpenAI function calling format"""
    openai_func: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
        },
    }


    input_schema = tool.inputSchema or {}

    params: dict[str, Any] = {
        "type": "object",
        "properties": input_schema.get("properties", {}),
    }


    if "required" in input_schema:
        params["required"] = input_schema["required"]

    openai_func["function"]["parameters"] = params
    return openai_func


# ============================================================
# 2. MCP 连接管理器（使用 AsyncExitStack 正确管理生命周期）
# ============================================================
class MCPHub:
    """
    管理多个 MCP server 的连接，每个 server 是独立子进程。

    使用 AsyncExitStack 确保 async context manager 的生命周期正确管理：
    - stdio_client 和 ClientSession 都是 async context manager
    - AsyncExitStack.push_async_exit() 把清理函数压栈
    - close() 时按 LIFO 顺序逐个退出
    """

    def __init__(self):
        self._stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._tool_map: dict[str, str] = {}        # tool_name → server_name

    async def connect_all(self, servers: dict) -> list[dict]:
        """启动所有 MCP server 并发现工具，返回 OpenAI 格式的工具列表"""
        openai_tools = []

        for name, cfg in servers.items():
            params = StdioServerParameters(
                command=cfg["command"],
                args=cfg["args"],
            )

            # 用 AsyncExitStack 管理 stdio_client 的生命周期
            transport = await self._stack.enter_async_context(stdio_client(params))
            read, write = transport
            session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()

            # 发现该 server 的所有工具
            result = await session.list_tools()
            print(f"[MCP Hub] {name}: {len(result.tools)} tools — "
                  f"{[t.name for t in result.tools]}")

            self._sessions[name] = session

            for tool in result.tools:
                self._tool_map[tool.name] = name
                openai_tools.append(mcp_tool_to_openai(tool))

        return openai_tools

    def get_caller(self, tool_name: str):
        """根据 tool name 返回对应 MCP server 的 call_tool 方法"""
        server_name = self._tool_map.get(tool_name)
        if server_name is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        session = self._sessions[server_name]

        async def call_mcp(**kwargs):
            result = await session.call_tool(tool_name, arguments=kwargs)
            # result.content 是 ContentBlock 列表，提取 text
            texts = [c.text for c in result.content if hasattr(c, "text")]
            return "\n".join(texts) if texts else str(result.content)

        return call_mcp

    async def close(self):
        """按 LIFO 顺序正确关闭所有连接"""
        await self._stack.aclose()






# ============================================================
# 3. Orchestrator 主逻辑
# ============================================================
class Orchestrator:
    def __init__(self, hub: MCPHub, tools: list[dict]):
        self.hub = hub
        self.tools = tools

    async def run(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        # ----- Step 1: LLM 决定调用哪些工具 -----
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
        )

        choice = response.choices[0].message

        # ----- Step 2: 无需工具，直接返回 -----
        if not choice.tool_calls:
            return choice.content or ""

        # ----- Step 3: 并行调用 MCP 工具 -----
        messages.append(choice)
        print(f"\n[Orchestrator] 调用 {len(choice.tool_calls)} 个工具:")

        async def execute_one(tc):
            tool_name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  -> MCP:{tool_name}({args})")

            call_mcp = self.hub.get_caller(tool_name)
            try:
                result = await call_mcp(**args)
            except Exception as e:
                result = f"Error: {e}"

            return tc.id, result

        # 真正并行执行
        results = await asyncio.gather(
            *[execute_one(tc) for tc in choice.tool_calls]
        )

        for tool_call_id, result_text in results:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_text,
            })

        # ----- Step 4: LLM 汇总 -----
        final = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return final.choices[0].message.content or ""


# ============================================================
# 4. 启动入口
# ============================================================
async def main():
    hub = MCPHub()

    print("=" * 50)
    print("Orchestrator Agent — MCP Native")
    print("=" * 50)
    print("Connecting to MCP servers...")

    tools = await hub.connect_all(MCP_SERVERS)
    print(f"Total tools: {len(tools)}\n")

    orch = Orchestrator(hub, tools)

    print("Ready! 试试: '查 TSLA 股价和旧金山天气'\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            if not user_input:
                continue

            answer = await orch.run(user_input)
            print(f"\nAI: {answer}\n")

    finally:
        await hub.close()
        print("Bye!")


if __name__ == "__main__":
    asyncio.run(main())
