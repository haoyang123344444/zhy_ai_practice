"""
OpenAI Agent SDK Demo — 对照 multi_agent.py 的 SDK 重写版
==========================================================

你的手写版 (multi_agent.py)           →  这个 SDK 版
────────────────────────────────────    ────────────────────────────────
Router 手动 JSON 解析 (~35 行)         →  handoff() 一行, LLM 自动判断
Agent ReAct 循环 (~50 行)             →  Runner.run() 一行
MCPHub + AsyncExitStack (~40 行)      →  Agent(mcp_servers=[...]) 一行
print() 调试                          →  Trace 自动记录调用链 + 耗时
无对话记忆                             →  Session 自动管理上下文
工具发现手动拼 OpenAI schema (~30行)   →  SDK 自动从 MCP server 发现

演示四种模式:
  Pattern 1 — 直接 Agent（替代 practice.py）
  Pattern 2 — Triage + Handoff（SDK 最优雅的 pattern，替代 Router）
  Pattern 3 — 并行 Dispatch（对齐 multi_agent.py 行为）
  Pattern 4 — 多轮对话记忆（手写版完全没有的能力）

依赖: pip install openai-agents
"""

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path

from agents import Agent, Runner, handoff, trace
from agents.memory.openai_conversations_session import OpenAIConversationsSession
from agents.mcp.server import MCPServerStdio, MCPServerStdioParams

# LangFuse 追踪（可选，用于和 multi_agent.py 统一查看 trace）
from tracing import get_langfuse, flush, NOOP

# ============================================================
# 路径配置（复用你现有的 MCP Server 子进程）
# ============================================================
BASE_DIR = Path(__file__).parent

# 三个 MCP Server 共用 pdf_rag 的 venv（里面装了 openai + mcp + yfinance + httpx）
PYTHON = str(BASE_DIR / "mcp_based/pdf_rag/.venv/bin/python")

WEATHER_SERVER_CFG: MCPServerStdioParams = {
    "command": PYTHON,
    "args": [str(BASE_DIR / "mcp_based/weather/weather_server.py")],
}


STOCK_SERVER_CFG: MCPServerStdioParams = {
    "command": PYTHON,
    "args": [str(BASE_DIR / "mcp_based/stock/stock_server.py")],
}

# RAG_SERVER_CFG: MCPServerStdioParams = {
#     "command": PYTHON,
#     "args": [str(BASE_DIR / "mcp_based/pdf_rag/rag_server.py")],
# }

WEB_SEARCH_SERVER_CFG: MCPServerStdioParams = {
    "command": PYTHON,
    "args": [str(BASE_DIR / "mcp_based/web_search/web_search_server.py")],
}


# ============================================================
# 构建 Agent（核心！）
# ============================================================
def make_servers():
    """
    创建 MCP Server 实例（未连接状态）。

    关键差异 vs 手写版:
      手写:  MCPHub.connect_all() → 手动 list_tools() → 手动拼 OpenAI schema
             → 手动写 Agent 类的 ReAct 循环

      SDK:   MCPServerStdio() → Agent(mcp_servers=[server])
             → SDK 自动发现工具、自动 ReAct 循环
    """
    weather_server = MCPServerStdio(
        name="weather_mcp",
        params=WEATHER_SERVER_CFG,
        cache_tools_list=True,
    )


    stock_server = MCPServerStdio(
        name="stock_mcp",
        params=STOCK_SERVER_CFG,
        cache_tools_list=True,
        client_session_timeout_seconds=30,  # yfinance API 有时候很慢
    )

    # rag_server = MCPServerStdio(
    #     name="rag_mcp",
    #     params=RAG_SERVER_CFG,
    #     cache_tools_list=True,
    #     client_session_timeout_seconds=60,  # vector search + LLM rerank 较慢
    # )

    web_search_server = MCPServerStdio(
        name="web_search_mcp",
        params=WEB_SEARCH_SERVER_CFG,
        cache_tools_list=True,
    )
    return weather_server, stock_server, web_search_server


def create_agents(weather_server, stock_server, web_search_server):
    """
    Agent 定义 — SDK 的核心抽象。

    手写版: AGENT_DEFS dict + Agent 类（~90 行）
    SDK 版: Agent(...) 三个构造函数

    关键参数:
      mcp_servers=[...]  → SDK 自动连接 MCP、发现工具、转换 schema
      handoffs=[...]     → SDK 自动生成 handoff 工具给 LLM 调用
      instructions       → 等同于手写版的 system_prompt
    """

    # ---- 天气专家 ----
    weather_agent = Agent(
        name="WeatherExpert",
        instructions="""\
你是一个天气分析专家。你可以查询全球城市的实时天气、天气预警和天气报告。

工作方式:
  1. 理解用户任务，判断需要哪些天气信息
  2. 逐步调用工具收集数据（可能需要查询多个城市）
  3. 基于数据给出专业的天气分析和实用建议

注意:
  - 如果用户提到多个城市，逐一查询
  - 给出实用建议（如是否需要带伞、适合什么户外活动等）
  - 回答简洁专业，用中文""",
        mcp_servers=[weather_server],  # ← SDK 自动发现 weather server 的全部工具
        model="gpt-4o-mini",
    )

    # ---- 股票专家 ----
    stock_agent = Agent(
        name="StockExpert",
        instructions="""\
你是一个股票分析专家。你可以查询实时股价、公司基本信息和历史价格数据。

工作方式:
  1. 理解用户任务，判断需要哪些股票数据
  2. 逐步调用工具收集数据（可能需要同时查股价+公司信息+历史走势）
  3. 基于数据给出专业分析

注意:
  - 如果用户提到多个股票，逐一查询
  - 结合公司基本面和历史数据给出更全面的分析
  - 不要给出具体的买卖投资建议，只提供数据和分析
  - 回答简洁专业，用中文""",
        mcp_servers=[stock_server],
        model="gpt-4o-mini",
    )

    # ---- 论文问答专家 ----
#     rag_agent = Agent(
#         name="RAGExpert",
#         instructions="""\
# 你是一个学术论文问答专家。你可以搜索和查询论文 PDF 的内容。

# 工作方式:
#   1. 理解用户的问题，提取关键信息
#   2. 调用 search_pdf_tool 在论文中搜索相关内容
#   3. 基于搜索结果给出准确的回答，引用原文内容

# 注意:
#   - 回答要基于论文实际内容，不要编造
#   - 如果论文中没有相关信息，诚实告知用户
#   - 用中文回答，保持学术严谨""",
#         mcp_servers=[rag_server],
#         model="gpt-4o-mini",
#     )

    # ---- 网络搜索专家 ----
    search_agent = Agent(
        name="SearchExpert",
        instructions="""\
你是一个网络搜索专家。你可以搜索互联网获取最新信息。

工作方式:
  1. 理解用户的问题，提取搜索关键词
  2. 调用 web_search 工具搜索相关内容
  3. 基于搜索结果给出准确、有用的回答

注意:
  - 搜索结果可能有时效性，优先使用最新的信息
  - 如果搜索结果不足以回答问题，诚实告知
  - 回答简洁专业，用中文""",
        mcp_servers=[web_search_server],
        model="gpt-4o-mini",
    )

    # ---- Triage Agent（路由器） ----
    # 没有 mcp_servers，只有 handoffs — LLM 看到四个 handoff 工具
    # 自动判断把任务转给谁
    triage_agent = Agent(
        name="Triage",
        instructions="""\
你是一个智能路由助手。根据用户的问题，将任务交给合适的专家处理。

规则:
  - 天气、温度、气候、出行建议 → 交给 WeatherExpert
  - 股票、股价、公司信息、投资分析 → 交给 StockExpert
  - 论文、论文内容、研究、学术问题 → 交给 RAGExpert
  - 新闻搜索、网络信息、最新资讯、百科知识 → 交给 SearchExpert
  - 如果用户同时问了多个领域，依次交给对应的专家
  - 用中文与用户交流""",
        handoffs=[
            handoff(weather_agent),
            handoff(stock_agent),
            handoff(search_agent),
        ],
        model="gpt-4o-mini",
    )

    return triage_agent, weather_agent, stock_agent, search_agent


# ============================================================
# Pattern 1: 直接 Agent（最简单场景）
# ============================================================
async def pattern1_direct_agent(weather_agent, stock_agent, search_agent):
    """
    最简模式：直接调用一个 Agent，不走路由。

    适合场景: 你明确知道用户要查什么
    对比: practice.py 的单 Agent + tool calling

    关键 API:
      Runner.run(agent, "user input")
      → SDK 自动执行 ReAct 循环 (Thought→ToolCall→Observation→...→Answer)
      → 你手写的 Agent.run() 第 145-179 行的循环，SDK 内部帮你做了
    """
    print("\n" + "=" * 60)
    print("Pattern 1: Direct Agent（直接调用）")
    print("=" * 60)

    # 查天气
    # with trace("Weather Query"):
    #     result = await Runner.run(weather_agent, "旧金山今天天气怎么样？适合出门吗？")

    # print(f"\n[WeatherExpert 回复]\n{result.final_output}")

    # # 查股票
    # with trace("Stock Query"):
    #     result = await Runner.run(stock_agent, "Apple 最近股价表现如何？")

    # print(f"\n[StockExpert 回复]\n{result.final_output}")

    # # 查论文
    # with trace("RAG Query"):
    #     result = await Runner.run(rag_agent, "这篇论文的主要研究内容是什么？")

    # print(f"\n[RAGExpert 回复]\n{result.final_output}")

    # 搜网络
    with trace("Web Search Query"):
        result = await Runner.run(search_agent, "最近有什么关于AI的新闻？")

    print(f"\n[SearchExpert 回复]\n{result.final_output}")


# ============================================================
# Pattern 2: Triage + Handoff（SDK 最优雅的路由模式）
# ============================================================
async def pattern2_triage_handoff(triage_agent):
    """
    Handoff 模式: Triage Agent 自动判断意图，交接给专家。

    适合场景: 你不知道用户要问什么领域，让 Triage 自动判断
    对比: multi_agent.py 的 Router LLM + JSON 解析（_route 方法 ~35 行）

    Handoff 工作原理:
      1. Triage LLM 看到 handoff 工具: transfer_to_WeatherExpert,
         transfer_to_StockExpert, transfer_to_RAGExpert, transfer_to_SearchExpert
      2. LLM 决定交给谁 → 调用对应 handoff 工具
      3. SDK 自动切换 Agent、传递上下文
      4. 专家用 mcp_servers 提供的工具完成任务
      5. 结果返回

    你手写的 Router:
      Router LLM → 输出 JSON → 手动 parse → 手动 dispatch → 手动 aggregate
      → 3 步手动操作，JSON parse 还要 try/catch

    SDK Handoff:
      Runner.run(triage_agent, input) → 自动完成全部
      → 一行代码
    """
    print("\n" + "=" * 60)
    print("Pattern 2: Triage + Handoff（自动路由）")
    print("=" * 60)

    # ---- 场景 A: 单一领域 ----
    print("\n--- 场景 A: 单领域查询 ---")
    # with trace 追踪标记
    # with trace("Triage → Weather"):
    #     result = await Runner.run(
    #         triage_agent,
    #         "查一下东京今天的天气，要不要带伞？",
    #     )
    # print(f"\n[路由结果] Triage → {result.last_agent.name}")
    # print(f"[回复] {result.final_output[:300]}...")

    # # ---- 场景 B: 复合查询（Triage 自动串行交给两个专家） ----
    # print("\n--- 场景 B: 复合查询（自动串行两个专家） ---")
    # with trace("Triage → Weather + Stock"):
    #     result = await Runner.run(
    #         triage_agent,
    #         "帮我查一下旧金山天气，再查一下 TSLA 股价",
    #     )
    # print(f"\n[路由结果] 最后处理: {result.last_agent.name}")
    # print(f"[回复] {result.final_output[:300]}...")

    # ---- 场景 C: 论文问答 ----
    # print("\n--- 场景 C: 论文问答 ---")
    # with trace("Triage → RAG"):
    #     result = await Runner.run(
    #         triage_agent,
    #         "这篇论文的研究方法是什么？",
    #     )
    # print(f"\n[路由结果] Triage → {result.last_agent.name}")
    # print(f"[回复] {result.final_output[:300]}...")

    # ---- 场景 D: 网络搜索 ----
    print("\n--- 场景 D: 网络搜索 ---")
    with trace("Triage → Search"):
        result = await Runner.run(
            triage_agent,
            "最近 OpenAI 有什么新闻？",
        )
    print(f"\n[路由结果] Triage → {result.last_agent.name}")
    print(f"[回复] {result.final_output[:300]}...")

    # ---- 查看完整的 handoff 调用链 ----
    print("\n--- Handoff 调用链（trace 级别） ---")
    for i, item in enumerate(result.new_items):
        if item.type == "handoff":
            name = getattr(item, 'name', str(item))
            print(f"  Step {i}: ↪ handoff → {name}")
        elif item.type == "tool_call":
            print(f"  Step {i}: 🔧 tool_call → {item.raw_item.name}")
        elif item.type == "message":
            # Just count messages
            pass


# ============================================================
# Pattern 3: 并行 Dispatch（对齐你手写的 multi_agent.py）
# ============================================================
async def pattern3_parallel_dispatch(weather_agent, stock_agent):
    """
    并行模式: 对复合问题，同时跑多个专家，汇总结果。

    适合场景: 用户问题明确跨领域，需要并行加速
    对比: multi_agent.py 的 asyncio.gather 并行 Agent.run()

    虽然 SDK 的 handoff 能串行处理复合问题，但有些场景你需要
    真正的并行。这时候 SDK 和手写版一样 — 用 asyncio.gather。

    区别只是手写版调 agent.run()，SDK 版调 Runner.run()。
    """
    print("\n" + "=" * 60)
    print("Pattern 3: Parallel Dispatch（并行分发）")
    print("=" * 60)

    # 拆解子任务
    subtasks = {
        "weather": "查询旧金山的实时天气并给出出行建议",
        "stock": "查询 TSLA 的当前股价和公司基本信息",
    }

    # 并行执行
    async def run_one(agent, task: str) -> str:
        with trace(f"Parallel: {agent.name}"):
            result = await Runner.run(agent, task)
        return result.final_output
    

    weather_result, stock_result = await asyncio.gather(
        run_one(weather_agent, subtasks["weather"]),
        run_one(stock_agent, subtasks["stock"]),
    )



    print(f"\n[WeatherExpert] {weather_result[:200]}...")
    print(f"\n[StockExpert]  {stock_result[:200]}...")

    # ---- Aggregator 汇总 ----
    aggregator = Agent(
        name="Aggregator",
        instructions="""\
整合多个专家的分析结果，给用户完整、连贯的回答。
不编造数据，只使用专家提供的实际信息。用中文回答。""",
        model="gpt-4o-mini",
    )

    prompt = f"""\
用户问题: 帮我查一下旧金山天气和 TSLA 股价

天气专家结果:
{weather_result}

股票专家结果:
{stock_result}

请整合以上信息，给出回答。"""

    with trace("Aggregate"):
        final = await Runner.run(aggregator, prompt)

    print(f"\n[Aggregator 汇总]\n{final.final_output}")


# ============================================================
# Pattern 4: Session 多轮对话记忆（手写版完全没有的能力）
# ============================================================
async def pattern4_conversation_memory(weather_agent):
    """
    多轮对话记忆: SDK 的 Session 自动管理上下文。

    你手写版的 Agent.run() 每次调用都是:
      messages = [system_prompt, user_task]  ← 全新的！

    SDK 的 Session 自动维护跨轮上下文。每轮 Runner.run 结束后，
    SDK 自动把本轮对话追加到 session。下一轮 LLM 能看到完整历史。
    """
    print("\n" + "=" * 60)
    print("Pattern 4: Session（多轮对话记忆）")
    print("=" * 60)

    session = OpenAIConversationsSession()


    # 第一轮：建立上下文
    result1 = await Runner.run(
        weather_agent,
        "我想了解一下旧金山的天气",
        session=session,
    )
    print(f"\n[用户] 我想了解一下旧金山的天气")
    print(f"[AI]   {result1.final_output[:200]}...")

    # 第二轮：用代词 "那里" — Agent 需要从上下文推断指旧金山
    result2 = await Runner.run(
        weather_agent,
        "那里适合户外跑步吗？",
        session=session,
    )
    print(f"\n[用户] 那里适合户外跑步吗？")
    print(f"[AI]   {result2.final_output[:200]}...")

    # 第三轮：继续追问 — Agent 需要记住之前聊了旧金山和户外
    result3 = await Runner.run(
        weather_agent,
        "那明天呢？天气还适合吗？",
        session=session,
    )
    print(f"\n[用户] 那明天呢？天气还适合吗？")
    print(f"[AI]   {result3.final_output[:200]}...")

    # ============================================================
    # Session 的本质:
    #   每次 Runner.run(agent, input, session=s)
    #   → SDK 自动把 input + response 追加到 session.history
    #   → 下次调用时，session.history 自动拼到 messages 里
    #
    # 你手写版要实现这个，需要在每次 Agent.run() 后手动维护
    # conversation_history，然后在下一轮拼到 messages 前部。
    # 这正是你当前系统缺失的能力。
    # ============================================================


# ============================================================
# 交互模式（对标 multi_agent.py 的交互循环）
# ============================================================
async def interactive_mode(triage_agent):
    """交互式对话 — handoff + session 的组合使用"""
    print("\n" + "=" * 60)
    print("Interactive Mode（交互模式）")
    print("=" * 60)
    print("试试:")
    print("  '查 TSLA 股价和旧金山天气'")
    print("  '分析 AAPL 最近走势'")
    print("  '东京明天适合旅游吗'")
    print("  '论文的主要贡献是什么'")
    print("  '最近 AI 领域有什么大新闻'")
    print("  '接着聊刚才的话题'  ← 测试对话记忆（Session 自动处理）")
    print("Type 'quit' to exit")
    print("=" * 60)

    session = OpenAIConversationsSession()

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        if not user_input:
            continue

        with trace("Interactive Query"):
            result = await Runner.run(
                triage_agent,
                user_input,
                session=session,
                max_turns=10,  # 限制 ReAct 循环步数，防止失控
            )

        print(f"\n[{result.last_agent.name}] {result.final_output}")

    print("Bye!")


# ============================================================
# 启动入口
# ============================================================
async def main():
    print("=" * 60)
    print("OpenAI Agent SDK Demo")
    print("对比: multi_agent.py (手写) vs agent_sdk_demo.py (SDK)")
    print("=" * 60)

    # LangFuse trace（如果已配置），包住整个 demo 调用
    # 注意: SDK 自己的 trace() 记录在 OpenAI dashboard，LangFuse 记录在这里
    # 如果想统一查看，可以在 LangFuse 中搜索 trace name
    langfuse = get_langfuse()
    top_trace = langfuse.trace(
        name="SDKDemo",
        input={"patterns": "Pattern 2: Triage + Handoff"},
    ) if langfuse else NOOP

    # ---- Setup: 连接 MCP Server ----
    weather_server, stock_server, web_search_server = make_servers()

    # enter_async_context = connect (__aenter__) + cleanup (__aexit__) 自动管理
    stack = AsyncExitStack()
    try:
        await stack.enter_async_context(weather_server)
        await stack.enter_async_context(stock_server)
        # await stack.enter_async_context(rag_server)
        await stack.enter_async_context(web_search_server)
        print(f"[Setup] MCP servers connected (weather + stock + rag + search)")

        # ---- 构造 Agent（server 已连接，SDK 自动发现工具） ----
        triage_agent, weather_agent, stock_agent, search_agent = create_agents(
            weather_server, stock_server, web_search_server
        )

        print("[Setup] Agents ready: Triage → (WeatherExpert, StockExpert, RAGExpert, SearchExpert)")

        # ---- 跑 Demo ----
        # Pattern 1: 最简直接调用（对比 practice.py）
        # await pattern1_direct_agent(weather_agent, stock_agent, search_agent)

        # Pattern 2: Handoff 自动路由（对比 Router JSON 解析）
        await pattern2_triage_handoff(triage_agent)

        # # Pattern 3: 并行 Dispatch（对齐 multi_agent.py）
        # await pattern3_parallel_dispatch(weather_agent, stock_agent)

        # # Pattern 4: 多轮对话记忆（手写版没有的能力）
        # await pattern4_conversation_memory(weather_agent)

        # ---- 交互模式 ----
        # 取消下面注释开启交互模式:


    finally:
        top_trace.end()
        flush()
        # SDK 的 server.cleanup() 自动关闭子进程和清理连接
        await stack.aclose()
        print("\n[Cleanup] MCP servers disconnected.")


if __name__ == "__main__":
    asyncio.run(main())
