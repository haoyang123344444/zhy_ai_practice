"""
Multi-Agent System — Hand-written from scratch

架构:
  ┌──────────────────────────────────────┐
  │   MultiAgentOrchestrator              │
  │   1. Router: 分析意图 → 选择Agent       │
  │   2. Dispatcher: 并行分配子任务         │
  │   3. Aggregator: 汇总各Agent结果        │
  └──────┬──────────┬────────────────────┘
         │          │
    ┌────▼───┐  ┌───▼────┐  ┌──────────┐
    │Weather │  │ Stock  │  │ PDF RAG  │
    │ Agent  │  │ Agent  │  │  Agent   │
    └────────┘  └────────┘  └──────────┘

每个 Agent = 专家 system prompt + 专属 MCP 工具 + 独立 ReAct 推理循环

和 orchestrator_mcp.py 的区别:
  - orchestrator_mcp: 单 LLM → 并行调用工具 → 汇总（工具没有推理能力）
  - multi_agent: Router → 选择 Agent → 每个 Agent 独立多步推理 → 汇总
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

# 复用 MCP 连接管理器
from orchestrator_mcp import MCPHub

# ============================================================
# 配置
# ============================================================
BASE_DIR = Path(__file__).parent
client = OpenAI()

MCP_SERVERS = {
    "weather": {
        "command": str(BASE_DIR / "mcp_based/weather/.venv/bin/python"),
        "args": [str(BASE_DIR / "mcp_based/weather/weather_server.py")],
    },
    "stock": {
        "command": str(BASE_DIR / "mcp_based/stock/.venv/bin/python"),
        "args": [str(BASE_DIR / "mcp_based/stock/stock_server.py")],
    },
    # 可选: 启用前需确认 mcp_based/pdf_rag/.venv 存在
    # "pdf_rag": {
    #     "command": str(BASE_DIR / "mcp_based/pdf_rag/.venv/bin/python"),
    #     "args": [str(BASE_DIR / "mcp_based/pdf_rag/rag_server.py")],
    # },
}

AGENT_DEFS = {
    "weather": {
        "name": "天气分析专家",
        "emoji": "\U0001f324\ufe0f",
        "model": "gpt-4o-mini",
        "max_steps": 5,
        "tools": ["get_weather", "get_alerts", "get_weather_report"],
        "system_prompt": """\
你是一个天气分析专家。你可以查询全球城市的实时天气、天气预警和天气报告。

工作方式:
  1. 理解用户任务，判断需要哪些天气信息
  2. 逐步调用工具收集数据（可能需要查询多个城市）
  3. 基于数据给出专业的天气分析和实用建议

注意:
  - 如果用户提到多个城市，逐一查询
  - 给出实用建议（如是否需要带伞、适合什么户外活动等）
  - 如果任务跟天气完全无关，回复 "SKIP: 此任务不在我的专业领域内"
  - 回答简洁专业，用中文""",
    },
    "stock": {
        "name": "股票分析专家",
        "emoji": "\U0001f4c8",
        "model": "gpt-4o-mini",
        "max_steps": 5,
        "tools": ["get_stock_price", "get_company_info", "get_historical"],
        "system_prompt": """\
你是一个股票分析专家。你可以查询实时股价、公司基本信息和历史价格数据。

工作方式:
  1. 理解用户任务，判断需要哪些股票数据
  2. 逐步调用工具收集数据（可能需要同时查股价+公司信息+历史走势）
  3. 基于数据给出专业分析

注意:
  - 如果用户提到多个股票，逐一查询
  - 结合公司基本面和历史数据给出更全面的分析
  - 不要给出具体的买卖投资建议，只提供数据和分析
  - 如果任务跟股票完全无关，回复 "SKIP: 此任务不在我的专业领域内"
  - 回答简洁专业，用中文""",
    },
    # 可选: 启用 pdf_rag agent
    # "pdf_rag": {
    #     "name": "文档检索专家",
    #     "emoji": "\U0001f4da",
    #     "model": "gpt-4o-mini",
    #     "max_steps": 5,
    #     "tools": ["search_pdf_tool"],
    #     "system_prompt": """\
# 你是一个文档检索专家。你可以搜索和检索PDF论文中的内容。
# 基于检索到的内容回答用户问题，引用原文支持你的回答。
# 如果任务跟文档搜索完全无关，回复 "SKIP: 此任务不在我的专业领域内"
# 回答简洁专业，用中文""",
    # },
}

# ============================================================
# Agent — 独立 ReAct 推理循环
# ============================================================
class Agent:
    """一个有独立推理能力的 Agent"""

    def __init__(
        self,
        agent_id: str,
        config: dict,
        hub: MCPHub,
        openai_tools: list[dict],
    ):
        self.agent_id = agent_id
        self.name = config["name"]
        self.emoji = config.get("emoji", "\U0001f916")
        self.system_prompt = config["system_prompt"]
        self.model = config.get("model", "gpt-4o-mini")
        self.max_steps = config.get("max_steps", 5)
        self.tools = openai_tools
        self.hub = hub

    async def run(self, task: str, verbose: bool = True) -> str:
        """
        ReAct 推理循环:
          Thought → Action(调用工具) → Observation → Thought → ... → Final Answer
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        for step in range(self.max_steps):
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            # Agent 认为任务完成 → 返回最终回答
            if not msg.tool_calls:
                if verbose:
                    print(f"  [{self.emoji} {self.name}] OK ({step + 1} step(s))")
                return msg.content or ""

            # Agent 需要调用工具
            messages.append(msg)
            if verbose:
                names = [tc.function.name for tc in msg.tool_calls]
                print(f"  [{self.emoji} {self.name}] step {step + 1}: {names}")

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                args = json.loads(tc.function.arguments)
                try:
                    call_mcp = self.hub.get_caller(tool_name)
                    result = await call_mcp(**args)
                except Exception as e:
                    result = f"Tool error: {e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # 达到 max_steps，强制总结
        if verbose:
            print(f"  [{self.emoji} {self.name}] max steps reached, forcing summary")

        messages.append({
            "role": "user",
            "content": "请基于已收集的所有信息，给出你的最终分析结论。",
        })
        final = client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return final.choices[0].message.content or ""



# ============================================================
# MultiAgentOrchestrator — 路由 + 分发 + 汇总
# ============================================================
ROUTER_PROMPT = """\
你是一个任务路由器。分析用户输入，判断需要哪些专家 agent 来处理，并为每个 agent 写一个清晰的子任务。

可用的专家 agent:
{agent_list}

规则:
1. 如果用户问题涉及多个领域，选择所有相关的 agent
2. 为每个选中的 agent 写一个清晰的子任务（中文），告诉它具体要查什么分析什么
3. 如果不需要某个 agent，不要包含它
4. 如果用户问题很复杂，将其拆解为适合各 agent 的子任务

只返回 JSON（不要 markdown 代码块）:
{{"agents": ["weather"], "tasks": {{"weather": "查询旧金山的实时天气并给出出行建议"}}}}"""

AGGREGATOR_PROMPT = """\
你是一个智能助手，负责整合多个专家 agent 的分析结果，给用户一个完整、连贯的回答。

用户原始问题: {user_input}

各专家分析结果:
{agent_results}

要求:
- 整合所有专家的信息，逻辑清晰地回答用户问题
- 不编造数据，只使用专家提供的实际信息
- 用中文回答，简洁有条理"""


class MultiAgentOrchestrator:
    def __init__(self, agents: dict[str, Agent]):
        self.agents = agents

    async def _route(self, user_input: str) -> dict[str, str]:
        """Router: 分析用户意图，返回 {agent_id: subtask}"""

        agent_list = "\n".join(
            f"- {aid}: {a.name}" for aid, a in self.agents.items()
        )
        router_prompt = ROUTER_PROMPT.format(agent_list=agent_list)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": router_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,
        )
        raw = response.choices[0].message.content or "{}"

        # 清理可能的 markdown 代码块
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  [Router] JSON parse failed, fallback: all agents")
            return {aid: user_input for aid in self.agents}

        return plan.get("tasks", {})



    async def run(self, user_input: str, verbose: bool = True) -> str:
        """完整的 multi-agent 流程: route → dispatch → aggregate"""

        if verbose:
            print(f"\n{'─' * 50}")
            print(f"[User] {user_input}")

        # ---- 1. Route: 分析意图，分配子任务 ----
        if verbose:
            print(f"\n[Router] analyzing intent...")
        tasks = await self._route(user_input)


        if not tasks:
            return "抱歉，我无法判断该问题需要哪些专家来处理。"

        if verbose:
            print(f"[Router] activated: {', '.join(tasks.keys())}")
            for aid, task in tasks.items():
                print(f"  {aid} → \"{task}\"")

        # ---- 2. Dispatch: 并行执行所有 Agent ----
        if verbose:
            print(f"\n[Agents] executing in parallel...")

        async def dispatch(aid: str, task: str) -> tuple[str, str]:
            agent = self.agents.get(aid)
            if agent is None:
                return aid, f"Agent '{aid}' not found"
            try:
                result = await agent.run(task, verbose=verbose)
                return aid, result
            except Exception as e:
                return aid, f"Agent error: {e}"

        results: dict[str, str] = dict(
            await asyncio.gather(
                *[dispatch(aid, task) for aid, task in tasks.items()]
            )
        )

        # ---- 3. Aggregate: 过滤 + 汇总 ----
        if verbose:
            print(f"\n[Aggregator] synthesizing results...")

        # 过滤掉 agent 明确表示不在其领域的回答
        valid: dict[str, str] = {}
        for aid, result in results.items():
            if result.strip().startswith("SKIP:"):
                if verbose:
                    print(f"  [{aid}] skipped (not in domain)")
            else:
                valid[aid] = result

        if not valid:
            return "所有专家都认为此问题不在其专业领域内，请尝试换一种问法。"

        # 只有一个 agent 有结果 → 直接返回
        if len(valid) == 1:
            return list(valid.values())[0]

        # 多个 agent → LLM 汇总
        results_text = "\n\n---\n\n".join(
            f"### {self.agents[aid].name}:\n{text}"
            for aid, text in valid.items()
        )
        agg_prompt = AGGREGATOR_PROMPT.format(
            user_input=user_input,
            agent_results=results_text,
        )

        final = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": agg_prompt}],
        )
        return final.choices[0].message.content or ""


# ============================================================
# 启动入口
# ============================================================
async def main():
    hub = MCPHub()

    print("=" * 50)
    print("Multi-Agent System")
    print("=" * 50)

    # 1. 连接所有 MCP Server
    print("\nConnecting to MCP servers...")
    all_tools = await hub.connect_all(MCP_SERVERS)
    print(f"Total MCP tools discovered: {len(all_tools)}")

    # 2. 为每个 Agent 装配专属工具
    agents: dict[str, Agent] = {}
    for aid, config in AGENT_DEFS.items():
        tool_names = set(config["tools"])
        agent_tools = [
            t for t in all_tools
            if t["function"]["name"] in tool_names
        ]

        if not agent_tools:
            print(f"  SKIP {aid}: required MCP server not connected")
            continue

        agents[aid] = Agent(aid, config, hub, agent_tools)
        tool_list = ", ".join(t["function"]["name"] for t in agent_tools)
        print(f"  {config['emoji']} {config['name']}: {tool_list}")

    if not agents:
        print("\nNo agents available. Exiting.")
        await hub.close()
        return

    # 3. 创建 Orchestrator
    orch = MultiAgentOrchestrator(agents)

    print(f"\n{'=' * 50}")
    print(f"Ready! {len(agents)} agents standing by.")
    print(f"")
    print(f"Try:")
    print(f"  '查 TSLA 股价和旧金山天气'")
    print(f"  '分析 AAPL 最近走势，结合公司基本面'")
    print(f"  '旧金山今天适合户外运动吗'")
    print(f"  '对比一下 NVDA 和 AMD'")
    print(f"Type 'quit' to exit")
    print(f"{'=' * 50}")

    try:
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            if not user_input:
                continue

            answer = await orch.run(user_input)
            print(f"\n{'=' * 50}")
            print(f"AI: {answer}")
            print(f"{'=' * 50}")

    finally:
        await hub.close()
        print("\nBye!")


if __name__ == "__main__":
    asyncio.run(main())
