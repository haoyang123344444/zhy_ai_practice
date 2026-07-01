"""
Multi-Agent System — Hand-written from scratch

架构:
  ┌──────────────────────────────────────────────────┐
  │   MultiAgentOrchestrator (+ Memory)                │
  │   0. Memory: 注入短/长期记忆到上下文                  │
  │   1. Router: 分析意图 → 选择Agent                   │
  │   2. Dispatcher: 并行分配子任务                     │
  │   3. Aggregator: 汇总各Agent结果                    │
  │   4. Reflector: 质检 + 修正（完整性/准确性/一致性）   │
  │   5. Memory: 提取关键信息 → 存入长期记忆              │
  │   6. AgentBus: Agent 间委派（delegate_task）        │
  └──────┬──────────┬──────────┬────────────────────┘
         │          │          │
    ┌────▼───┐  ┌───▼────┐  ┌──▼──────┐
    │Weather │◄─┤delegate├─►│ Stock   │
    │ Agent  │  │  task   │  │  Agent  │
    └────────┘  └────────┘  └─────────┘

每个 Agent = 专家 system prompt + 专属 MCP 工具 + delegate_task + 独立 ReAct 推理循环

Agent 间通信:
  - Agent 可以在 ReAct 循环中调用 delegate_task 委派子任务给其他 Agent
  - 委派链深度限制 MAX_DELEGATION_DEPTH=3，循环委派会被拦截
  - 委派结果作为 tool result 返回给委派方，委派方负责整合到最终回答

和 orchestrator_mcp.py 的区别:
  - orchestrator_mcp: 单 LLM → 并行调用工具 → 汇总（工具没有推理能力）
  - multi_agent: Router → 选择 Agent → 每个 Agent 独立多步推理 → 汇总
    且 Agent 之间可以互相委派子任务
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

# 复用 MCP 连接管理器
from orchestrator_mcp import MCPHub

# LangFuse 追踪
from tracing import get_langfuse, flush, NOOP

# Human-in-the-Loop
from hitl import HITLManager, HITLDecision, HITLMode

# Memory 系统
from memory_lib import ConversationMemory, LongTermMemory

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
  - 如果任务涉及股票/金融等其他领域，使用 delegate_task 委派给对应专家
  - 只有完全无法处理且无法委派时，才回复 "SKIP: 此任务不在我的专业领域内"
  - 委派得到的结果要整合进你的最终回答
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
  - 如果任务涉及天气等其他领域，使用 delegate_task 委派给对应专家
  - 只有完全无法处理且无法委派时，才回复 "SKIP: 此任务不在我的专业领域内"
  - 委派得到的结果要整合进你的最终回答
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
# Agent 委派工具 — 让 Agent 之间可以互相求助
# ============================================================
def make_delegate_tool(agent_registry: dict[str, dict], self_id: str) -> dict | None:
    """为某个 agent 生成 delegate_task 工具 schema。

    列出除了自己以外的所有可用 agent，LLM 可以选择委派给谁。
    如果只有自己一个 agent，返回 None（不需要委派工具）。
    """
    others = {
        aid: info for aid, info in agent_registry.items() if aid != self_id
    }
    if not others:
        return None

    descriptions = "\n".join(
        f"  - {aid}: {info['name']}"
        for aid, info in others.items()
    )
    return {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": (
                f"将子任务委派给其他专家 agent。"
                f"当你需要其他领域的专业分析时调用此工具。"
                f"\n\n可委派的 agent:\n{descriptions}"
                f"\n\n委派后你会收到目标 agent 的分析结果，"
                f"请将其整合到你的最终回答中。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "enum": list(others.keys()),
                        "description": "目标 agent 的 ID",
                    },
                    "task": {
                        "type": "string",
                        "description": "委派给目标 agent 的具体子任务，要清晰明确，用中文",
                    },
                },
                "required": ["agent_id", "task"],
            },
        },
    }


# ============================================================
# Agent — 独立 ReAct 推理循环
# ============================================================
MAX_DELEGATION_DEPTH = 3  # 最大委派深度，防止无限委派


class Agent:
    """一个有独立推理能力的 Agent，支持向其他 Agent 委派子任务"""

    def __init__(
        self,
        agent_id: str,
        config: dict,
        hub: MCPHub,
        openai_tools: list[dict],
        hitl_manager: HITLManager | None = None,
        delegate_tool: dict | None = None,
        agents_info: dict[str, dict] | None = None,
    ):
        self.agent_id = agent_id
        self.name = config["name"]
        self.emoji = config.get("emoji", "\U0001f916")
        self.system_prompt = config["system_prompt"]
        self.model = config.get("model", "gpt-4o-mini")
        self.max_steps = config.get("max_steps", 5)
        self.tools = list(openai_tools)
        if delegate_tool:
            self.tools.append(delegate_tool)
        self._has_delegate = delegate_tool is not None
        self.hub = hub
        self.hitl = hitl_manager or HITLManager()
        self.agents_info = agents_info or {}

    async def run(
        self,
        task: str,
        verbose: bool = True,
        langfuse_trace=None,
        delegate_handler=None,
        delegation_depth: int = 0,
        delegation_chain: tuple[str, ...] = (),
    ) -> str:
        """
        ReAct 推理循环:
          Thought → Action(调用工具) → Observation → Thought → ... → Final Answer

        langfuse_trace: 可选的 LangFuse trace 对象
        delegate_handler: async (target_id, task) -> str，用于处理委派
        delegation_depth: 当前委派深度，顶层调用为 0
        delegation_chain: 委派链上的 agent ID 序列，用于防循环
        """
        trace = langfuse_trace or NOOP
        agent_span = trace.span(
            name=f"agent:{self.agent_id}",
            input={"task": task},
            metadata={
                "agent_name": self.name,
                "model": self.model,
                "max_steps": self.max_steps,
                "delegation_depth": delegation_depth,
                "delegation_chain": list(delegation_chain),
            },
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        try:
            for step in range(self.max_steps):
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                )
                msg = response.choices[0].message

                # ---- 记录 LLM generation ----
                gen = agent_span.generation(
                    name=f"step_{step + 1}_llm",
                    model=self.model,
                    input={
                        "message_count": len(messages),
                        "last_role": messages[-1]["role"],
                    },
                )
                if response.usage:
                    gen.update(
                        output=msg.content or ("[tool_calls: " + ", ".join(
                            tc.function.name for tc in (msg.tool_calls or [])
                        ) + "]"),
                        usage={
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        },
                        metadata={"step": step + 1, "has_tool_calls": bool(msg.tool_calls)},
                    )
                gen.end()

                # Agent 认为任务完成 → 返回最终回答
                if not msg.tool_calls:
                    if verbose:
                        print(f"  [{self.emoji} {self.name}] OK ({step + 1} step(s))")
                    agent_span.end(output={"answer": msg.content})
                    return msg.content or ""

                # Agent 需要调用工具
                messages.append(msg)
                if verbose:
                    names = [tc.function.name for tc in msg.tool_calls]
                    print(f"  [{self.emoji} {self.name}] step {step + 1}: {names}")

                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    args = json.loads(tc.function.arguments)

                    # ---- 处理 Agent 委派 ----
                    if tool_name == "delegate_task":
                        target_id = args.get("agent_id", "")
                        subtask = args.get("task", "")

                        if verbose:
                            target_name = self.agents_info.get(target_id, {}).get("name", target_id) if hasattr(self, "agents_info") else target_id
                            print(f"  [{self.emoji} {self.name}] → delegates to [{target_id}] \"{subtask[:60]}...\"")

                        if delegate_handler is None:
                            result = "Error: 委派功能不可用（没有可用的其他 agent）"
                        elif delegation_depth >= MAX_DELEGATION_DEPTH:
                            result = "Error: 已达到最大委派深度，请用已有信息给出回答"
                        elif target_id in delegation_chain:
                            result = f"Error: Agent '{target_id}' 已在委派链中（{delegation_chain}），不能循环委派"
                        else:
                            try:
                                result = await delegate_handler(
                                    from_agent_id=self.agent_id,
                                    target_id=target_id,
                                    task=subtask,
                                    depth=delegation_depth + 1,
                                    chain=delegation_chain + (self.agent_id,),
                                )
                            except Exception as e:
                                result = f"委派失败: {e}"

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                        continue

                    # ---- HITL: 工具调用审批 ----
                    decision = await self.hitl.approve_tool_call(
                        agent_id=self.agent_id,
                        agent_name=self.name,
                        tool_name=tool_name,
                        args=args,
                    )
                    if decision == HITLDecision.REJECT:
                        result = "Tool call rejected by user."
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                        continue
                    elif decision == HITLDecision.SKIP:
                        result = "[User chose to skip this tool call. Continue without this data.]"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                        continue

                    # ---- 记录工具调用 span ----
                    tool_span = agent_span.span(
                        name=f"tool:{tool_name}",
                        input=args,
                    )

                    try:
                        call_mcp = self.hub.get_caller(tool_name)
                        result = await call_mcp(**args)
                        tool_span.end(output={"result": result[:500]})
                    except Exception as e:
                        tool_span.end(output={"error": str(e)}, level="ERROR")
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
            # 记录强制总结的 generation
            gen = agent_span.generation(
                name="forced_summary_llm",
                model=self.model,
                input={"message_count": len(messages)},
            )
            if final.usage:
                gen.update(
                    output=final.choices[0].message.content,
                    usage={
                        "prompt_tokens": final.usage.prompt_tokens,
                        "completion_tokens": final.usage.completion_tokens,
                        "total_tokens": final.usage.total_tokens,
                    },
                )
            gen.end()

            answer = final.choices[0].message.content or ""
            agent_span.end(output={"answer": answer, "forced_summary": True})
            return answer

        except Exception as e:
            agent_span.end(output={"error": str(e)}, level="ERROR")
            raise



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

REFLECTOR_PROMPT = """\
你是一个严格的质量审查员。你需要审查以下 AI 助手的回答是否合格。

用户原始问题:
{user_input}

各专家原始分析结果:
{agent_results}

AI 助手的整合回答:
{aggregated_answer}

请从以下三个维度审查:

1. **完整性** — 用户提出的每个子问题都被回答了吗？有没有遗漏？
2. **准确性** — 整合回答中的所有数据（数字、日期、名称等）是否和专家原始结果一致？有没有编造或修改数据？
3. **一致性** — 如果多个专家提供了相关信息，它们之间是否有矛盾？如有矛盾，是否被妥善处理？

判断规则:
- 如果回答完全合格（无遗漏、无编造、无矛盾），直接原样输出该回答，不要加任何修改或前缀。
- 如果回答有小问题（遗漏了某个子问题、某个数据不准确、表述不够清晰），请在原回答基础上修正，输出修正后的完整回答。不要输出你的审查过程，只输出修正后的回答。
- 如果回答有严重问题（大量编造数据、完全未回答用户问题），请基于专家原始结果重新生成一个正确的回答。

重要: 你的输出就是最终给用户的回答。不要加"审查结果："、"修正后："等前缀，直接输出最终答案。用中文。"""



class MultiAgentOrchestrator:
    def __init__(
        self,
        agents: dict[str, Agent],
        hitl_manager: HITLManager | None = None,
        conv_memory: ConversationMemory | None = None,
        long_memory: LongTermMemory | None = None,
    ):
        self.agents = agents
        self.hitl = hitl_manager or HITLManager()
        self.conv_memory = conv_memory or ConversationMemory()
        self.long_memory = long_memory
        self._current_trace = NOOP

    async def _handle_delegation(
        self,
        from_agent_id: str,
        target_id: str,
        task: str,
        depth: int,
        chain: tuple[str, ...],
    ) -> str:
        """处理 Agent 委派：运行目标 agent 并返回其结果。

        目标和深度检查已在 Agent.run() 中完成，这里只负责执行。
        """
        target = self.agents.get(target_id)
        if target is None:
            return f"Error: Agent '{target_id}' 不存在，可用: {list(self.agents.keys())}"
        return await target.run(
            task=task,
            verbose=True,
            langfuse_trace=self._current_trace,
            delegate_handler=self._make_delegate_handler(),
            delegation_depth=depth,
            delegation_chain=chain,
        )

    def _make_delegate_handler(self):
        """创建一个委派处理器闭包，捕获当前 orchestrator 的 agents 引用。"""
        orch = self

        async def handler(from_agent_id: str, target_id: str, task: str,
                          depth: int, chain: tuple[str, ...]) -> str:
            return await orch._handle_delegation(from_agent_id, target_id, task, depth, chain)

        return handler

    def _build_memory_context(self, user_input: str) -> str:
        """将短/长期记忆注入用户输入，让 Router 获得更完整的上下文"""
        parts = []

        # 长期记忆
        if self.long_memory is not None and self.long_memory.count > 0:
            relevant = self.long_memory.search(user_input, top_k=3)
            if relevant:
                parts.append("## 相关长期记忆:\n" + "\n".join(f"- {m}" for m in relevant))

        # 短期对话历史
        if not self.conv_memory.is_empty():
            parts.append(self.conv_memory.get_context())

        if not parts:
            return user_input

        # 记忆在前，用户输入在后
        parts.append(f"## 用户当前问题:\n{user_input}")
        return "\n\n".join(parts)

    async def _save_memories(self, user_input: str, answer: str):
        """提取关键信息到长期记忆（后台），并保存到短期对话历史"""
        self.conv_memory.add_turn(user_input, answer)
        if self.long_memory is not None:
            # 后台提取，不阻塞用户看到回答
            asyncio.create_task(self.long_memory.extract_and_save(user_input, answer))

    async def _route(self, user_input: str, langfuse_trace=None) -> dict[str, str]:
        """Router: 分析用户意图，返回 {agent_id: subtask}"""
        trace = langfuse_trace or NOOP
        router_span = trace.span(name="router", input={"user_input": user_input})

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

        # 记录 router LLM generation
        gen = router_span.generation(
            name="router_llm",
            model="gpt-4o-mini",
            input={"prompt": router_prompt[:1000], "user_input": user_input},
        )
        if response.usage:
            gen.update(
                output=raw,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )
        gen.end()

        # 清理可能的 markdown 代码块
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  [Router] JSON parse failed, fallback: all agents")
            plan = {"tasks": {aid: user_input for aid in self.agents}}

        tasks = plan.get("tasks", {})
        router_span.end(output={"activated_agents": list(tasks.keys()), "tasks": tasks})
        return tasks



    async def _reflect(
        self,
        user_input: str,
        aggregated_answer: str,
        agent_results: dict[str, str],
        langfuse_trace=None,
    ) -> str:
        """Reflector: 审查 Aggregator 输出是否合格，不合格则修正"""
        trace = langfuse_trace or NOOP
        refl_span = trace.span(name="reflector", input={"answer_length": len(aggregated_answer)})

        results_text = "\n\n---\n\n".join(
            f"### {self.agents[aid].name}:\n{text}"
            for aid, text in agent_results.items()
        )
        refl_prompt = REFLECTOR_PROMPT.format(
            user_input=user_input,
            agent_results=results_text,
            aggregated_answer=aggregated_answer,
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": refl_prompt}],
            temperature=0.1,
        )

        refl_gen = refl_span.generation(
            name="reflector_llm",
            model="gpt-4o-mini",
            input={"prompt": refl_prompt[:1000]},
        )
        if response.usage:
            refl_gen.update(
                output=response.choices[0].message.content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )
        refl_gen.end()

        refined = response.choices[0].message.content or aggregated_answer
        refl_span.end(output={"refined_answer": refined})

        # 如果修正后的答案和原答案不同，打日志
        if refined.strip() != aggregated_answer.strip():
            print(f"  [Reflector] answer refined (changed)")

        return refined

    async def run(self, user_input: str, verbose: bool = True) -> str:
        """完整的 multi-agent 流程: memory → route → dispatch → aggregate → reflect → memory"""

        self.hitl.reset()

        # ---- 0. Memory: 注入上下文 ----
        enhanced_input = self._build_memory_context(user_input)
        if verbose:
            print(f"\n{'─' * 50}")
            print(f"[User] {user_input}")
            if self.long_memory and self.long_memory.count > 0:
                print(f"[Memory] {self.long_memory.count} long-term, "
                      f"{len(self.conv_memory.history) // 2} conversation turns")

        # 创建 LangFuse trace（如果已配置）
        langfuse = get_langfuse()
        print("LangFuse:", langfuse)
        trace = langfuse.trace(
            name=f"MultiAgent: {user_input[:80]}",
            input={"user_input": user_input, "enhanced_input": enhanced_input},
            metadata={"available_agents": list(self.agents.keys())},
        ) if langfuse else NOOP
        self._current_trace = trace

        # ---- 1. Route: 分析意图，分配子任务 ----
        if verbose:
            print(f"\n[Router] analyzing intent...")
        tasks = await self._route(enhanced_input, langfuse_trace=trace)

        if not tasks:
            trace.update(output={"error": "no agents matched"})
            return "抱歉，我无法判断该问题需要哪些专家来处理。"

        # ---- HITL: Router 确认 ----
        tasks = await self.hitl.confirm_router_plan(tasks, self.agents)
        if not tasks:
            trace.update(output={"error": "user cancelled router plan"})
            return "已取消。"

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
                result = await agent.run(
                    task,
                    verbose=verbose,
                    langfuse_trace=trace,
                    delegate_handler=self._make_delegate_handler(),
                )
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
            trace.update(output={"error": "all agents skipped"})
            return "所有专家都认为此问题不在其专业领域内，请尝试换一种问法。"

        # 只有一个 agent 有结果 → 也走 Reflection 质检
        if len(valid) == 1:
            answer = list(valid.values())[0]
            if verbose:
                print(f"\n[Reflector] reviewing single-agent answer...")
            refined = await self._reflect(user_input, answer, valid, langfuse_trace=trace)
            trace.update(output={"answer": refined, "aggregation": "single_agent_reflected"})
            # ---- Memory: 保存 ----
            await self._save_memories(user_input, refined)
            return refined

        # 多个 agent → LLM 汇总
        results_text = "\n\n---\n\n".join(
            f"### {self.agents[aid].name}:\n{text}"
            for aid, text in valid.items()
        )
        agg_prompt = AGGREGATOR_PROMPT.format(
            user_input=user_input,
            agent_results=results_text,
        )

        # ---- 记录 Aggregator span + generation ----
        agg_span = trace.span(name="aggregator", input={"agent_count": len(valid)})
        final = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": agg_prompt}],
        )
        agg_gen = agg_span.generation(
            name="aggregator_llm",
            model="gpt-4o-mini",
            input={"prompt": agg_prompt[:1000]},
        )
        if final.usage:
            agg_gen.update(
                output=final.choices[0].message.content,
                usage={
                    "prompt_tokens": final.usage.prompt_tokens,
                    "completion_tokens": final.usage.completion_tokens,
                    "total_tokens": final.usage.total_tokens,
                },
            )
        agg_gen.end()
        answer = final.choices[0].message.content or ""
        agg_span.end(output={"answer": answer})

        # ---- 4. Reflect: 质检 + 修正 ----
        if verbose:
            print(f"\n[Reflector] reviewing aggregated answer...")
        refined = await self._reflect(user_input, answer, valid, langfuse_trace=trace)
        trace.update(output={"answer": refined, "aggregation": "multi_agent_reflected"})
        # ---- Memory: 保存 ----
        await self._save_memories(user_input, refined)
        return refined


# ============================================================
# 启动入口
# ============================================================
async def main():
    hub = MCPHub()

    # ---- HITL 模式选择 ----
    print("=" * 50)
    print("Multi-Agent System (with Memory & Human-in-the-Loop)")
    print("=" * 50)
    print("\nHITL 模式:")
    print("  1. OFF    — 全自动执行（默认）")
    print("  2. ROUTER — 确认 Router 的 Agent 派发计划")
    print("  3. TOOLS  — 审批每个工具调用")
    print("  4. FULL   — Router + Tools 都需确认")

    mode_map = {
        "1": HITLMode.OFF, "": HITLMode.OFF,
        "2": HITLMode.ROUTER_ONLY,
        "3": HITLMode.TOOLS_ONLY,
        "4": HITLMode.FULL,
    }
    mode_input = input("\n选择模式 [1]: ").strip()
    print(f"DEBUG: repr={mode_input!r}")  # 加这行看真实输入了什么
    hitl_mode = mode_map.get(mode_input, HITLMode.OFF)
    hitl = HITLManager(mode=hitl_mode)
    print(f"HITL: {hitl_mode.value}\n")

    # 0. 初始化 Memory
    conv_memory = ConversationMemory(max_turns=10)
    memory_store_path = BASE_DIR / "agent_memories.json"
    long_memory = LongTermMemory(store_path=memory_store_path, client=client)
    print(f"\nMemory: {long_memory.count} long-term memories loaded from {memory_store_path.name}")

    # 1. 连接所有 MCP Server
    print("Connecting to MCP servers...")
    all_tools = await hub.connect_all(MCP_SERVERS)
    print(f"Total MCP tools discovered: {len(all_tools)}")

    # 2. 为每个 Agent 装配专属工具 + 委派工具（共享同一个 HITLManager）
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

        delegate_tool = make_delegate_tool(AGENT_DEFS, aid)
        agents[aid] = Agent(
            aid, config, hub, agent_tools,
            hitl_manager=hitl,
            delegate_tool=delegate_tool,
            agents_info={k: {"name": v["name"]} for k, v in AGENT_DEFS.items()},
        )
        tool_list = ", ".join(t["function"]["name"] for t in agent_tools)
        if delegate_tool:
            tool_list += " + delegate_task"
        print(f"  {config['emoji']} {config['name']}: {tool_list}")

    if not agents:
        print("\nNo agents available. Exiting.")
        await hub.close()
        return

    # 3. 创建 Orchestrator（共享 HITLManager + Memory）
    orch = MultiAgentOrchestrator(
        agents,
        hitl_manager=hitl,
        conv_memory=conv_memory,
        long_memory=long_memory,
    )

    print(f"\n{'=' * 50}")
    print(f"Ready! {len(agents)} agents standing by.")
    print(f"")
    print(f"Multi-domain queries (Router dispatches to multiple agents):")
    print(f"  '查 TSLA 股价和旧金山天气'")
    print(f"  '旧金山今天适合户外运动吗'")
    print(f"Single-domain with delegation (agent delegates to another):")
    print(f"  'TSLA 股价怎么样？另外帮我查一下纽约天气'")
    print(f"  '旧金山天气如何？这对 AAPL 股价有什么影响'")
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
        flush()
        print("\nBye!")


if __name__ == "__main__":
    asyncio.run(main())
