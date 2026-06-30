"""
Human-in-the-Loop (HITL) Manager

Provides interactive approval points in the multi-agent pipeline:
  - Router Confirmation: review/modify Router's agent dispatch plan before execution
  - Tool Approval: approve/reject/skip individual tool calls during Agent ReAct loops

Usage:
    from hitl import HITLManager, HITLMode

    hitl = HITLManager(mode=HITLMode.FULL)
    orch = MultiAgentOrchestrator(agents, hitl_manager=hitl)
"""

from enum import Enum


class HITLMode(Enum):
    OFF = "off"           # Fully automatic, no human intervention
    ROUTER_ONLY = "router"  # Only confirm Router dispatch plan
    TOOLS_ONLY = "tools"    # Only approve individual tool calls
    FULL = "full"           # Both Router + Tool approval


class HITLDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    SKIP = "skip"                  # Return empty result, let agent continue
    AUTO_APPROVE_ALL = "auto_all"  # Approve this and all future calls from this agent


class HITLManager:
    """Manages human-in-the-loop interaction points.

    Pure terminal I/O via input(). The ask_user() method can be overridden
    or replaced to swap in a different UI (web, Slack, etc.).
    """

    def __init__(self, mode: HITLMode = HITLMode.OFF):
        self.mode = mode
        self._agent_auto_approved: set[str] = set()

    # ---- Public API ----

    @property
    def router_active(self) -> bool:
        return self.mode in (HITLMode.ROUTER_ONLY, HITLMode.FULL)

    @property
    def tools_active(self) -> bool:
        return self.mode in (HITLMode.TOOLS_ONLY, HITLMode.FULL)

    def reset(self):
        """Reset per-run state (auto-approve flags)."""
        self._agent_auto_approved.clear()

    async def confirm_router_plan(
        self, tasks: dict[str, str], agents: dict
    ) -> dict[str, str]:
        """Show Router's dispatch plan and wait for user confirmation.

        Args:
            tasks: {agent_id: subtask} from Router
            agents: {agent_id: Agent} dict for display names

        Returns:
            Confirmed/modified tasks dict, or empty dict to cancel.
            User can type a modified JSON to change tasks.
        """
        if not self.router_active or len(tasks) <= 1:
            return tasks

        print(f"\n{'─' * 50}")
        print(f"[HITL · Router] 计划派发 {len(tasks)} 个 Agent:")
        for i, (aid, task) in enumerate(tasks.items(), 1):
            agent_name = agents[aid].name if aid in agents else aid
            print(f"  {i}. {agent_name} → \"{task}\"")
        print(f"{'─' * 50}")

        user_input = await self._ask_user(
            "确认执行? (y=确认 / n=取消 / 或输入修改后的 JSON tasks): "
        )

        if not user_input:
            print("  [HITL] 已取消")
            return {}

        stripped = user_input.strip().lower()

        if stripped == "y":
            print("  [HITL] 已确认，开始执行\n")
            return tasks

        if stripped == "n":
            print("  [HITL] 已取消")
            return {}

        # Try to parse as modified JSON tasks
        try:
            import json
            new_tasks = json.loads(user_input.strip())
            if isinstance(new_tasks, dict):
                print(f"  [HITL] 已更新任务: {list(new_tasks.keys())}\n")
                return new_tasks
        except (json.JSONDecodeError, ValueError):
            pass

        print(f"  [HITL] 无法解析输入，已取消")
        return {}

    async def approve_tool_call(
        self, agent_id: str, agent_name: str, tool_name: str, args: dict
    ) -> HITLDecision:
        """Check whether a tool call should proceed.

        Called before each tool execution in the Agent ReAct loop.

        Returns:
            - APPROVE: execute the tool normally
            - REJECT: skip this tool, agent receives error message
            - SKIP: skip this tool, agent receives empty result
            - AUTO_APPROVE_ALL: approve this and all future calls from this agent
        """
        if not self.tools_active:
            return HITLDecision.APPROVE

        if agent_id in self._agent_auto_approved:
            return HITLDecision.APPROVE

        # Format args compactly
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(f"\n  [HITL · {agent_name}] 即将调用:")
        print(f"    {tool_name}({args_str})")

        user_input = await self._ask_user(
            "  批准? (y=本次/n=拒绝/a=全部批准/s=跳过): "
        )

        if not user_input:
            return HITLDecision.REJECT

        stripped = user_input.strip().lower()

        if stripped == "y":
            return HITLDecision.APPROVE
        elif stripped == "a":
            self._agent_auto_approved.add(agent_id)
            print(f"    → 后续 {agent_name} 的调用将自动批准")
            return HITLDecision.AUTO_APPROVE_ALL
        elif stripped == "s":
            return HITLDecision.SKIP
        else:
            return HITLDecision.REJECT

    # ---- Hook for UI replacement ----

    async def _ask_user(self, prompt: str) -> str:
        """Prompt the user and return their input.

        Override this method to swap in a different UI layer
        (e.g., WebSocket, Slack bot, custom CLI).
        """
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            return ""
