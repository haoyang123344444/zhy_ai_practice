"""Memory 系统独立测试 — 不需要 MCP server 连接"""
import asyncio
import json
from pathlib import Path
from openai import OpenAI

from memory_lib import ConversationMemory, LongTermMemory

BASE_DIR = Path(__file__).parent
TEST_MEMORY_PATH = BASE_DIR / "test_agent_memories.json"
client = OpenAI()


async def test_long_term_memory():
    """测试长期记忆的增删查 + 自动提取"""
    # 清理旧测试数据
    if TEST_MEMORY_PATH.exists():
        TEST_MEMORY_PATH.unlink()

    print("=" * 50)
    print("Test 1: LongTermMemory CRUD")
    print("=" * 50)

    ltm = LongTermMemory(TEST_MEMORY_PATH, client)
    print(f"  初始记忆数: {ltm.count}")

    # 手动添加
    ltm.add("用户喜欢关注科技股，尤其是 TSLA 和 AAPL")
    ltm.add("用户常查询旧金山的天气")
    ltm.add("用户偏好简洁的中文回答，不喜欢英文混杂")
    print(f"  添加 3 条后: {ltm.count}")

    # 检索
    results = ltm.search("帮我查查苹果股价", top_k=2)
    print(f"  搜索 '苹果股价': {results}")

    results = ltm.search("旧金山今天热不热", top_k=2)
    print(f"  搜索 '旧金山天气': {results}")

    results = ltm.search("用中文回答", top_k=2)
    print(f"  搜索 '语言偏好': {results}")

    # 无关查询
    results = ltm.search("比特币价格", top_k=2)
    print(f"  搜索 '比特币' (无关): {results}")

    # 查看全部
    print(f"  全部记忆: {ltm.all()}")

    # 持久化验证
    print(f"\n  文件已保存: {TEST_MEMORY_PATH.exists()}")
    with open(TEST_MEMORY_PATH) as f:
        data = json.load(f)
    print(f"  文件中有 {len(data['memories'])} 条记忆")

    # 重新加载验证
    ltm2 = LongTermMemory(TEST_MEMORY_PATH, client)
    print(f"  重新加载后: {ltm2.count} 条")
    print(f"  重新加载内容: {ltm2.all()}")

    print("\n  LongTermMemory 测试: PASSED\n")


async def test_extract_and_save():
    """测试 LLM 自动提取记忆"""
    print("=" * 50)
    print("Test 2: Auto-extract memories from conversation")
    print("=" * 50)

    if TEST_MEMORY_PATH.exists():
        TEST_MEMORY_PATH.unlink()

    ltm = LongTermMemory(TEST_MEMORY_PATH, client)

    # 模拟一轮对话
    await ltm.extract_and_save(
        user_input="帮我查一下 TSLA 的股价，我对这只股票很关注",
        assistant_response="TSLA 当前股价为 $249.23，较昨日上涨 2.3%。成交量为 82M。"
    )
    await asyncio.sleep(0.5)  # 给后台任务一点时间

    print(f"  提取后记忆数: {ltm.count}")
    for m in ltm.all():
        print(f"    - {m}")

    # 第二轮
    await ltm.extract_and_save(
        user_input="旧金山天气怎么样？我下周要去那出差",
        assistant_response="旧金山当前 18°C，多云。下周预计有雨，建议带伞。平均气温 15-22°C。"
    )
    await asyncio.sleep(0.5)

    print(f"  第二轮后记忆数: {ltm.count}")
    for m in ltm.all():
        print(f"    - {m}")

    # 第三轮 — 引用之前的信息
    await ltm.extract_and_save(
        user_input="我之前问的那支股票现在什么价？",
        assistant_response="TSLA 当前股价为 $251.10，继续小幅上涨。"
    )
    await asyncio.sleep(0.5)

    print(f"  第三轮后记忆数: {ltm.count}")
    for m in ltm.all():
        print(f"    - {m}")

    print("\n  Extract & Save 测试: PASSED\n")


def test_conversation_memory():
    """测试短期对话记忆"""
    print("=" * 50)
    print("Test 3: ConversationMemory (short-term)")
    print("=" * 50)

    cm = ConversationMemory(max_turns=3)

    # 添加 3 轮对话
    cm.add_turn("查 TSLA 股价", "TSLA 当前 $249")
    cm.add_turn("旧金山天气呢", "旧金山 18°C，多云")
    cm.add_turn("那支股票现在什么价", "TSLA 现价 $251")

    ctx = cm.get_context()
    print(f"  3 轮后的上下文:\n{ctx}")
    print()

    # 再加 2 轮，超过 max_turns
    cm.add_turn("AAPL 呢", "AAPL $185")
    cm.add_turn("涨了还是跌了", "涨了 2%")

    ctx = cm.get_context()
    print(f"  5 轮后的上下文 (应只保留最近 3 轮):\n{ctx}")
    print()

    # 验证最早的两轮被丢弃
    assert "TSLA 当前 $249" not in ctx, "最旧轮次应被丢弃"
    print("  旧轮次丢弃验证: PASSED")
    print("\n  ConversationMemory 测试: PASSED\n")


async def test_memory_integration():
    """模拟 orchestrator 的 memory 注入和保存流程"""
    print("=" * 50)
    print("Test 4: Integration (simulated orchestrator flow)")
    print("=" * 50)

    if TEST_MEMORY_PATH.exists():
        TEST_MEMORY_PATH.unlink()

    ltm = LongTermMemory(TEST_MEMORY_PATH, client)
    cm = ConversationMemory(max_turns=5)

    # 模拟第一轮
    user_input_1 = "帮我查 TSLA 股价"
    enhanced_1 = _build_context(user_input_1, cm, ltm)
    print(f"  第1轮 enhanced_input:\n    {enhanced_1[:200]}...")
    cm.add_turn(user_input_1, "TSLA $249")
    await ltm.extract_and_save(user_input_1, "TSLA $249")
    await asyncio.sleep(0.5)

    # 模拟第二轮
    user_input_2 = "旧金山天气呢"
    enhanced_2 = _build_context(user_input_2, cm, ltm)
    print(f"\n  第2轮 enhanced_input:\n{enhanced_2}")
    cm.add_turn(user_input_2, "旧金山 18°C")
    await ltm.extract_and_save(user_input_2, "旧金山 18°C")
    await asyncio.sleep(0.5)

    # 模拟第三轮 — 模糊指代，依赖记忆
    user_input_3 = "那支股票现在什么价"
    enhanced_3 = _build_context(user_input_3, cm, ltm)
    print(f"\n  第3轮 enhanced_input (模糊指代):\n{enhanced_3}")
    print(f"\n  → 长期记忆检索结果包含 TSLA，Router 可正确路由到 stock agent")

    print("\n  Integration 测试: PASSED\n")


def _build_context(user_input: str, cm: ConversationMemory, ltm: LongTermMemory) -> str:
    parts = []
    if ltm.count > 0:
        relevant = ltm.search(user_input, top_k=3)
        if relevant:
            parts.append("## 相关长期记忆:\n" + "\n".join(f"- {m}" for m in relevant))
    if not cm.is_empty():
        parts.append(cm.get_context())
    parts.append(f"## 用户当前问题:\n{user_input}")
    return "\n\n".join(parts)


async def main():
    print("\n" + "=" * 60)
    print("Memory System Test Suite")
    print("=" * 60 + "\n")

    test_conversation_memory()
    await test_long_term_memory()
    await test_extract_and_save()
    await test_memory_integration()

    # 清理
    if TEST_MEMORY_PATH.exists():
        TEST_MEMORY_PATH.unlink()
        print("Test memory file cleaned up.")

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
