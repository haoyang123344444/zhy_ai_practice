"""
Memory 系统 — 短期对话记忆 + 长期向量记忆

  短期记忆 (ConversationMemory)
    - 会话内滑动窗口，保留最近 N 轮对话
    - 注入到 Router 和 Aggregator 的上下文中

  长期记忆 (LongTermMemory)
    - 基于 OpenAI Embeddings + 本地 JSON 存储
    - 跨会话持久化，自动提取关键信息
    - 检索时用余弦相似度返回最相关记忆
"""

import json
from pathlib import Path

import numpy as np
from openai import OpenAI


# ============================================================
# 短期记忆 — 会话内上下文窗口
# ============================================================
class ConversationMemory:
    """滑动窗口式对话历史"""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: list[dict] = []

    def add_turn(self, user_input: str, assistant_response: str):
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": assistant_response})
        # 保持最近 N 轮
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]
        
        

    def get_context(self) -> str:
        """返回格式化的对话历史，用于注入 prompt"""
        if not self.history:
            return ""
        lines = ["## 对话历史 (最近对话):"]
        for i, msg in enumerate(self.history):
            role = "用户" if msg["role"] == "user" else "AI助手"
            # 截断过长的单条消息
            text = msg["content"]
            if len(text) > 500:
                text = text[:500] + "..."
            lines.append(f"{role}: {text}")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return len(self.history) == 0


# ============================================================
# 长期记忆 — Embeddings + 本地 JSON
# ============================================================
MEMORY_EXTRACTOR_PROMPT = """\
基于以下对话，提取值得长期记住的信息。只提取用户相关的、可能在未来对话中有用的信息。

规则:
- 提取用户偏好、身份、兴趣、习惯等个人信息
- 提取重要的上下文（例如用户关注的股票、关心的城市等）
- 每条记忆用一句话概括，简洁明确
- 如果本轮对话没有任何值得长期记住的信息，返回空列表 []
- 不要记录临时性的、一次性的查询内容

对话:
{conversation}

返回 JSON 对象（必须），格式如下:
{{"memories": ["用户偏好...", "用户关注..."]}} 或 {{"memories": []}} 如果没有值得记录的信息"""


class LongTermMemory:
    """基于 embedding 的长期记忆，持久化到本地 JSON"""

    def __init__(self, store_path: Path, client: OpenAI, embedding_model: str = "text-embedding-3-small"):
        self.store_path = store_path
        self.client = client
        self.embedding_model = embedding_model
        self._memories: list[dict] = []
        self._load()

    # ---- 持久化 ----
    def _load(self):
        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text())
                self._memories = data.get("memories", [])
            except (json.JSONDecodeError, KeyError):
                self._memories = []
    


    def _save(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps({"memories": self._memories}, ensure_ascii=False, indent=2))

    # ---- CRUD ----
    def add(self, text: str):
        """添加一条长期记忆"""
        resp = self.client.embeddings.create(model=self.embedding_model, input=text)
        embedding = resp.data[0].embedding
        self._memories.append({"text": text, "embedding": embedding})
        self._save()

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """检索与 query 最相关的长期记忆"""
        if not self._memories:
            return []

        resp = self.client.embeddings.create(model=self.embedding_model, input=query)
        query_emb = resp.data[0].embedding

        scored = []
        for m in self._memories:
            sim = self._cosine_sim(query_emb, m["embedding"])
            scored.append((sim, m["text"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        

        return [text for sim, text in scored[:top_k] if sim > 0.3]

    # ---- 自动提取 ----
    async def extract_and_save(self, user_input: str, assistant_response: str):
        """从一轮对话中提取值得长期记住的信息并保存"""
        conversation = f"用户: {user_input}\nAI助手: {assistant_response[:1000]}"

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": MEMORY_EXTRACTOR_PROMPT.format(conversation=conversation)},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "[]"
            facts = json.loads(raw)
            if isinstance(facts, dict):
                # 兼容 {"memories": [...]} 格式
                facts = facts.get("memories", [])
            if isinstance(facts, list):
                for fact in facts:
                    if isinstance(fact, str) and fact.strip():
                        self.add(fact.strip())
                        print(f"  [Memory] saved: {fact.strip()[:80]}")
        except Exception as e:
            print(f"  [Memory] extract error: {e}")

    # ---- 工具 ----
    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        a_np = np.array(a)
        b_np = np.array(b)
        denom = np.linalg.norm(a_np) * np.linalg.norm(b_np)
        if denom == 0:
            return 0.0
        return float(np.dot(a_np, b_np) / denom)

    @property
    def count(self) -> int:
        return len(self._memories)

    def all(self) -> list[str]:
        return [m["text"] for m in self._memories]
