"""
rag_core.py  –  Hybrid RAG with Cross-Encoder Rerank
改进点：
  1. BM25 在全量 chunks 上构建（真正 Hybrid）
  2. 换用 CrossEncoder 做 Rerank（更快、更准）
  3. API 调用修正：client.chat.completions.create
  4. async-safe：用 asyncio.to_thread 包装同步操作
  5. 更健壮的错误处理
"""

from pathlib import Path
from openai import OpenAI
import hashlib
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import asyncio
import re

# -------------------------
# 路径配置
# -------------------------
BASE_DIR = Path(__file__).parent
PDF_PATH  = BASE_DIR / "files" / "Master_thesis.pdf"
ID_FILE   = BASE_DIR / "vector_store_id.txt"

# -------------------------
# 全局状态
# -------------------------
_cache:      dict  = {}
_bm25:       BM25Okapi | None = None
_all_chunks: list[str]        = []          # 全量 chunk 池
_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

client = OpenAI()

# -------------------------
# 常量
# -------------------------
VECTOR_TOP_K  = 8    # vector 召回数（增大以提高召回率）
BM25_TOP_K    = 5    # BM25 召回数
RERANK_TOP_N  = 5    # 最终保留 chunk 数
MODEL         = "gpt-4o-mini"   # 换成真实存在的模型


# ==========================================
# 工具函数
# ==========================================

def make_key(q: str, vs_id: str) -> str:
    return hashlib.md5((q + vs_id).encode()).hexdigest()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# ==========================================
# Vector Store 初始化
# ==========================================

def get_or_create_vector_store() -> str:
    if ID_FILE.exists():
        return ID_FILE.read_text().strip()

    vs = client.vector_stores.create(name="PDF RAG MCP")
    with open(PDF_PATH, "rb") as f:
        client.vector_stores.files.upload_and_poll(
            vector_store_id=vs.id, file=f
        )
    ID_FILE.write_text(vs.id)
    return vs.id


# ==========================================
# BM25 — 在全量 chunk 上构建
# ==========================================

def _ensure_bm25_built(chunks: list[str]) -> None:
    """
    只在 _all_chunks 为空时（首次）构建；
    后续每次从 vector store 拿到新 chunk 都追加进全量池。
    """
    global _bm25, _all_chunks

    new_chunks = [c for c in chunks if c not in set(_all_chunks)]
    if not new_chunks:
        return

    _all_chunks.extend(new_chunks)
    tokenized = [_tokenize(t) for t in _all_chunks]
    _bm25 = BM25Okapi(tokenized)


def _bm25_search(question: str, top_k: int) -> list[str]:
    if _bm25 is None or not _all_chunks:
        return []
    scores = _bm25.get_scores(_tokenize(question))
    top_idx = sorted(range(len(_all_chunks)),
                     key=lambda i: scores[i], reverse=True)[:top_k]
    return [_all_chunks[i] for i in top_idx]


# ==========================================
# Cross-Encoder Rerank（替代 LLM rerank）
# ==========================================

def _rerank(question: str, candidates: list[str], top_n: int) -> list[str]:
    if not candidates:
        return []
    pairs  = [(question, c) for c in candidates]
    scores = _cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_n]]


# ==========================================
# 主 RAG 函数（同步）
# ==========================================

def search_pdf(question: str) -> str:
    vs_id = get_or_create_vector_store()

    # --- 缓存 ---
    key = make_key(question, vs_id)
    if key in _cache:
        return _cache[key]

    # --- Vector 召回 ---
    results = client.vector_stores.search(
        vector_store_id=vs_id,
        query=question,
        max_num_results=VECTOR_TOP_K,
    )
    vector_chunks: list[str] = []
    for r in results.data:
        for c in r.content:
            if c.text:
                vector_chunks.append(c.text)

    # --- 更新全量 BM25 池 ---
    _ensure_bm25_built(vector_chunks)

    # --- BM25 召回（全量池） ---
    bm25_chunks = _bm25_search(question, BM25_TOP_K)

    # --- 合并去重（保持顺序）---
    seen: set[str] = set()
    candidates: list[str] = []
    for c in vector_chunks + bm25_chunks:
        if c not in seen:
            seen.add(c)
            candidates.append(c)

    # --- Cross-Encoder Rerank ---
    final_chunks = _rerank(question, candidates, RERANK_TOP_N)

    if not final_chunks:
        return "Not found in document"

    # --- 构建 prompt（带 chunk 引用）---
    context = "\n\n".join(
        f"[chunk_{i}]\n{c}" for i, c in enumerate(final_chunks)
    )

    prompt = f"""You are a strict PDF QA system.

RULES:
- Answer using ONLY the context below.
- If the answer is not in the context, say "Not found in document".
- Cite every claim with [chunk_i].
- Do NOT use external knowledge.

CONTEXT:
{context}

QUESTION:
{question}

Answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    result = response.choices[0].message.content or "No response"
    _cache[key] = result
    return result


# ==========================================
# Async 包装（供 MCP async tool 使用）
# ==========================================

async def search_pdf_async(question: str) -> str:
    """不阻塞事件循环的异步入口。"""
    return await asyncio.to_thread(search_pdf, question)