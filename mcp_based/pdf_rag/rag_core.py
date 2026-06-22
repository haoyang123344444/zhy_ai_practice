from pathlib import Path
from openai import OpenAI
import hashlib
from rank_bm25 import BM25Okapi
import re

BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "files" / "Master_thesis.pdf"
ID_FILE = BASE_DIR / "vector_store_id.txt"

_cache = {}
_bm25 = None
_chunks = None

client = OpenAI()


# -------------------------
# cache key
# -------------------------
def make_key(q: str, vs_id: str):
    return hashlib.md5((q + vs_id).encode()).hexdigest()


# -------------------------
# BM25 builder
# -------------------------
def build_bm25(texts):
    global _bm25, _chunks

    _chunks = texts
    tokenized = [re.findall(r"\w+", t.lower()) for t in texts]
    _bm25 = BM25Okapi(tokenized)


# -------------------------
# vector store init
# -------------------------
def get_or_create_vector_store() -> str:
    if ID_FILE.exists():
        return ID_FILE.read_text().strip()

    vector_store = client.vector_stores.create(name="PDF RAG MCP")

    with open(PDF_PATH, "rb") as f:
        client.vector_stores.files.upload_and_poll(
            vector_store_id=vector_store.id,
            file=f,
        )

    ID_FILE.write_text(vector_store.id)
    return vector_store.id


# -------------------------
# rerank step (GPT-based simple version)
# -------------------------
def rerank(question, chunks):
    prompt = f"""
You are a reranker.

Select the TOP 3 most relevant chunks.

QUESTION:
{question}

CHUNKS:
{chr(10).join([f"[{i}] {c}" for i, c in enumerate(chunks)])}

Return ONLY comma-separated indices like:
0,2,3
"""

    res = client.responses.create(
        model="gpt-5.5",
        input=prompt,
    )

    text = res.output_text.strip()
    idxs = [int(x) for x in text.split(",") if x.strip().isdigit()]

    return [chunks[i] for i in idxs[:3]]


# -------------------------
# main RAG function
# -------------------------
def search_pdf(question: str) -> str:

    vector_store_id = get_or_create_vector_store()

    # -------------------------
    # cache FIRST
    # -------------------------
    key = make_key(question, vector_store_id)
    if key in _cache:
        return _cache[key]

    # -------------------------
    # VECTOR retrieval
    # -------------------------
    results = client.vector_stores.search(
        vector_store_id=vector_store_id,
        query=question,
        max_num_results=5,
    )

    vector_chunks = []
    for r in results.data:
        for c in r.content:
            vector_chunks.append(c.text)

    # -------------------------
    # BM25 retrieval (TRUE HYBRID)
    # -------------------------
    if _bm25 is None:
        build_bm25(vector_chunks)

    tokenized_q = re.findall(r"\w+", question.lower())
    bm25_scores = _bm25.get_scores(tokenized_q)

    bm25_top_idx = sorted(
        range(len(_chunks)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:3]

    bm25_chunks = [_chunks[i] for i in bm25_top_idx]

    # -------------------------
    # MERGE (hybrid)
    # -------------------------
    candidates = list(set(vector_chunks + bm25_chunks))

    # -------------------------
    # RERANK (IMPORTANT STEP)
    # -------------------------
    final_chunks = rerank(question, candidates)

    # -------------------------
    # build context with citation
    # -------------------------
    context = "\n\n".join(
        f"[chunk_{i}]\n{c}" for i, c in enumerate(final_chunks)
    )

    prompt = f"""
You are a strict PDF QA system.

RULES:
- Use ONLY context
- If not found → say "Not found in document"
- MUST cite like [chunk_i]
- NO external knowledge

CONTEXT:
{context}

QUESTION:
{question}

Answer:
"""

    response = client.responses.create(
        model="gpt-5.5",
        input=prompt,
    )

    result = response.output_text

    _cache[key] = result
    return result