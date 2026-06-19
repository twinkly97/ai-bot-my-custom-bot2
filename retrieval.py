from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "data" / "index"


try:
    from langchain_core.embeddings import Embeddings as _LCEmbeddings
except Exception:
    class _LCEmbeddings:  # type: ignore[no-redef]
        pass


def tokenize(text: str) -> list[str]:
    # Split English/numeric runs from Korean runs so "NVDA의" -> ["nvda", "의"].
    return re.findall(r"[A-Za-z0-9_]+|[가-힣]+", text.lower())


class HashEmbeddings(_LCEmbeddings):
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        import hashlib
        vec = [0.0] * self.dim
        for token in tokenize(text):
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0 if (h >> 8) % 2 == 0 else -1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def load_chunks() -> list[dict[str, Any]]:
    path = INDEX_DIR / "chunks.jsonl"
    if not path.exists():
        return []
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def bm25_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    chunks = load_chunks()
    if not chunks:
        return []
    tokenized = [tokenize(c["text"]) for c in chunks]
    q_tokens = tokenize(query)
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(tokenized)
        scores = list(bm25.get_scores(q_tokens))
    except Exception:
        scores = [float(sum(1 for t in toks if t in set(q_tokens))) for toks in tokenized]
    # Always fall back to token overlap so single-chunk corpora and BM25 IDF=0 cases still surface results.
    overlap = [float(sum(1 for t in toks if t in set(q_tokens))) for toks in tokenized]
    combined = [s + 0.01 * o for s, o in zip(scores, overlap)]
    ranked = sorted(zip(chunks, combined), key=lambda x: x[1], reverse=True)[:k]
    return [{**c, "score": float(s), "retriever": "bm25"} for c, s in ranked if s > 0]


def semantic_search(query: str, k: int = 5, embedding_model: str = "text-embedding-3-small") -> list[dict[str, Any]]:
    meta_path = INDEX_DIR / "index_metadata.json"
    faiss_dir = INDEX_DIR / "faiss"
    if not meta_path.exists() or not faiss_dir.exists():
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    try:
        from langchain_community.vectorstores import FAISS
        if meta.get("embedding_provider") == "openai":
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(model=embedding_model, api_key=os.getenv("OPENAI_API_KEY"))
        else:
            embeddings = HashEmbeddings()
        db = FAISS.load_local(str(faiss_dir), embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search_with_score(query, k=k)
    except Exception as exc:
        if os.getenv("RAG_DEBUG"):
            print(f"[semantic_search] failed: {type(exc).__name__}: {exc}")
        return []
    return [
        {
            "id": i,
            "text": d.page_content,
            "metadata": d.metadata,
            "score": float(score),
            "retriever": "semantic",
        }
        for i, (d, score) in enumerate(docs)
    ]


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def merge_results(*groups: list[dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
    """Normalize each retriever's scores to [0,1] then sum so semantic and BM25 are comparable."""
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        if not group:
            continue
        norms = _minmax([float(item.get("score", 0)) for item in group])
        for item, norm in zip(group, norms):
            key = item.get("text", "")[:300] + str(item.get("metadata", {}))
            if key in merged:
                merged[key]["hybrid_score"] += norm
                merged[key]["retriever"] += "+" + item.get("retriever", "")
            else:
                merged[key] = {**item, "hybrid_score": norm}
    return sorted(merged.values(), key=lambda x: x.get("hybrid_score", 0), reverse=True)[:k]


def heuristic_rerank(query: str, docs: list[dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
    q_terms = set(tokenize(query))
    def score(doc):
        terms = tokenize(doc.get("text", ""))
        overlap = sum(1 for t in terms if t in q_terms)
        source_bonus = 0.1 if doc.get("metadata", {}).get("source") else 0.0
        return overlap + source_bonus + doc.get("hybrid_score", 0)
    return sorted(docs, key=score, reverse=True)[:k]


def llm_rerank(query: str, docs: list[dict[str, Any]], k: int = 5, model: str | None = None) -> list[dict[str, Any]]:
    """LLM-based rerank. Falls back to heuristic_rerank if API key is missing or call fails."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not docs:
        return heuristic_rerank(query, docs, k=k)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        listing = "\n\n".join(f"[{i}] {d.get('text','')[:600]}" for i, d in enumerate(docs))
        prompt = (
            "다음은 사용자 질문과 후보 문서 조각들이다. 질문과 가장 관련된 순서로 인덱스를 "
            f"쉼표로 구분된 정수 리스트로만 반환하라. 최대 {k}개.\n\n"
            f"질문: {query}\n\n후보:\n{listing}\n\n"
            "응답 예: 2,0,4,1"
        )
        resp = client.responses.create(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        )
        text = (getattr(resp, "output_text", "") or "").strip()
        order = [int(x) for x in re.findall(r"\d+", text) if int(x) < len(docs)]
        seen = set()
        ranked = []
        for idx in order:
            if idx in seen:
                continue
            seen.add(idx)
            ranked.append(docs[idx])
            if len(ranked) >= k:
                break
        # Backfill from heuristic order in case LLM dropped some
        if len(ranked) < k:
            for d in heuristic_rerank(query, docs, k=k):
                if d not in ranked:
                    ranked.append(d)
                    if len(ranked) >= k:
                        break
        return ranked[:k]
    except Exception as exc:
        if os.getenv("RAG_DEBUG"):
            print(f"[llm_rerank] fallback to heuristic: {type(exc).__name__}: {exc}")
        return heuristic_rerank(query, docs, k=k)


def retrieve(
    query: str,
    strategy: str = "hybrid_rerank",
    k: int = 5,
    embedding_model: str = "text-embedding-3-small",
    llm_rerank_enabled: bool | None = None,
) -> list[dict[str, Any]]:
    if strategy == "bm25":
        return bm25_search(query, k=k)
    if strategy == "semantic":
        semantic = semantic_search(query, k=k, embedding_model=embedding_model)
        return semantic or bm25_search(query, k=k)
    semantic = semantic_search(query, k=max(k * 2, k), embedding_model=embedding_model)
    keyword = bm25_search(query, k=max(k * 2, k))
    merged = merge_results(semantic, keyword, k=max(k * 2, k))
    if strategy == "hybrid_rerank":
        if llm_rerank_enabled is None:
            llm_rerank_enabled = os.getenv("OPENAI_API_KEY") is not None
        if llm_rerank_enabled:
            return llm_rerank(query, merged, k=k)
        return heuristic_rerank(query, merged, k=k)
    return merged[:k]


def format_sources(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "검색된 문서 근거 없음"
    lines = []
    for i, d in enumerate(docs, start=1):
        source = d.get("metadata", {}).get("source", "unknown")
        snippet = re.sub(r"\s+", " ", d.get("text", ""))[:350]
        lines.append(f"[{i}] {source}: {snippet}")
    return "\n".join(lines)
