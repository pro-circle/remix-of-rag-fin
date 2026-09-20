"""BM25 lexical retrieval (rank-bm25 when available, otherwise a small local BM25)."""
from __future__ import annotations

import math
import re
from collections import Counter

from ..chunking.semantic import Chunk
from .base import Candidate

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Bm25Retriever:
    def __init__(self) -> None:
        self._chunks: dict[str, tuple[str, dict, list[str]]] = {}
        self._index = None

    def add_documents(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = (chunk.text, chunk.metadata, tokenize(chunk.text))
        self._index = None

    def delete_document(self, document_id: str) -> None:
        for cid in [k for k, v in self._chunks.items() if v[1].get("document_id") == document_id]:
            self._chunks.pop(cid, None)
        self._index = None

    def _scores(self, ids: list[str], query_tokens: list[str]) -> list[float]:
        corpus = [self._chunks[i][2] for i in ids]
        try:
            from rank_bm25 import BM25Okapi

            return list(BM25Okapi(corpus).get_scores(query_tokens))
        except Exception:
            return self._fallback_scores(corpus, query_tokens)

    @staticmethod
    def _fallback_scores(corpus: list[list[str]], query_tokens: list[str], k1: float = 1.5, b: float = 0.75):
        n = len(corpus)
        avgdl = sum(len(d) for d in corpus) / n if n else 0.0
        df = Counter(t for doc in corpus for t in set(doc))
        scores = []
        for doc in corpus:
            freqs = Counter(doc)
            score = 0.0
            for term in query_tokens:
                if term not in freqs:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                tf = freqs[term]
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / (avgdl or 1)))
            scores.append(score)
        return scores

    def search(self, query: str, k: int, document_ids: list[str] | None = None) -> list[Candidate]:
        ids = [
            cid
            for cid, (_, meta, _) in self._chunks.items()
            if not document_ids or meta.get("document_id") in document_ids
        ]
        if not ids:
            return []
        scores = self._scores(ids, tokenize(query))
        ranked = sorted(zip(ids, scores), key=lambda r: r[1], reverse=True)[:k]
        top = ranked[0][1] or 1.0
        return [
            Candidate(
                chunk_id=cid,
                text=self._chunks[cid][0],
                metadata=dict(self._chunks[cid][1]),
                bm25_score=round(score / top, 4),
                bm25_rank=rank,
            )
            for rank, (cid, score) in enumerate(ranked, start=1)
            if score > 0
        ]
