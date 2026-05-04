from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional


class Reranker:
    def __init__(
        self,
        cfg: Dict[str, Any],
        llm_call: Optional[Callable[[str, List[Dict[str, Any]]], str]] = None,
    ) -> None:
        self.enabled = bool(cfg.get("rerank_enabled", True))
        self.model_name = str(cfg.get("rerank_model", "BAAI/bge-reranker-v2-m3"))
        self.default_top_n = int(cfg.get("rerank_top_n", 4))
        self.default_min_score = float(cfg.get("rerank_min_score", 0.15))
        self._model = None
        self._model_ready = False
        self._llm_call = llm_call

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not query.strip() or not candidates:
            return candidates

        top_n = top_n or self.default_top_n
        min_score = self.default_min_score if min_score is None else float(min_score)

        scored = self._try_cross_encoder(query, candidates)
        if not scored:
            scored = self._try_llm_rerank(query, candidates)
        if not scored:
            scored = self._fallback_lexical_rerank(query, candidates)

        sorted_hits = sorted(
            scored, key=lambda x: x.get("rerank_score", 0.0), reverse=True
        )
        filtered = [
            h for h in sorted_hits if float(h.get("rerank_score", 0.0)) >= min_score
        ]
        if not filtered:
            # 如果全部低分，则回退到 embedding 原始顺序，保证流程不中断
            return candidates[:top_n]
        return filtered[:top_n]

    def _try_cross_encoder(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        try:
            self._ensure_model()
            if self._model is None:
                return []
            pairs = [(query, str(c.get("text", ""))) for c in candidates]
            scores = self._model.predict(pairs)
            out: List[Dict[str, Any]] = []
            for c, s in zip(candidates, scores):
                item = dict(c)
                item["rerank_score"] = float(s)
                out.append(item)
            return out
        except Exception:
            return []

    def _ensure_model(self) -> None:
        if self._model_ready:
            return
        self._model_ready = True
        try:
            from sentence_transformers import CrossEncoder

            # 不自动下载大模型，若本地未缓存则快速失败并回退到 LLM/词法重排。
            self._model = CrossEncoder(self.model_name, local_files_only=True)
        except Exception:
            self._model = None

    def _try_llm_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self._llm_call is None:
            return []
        brief_items = []
        for idx, c in enumerate(candidates, start=1):
            text = str(c.get("text", "")).strip()
            brief_items.append({"id": idx, "text": text[:900]})

        prompt = (
            "你是检索重排器。根据 query 与候选片段相关性打分。"
            '仅输出 JSON 对象: {"scores":[{"id":1,"score":0.82}]}.'
            "score 取值 0~1。"
            f"\nquery: {query}"
            f"\ncandidates: {json.dumps(brief_items, ensure_ascii=False)}"
        )
        try:
            content = self._llm_call(prompt, [{"role": "user", "content": prompt}])
            data = self._extract_json(content)
            scores = data.get("scores", []) if isinstance(data, dict) else []
            if not isinstance(scores, list):
                return []
            score_map: Dict[int, float] = {}
            for row in scores:
                if not isinstance(row, dict):
                    continue
                try:
                    rid = int(row.get("id"))
                    score = float(row.get("score", 0.0))
                except Exception:
                    continue
                score_map[rid] = score
            out: List[Dict[str, Any]] = []
            for idx, c in enumerate(candidates, start=1):
                item = dict(c)
                item["rerank_score"] = float(score_map.get(idx, 0.0))
                out.append(item)
            return out
        except Exception:
            return []

    def _fallback_lexical_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        tokens = set(self._tokenize(query))
        out: List[Dict[str, Any]] = []
        for c in candidates:
            text = str(c.get("text", ""))
            text_tokens = set(self._tokenize(text))
            overlap = len(tokens & text_tokens)
            denom = max(len(tokens), 1)
            score = overlap / denom
            item = dict(c)
            item["rerank_score"] = float(score)
            out.append(item)
        return out

    def _tokenize(self, text: str) -> List[str]:
        lowered = text.lower()
        # 混合中英文分词：中文按连续汉字段，英文按单词
        zh = re.findall(r"[\u4e00-\u9fff]{1,}", lowered)
        en = re.findall(r"[a-z0-9_]{2,}", lowered)
        return zh + en

    def _extract_json(self, content: str) -> Any:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.S)
        if match:
            content = match.group(1).strip()
        return json.loads(content)
