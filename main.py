import os
import toml
import json
import re
import base64
import requests
import time
import threading
from pathlib import Path
from urllib.parse import quote
from types import SimpleNamespace
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Optional, Callable, cast
from openai import OpenAI, APIConnectionError, APIStatusError, InternalServerError
from docxtpl import DocxTemplate
import fitz
import numpy as np

from lib.reference_index import ReferenceIndexManager


class HomeworkAutomator:
    CHOICE_QUESTION_LIMIT = 50

    def __init__(self, config_path: str = "config.toml"):
        self.config = self._load_config(config_path)
        self._setup_clients()

    def _load_config(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"未找到配置文件: {path}. 请从模板config-template.toml创建"
            )
        return toml.load(path)

    def _setup_clients(self):
        # Setup Simple Model Client
        simple_cfg = self.config["llm"]["simple"]
        self.simple_client = OpenAI(
            api_key=simple_cfg["api_key"], base_url=simple_cfg["base_url"]
        )
        self.simple_model = simple_cfg["model"]

        # Setup Complex Model Client
        complex_cfg = self.config["llm"]["complex"]
        self.complex_client = OpenAI(
            api_key=complex_cfg["api_key"], base_url=complex_cfg["base_url"]
        )
        self.complex_model = complex_cfg["model"]

        # Setup Tools
        searxng_cfg = self.config.get("searxng", {})
        self.searxng_base = searxng_cfg.get("base_url", "http://localhost:8089/")
        self.searxng_timeout = int(searxng_cfg.get("timeout_seconds", 3600))
        self.retrieval_cfg = self.config.get("retrieval", {})
        self.flash_client = self.simple_client
        self.flash_model = str(self.retrieval_cfg.get("flash_model", self.simple_model))
        self.reference_index = ReferenceIndexManager(
            self.retrieval_cfg,
            image_to_text_fn=self._describe_image_for_embedding,
            low_quality_chunk_from_source_fn=self._rewrite_low_quality_chunk_from_source,
        )
        self._active_question_context: Optional[Dict[str, Any]] = None
        self._active_question_image_inputs: List[Tuple[str, str]] = []
        self.web_top_k = int(self.retrieval_cfg.get("web_top_k", 8))
        self.local_top_k = int(self.retrieval_cfg.get("local_top_k", 6))
        self.max_keyword_rounds = int(self.retrieval_cfg.get("max_keyword_rounds", 4))
        self.web_filter_min_relevance = float(
            self.retrieval_cfg.get("web_filter_min_relevance", 0.55)
        )
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "python_interpreter",
                    "description": "执行 Python 代码。用于数学计算、子网划分等逻辑运算，只有涉及验证计算才能使用，无法请求网络。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "要运行的 Python 代码。使用 print() 输出结果。",
                            }
                        },
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "统一检索工具。自动并行检索本地知识库与互联网，返回结构化证据。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "检索语句，必须包含题干中的关键实体和知识点",
                            },
                            "question_id": {
                                "type": "string",
                                "description": "可选，当前题号，便于日志与核查",
                            },
                            "intent": {
                                "type": "string",
                                "description": "可选，factual/calculation/protocol_compare 等检索意图",
                            },
                            "local_top_k": {
                                "type": "integer",
                                "description": "可选，本地知识库返回条数",
                            },
                            "web_top_k": {
                                "type": "integer",
                                "description": "可选，网络搜索采样条数",
                            },
                            "require_citation": {
                                "type": "boolean",
                                "description": "是否返回结构化出处。默认 true。",
                            },
                            "question_context": {
                                "type": "object",
                                "description": "可选，当前完整题目对象，例如包含 id、question、feedback 等字段；用于生成搜索关键词。",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def _describe_image_for_embedding(self, image_path: str) -> str:
        """调用 flash 模型将图片转成可检索文本。"""
        if not image_path or not os.path.exists(image_path):
            return ""

        prompt = (
            "你是检索预处理助手。请只基于图片内容输出客观文字描述，"
            "用于后续向量检索。不要输出推理过程，不要添加无关结论。"
            "优先包含：标题、章节名、关键词、协议名、公式、编号、表格要点。"
            "输出纯文本，100~300字。"
        )
        try:
            messages = self._build_image_message(
                prompt,
                [(f"参考页截图：{os.path.basename(image_path)}", image_path)],
            )
            res = self._call_ai(
                self.flash_client,
                self.flash_model,
                messages,
                use_tools=False,
                quiet=True,
            )
            if not res or not hasattr(res, "choices"):
                return ""
            content = (res.choices[0].message.content or "").strip()
            if not content:
                return ""
            return self._clean_markdown(content)
        except Exception as e:
            print(f">>> [Embedding/图片转文字] flash 模型失败: {e}")
            return ""

    def _rewrite_low_quality_chunk_from_source(
        self, source_path: str, chunk_meta: Dict[str, Any]
    ) -> str:
        """低质量块重写：基于源文件内容/页面调用 flash 生成可检索文本。"""
        src = Path(source_path)
        if not src.exists():
            return ""

        raw_chunk = str(chunk_meta.get("raw_chunk", "") or "").strip()
        chunk_index = int(chunk_meta.get("chunk_index", 1) or 1)
        total_chunks = max(1, int(chunk_meta.get("total_chunks", 1) or 1))
        suffix = str(chunk_meta.get("suffix", src.suffix.lower()) or "").lower()

        prompt = (
            "你是知识库文本重建助手。现在需要为一个低质量文本块生成可检索文字。"
            "必须基于给定源文件内容（或页面截图）来重写该块，不要根据乱码猜测。\n"
            "要求：\n"
            "1) 输出客观、简洁、可检索的中文文本，保留术语/公式名/协议名。\n"
            "2) 不要输出解释、免责声明或推理过程。\n"
            "3) 若无法确定内容，返回空字符串。\n"
            '仅输出 JSON: {"chunk_text":"..."}\n\n'
            f"块序号: {chunk_index}/{total_chunks}\n"
            f"源文件: {src.as_posix()}\n"
            f"低质量块原文(仅供定位):\n{raw_chunk[:800]}"
        )

        try:
            messages: List[Dict[str, Any]]
            if suffix == ".pdf":
                image_inputs = self._build_repair_pdf_page_inputs(
                    pdf_path=src,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                )
                if image_inputs:
                    messages = self._build_image_message(prompt, image_inputs)
                else:
                    context_text = self._extract_source_window_text(
                        src, chunk_index=chunk_index, total_chunks=total_chunks
                    )
                    messages = [
                        {
                            "role": "user",
                            "content": f"{prompt}\n\n源文件文本片段:\n{context_text}",
                        }
                    ]
            else:
                context_text = self._extract_source_window_text(
                    src, chunk_index=chunk_index, total_chunks=total_chunks
                )
                if not context_text.strip():
                    return ""
                messages = [
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n源文件文本片段:\n{context_text}",
                    }
                ]

            res = self._call_ai(
                self.flash_client,
                self.flash_model,
                messages,
                use_tools=False,
                response_format={"type": "json_object"},
                quiet=True,
            )
            if not res or not hasattr(res, "choices"):
                return ""
            content = (res.choices[0].message.content or "").strip()
            if not content:
                return ""
            payload = self._parse_json_safe(content)
            rewritten = ""
            if isinstance(payload, dict):
                rewritten = str(payload.get("chunk_text", "") or "").strip()
            rewritten = self._clean_markdown(rewritten)
            rewritten = re.sub(r"\n{3,}", "\n\n", rewritten).strip()
            return rewritten
        except Exception as e:
            print(f">>> [Embedding/低质量块重写] flash 调用失败: {e}")
            return ""

    def _build_repair_pdf_page_inputs(
        self, pdf_path: Path, chunk_index: int, total_chunks: int
    ) -> List[Tuple[str, str]]:
        """按块序号近似映射到 PDF 页并输出邻近页截图输入。"""
        inputs: List[Tuple[str, str]] = []
        try:
            pdf = fitz.open(str(pdf_path))
        except Exception:
            return inputs

        try:
            page_count = len(pdf)
            if page_count <= 0:
                return inputs

            ratio = (max(1, chunk_index) - 0.5) / max(1, total_chunks)
            target_page = int(ratio * page_count) + 1
            target_page = max(1, min(page_count, target_page))

            page_candidates = []
            for p in (target_page - 1, target_page, target_page + 1):
                if 1 <= p <= page_count and p not in page_candidates:
                    page_candidates.append(p)

            out_dir = Path("problems") / "_reference_mm"
            out_dir.mkdir(parents=True, exist_ok=True)

            for page_no in page_candidates:
                try:
                    page = pdf[page_no - 1]
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
                    image_path = out_dir / f"_repair_{pdf_path.stem}_p{page_no}.png"
                    pix.save(str(image_path))
                    inputs.append((f"源文件第{page_no}页", image_path.as_posix()))
                except Exception:
                    continue
        finally:
            pdf.close()

        return inputs

    def _extract_source_window_text(
        self, source_path: Path, chunk_index: int, total_chunks: int
    ) -> str:
        """对文本类源文件提取与块位置对应的局部窗口。"""
        try:
            text = source_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

        clean = text.strip()
        if not clean:
            return ""

        total_len = len(clean)
        center = int(((max(1, chunk_index) - 0.5) / max(1, total_chunks)) * total_len)
        win = min(6000, max(1600, total_len // max(1, total_chunks)))
        start = max(0, center - win // 2)
        end = min(total_len, start + win)
        return clean[start:end]

    def _execute_python(self, code: str) -> str:
        """执行 Python 代码并返回输出和返回值，限制文件和网络访问"""
        import sys
        from io import StringIO
        import math, re, json, base64, datetime, itertools

        # 安全预检查
        forbidden_keywords = [
            "open(",
            "write(",
            "read(",
            "socket",
            "requests",
            "urllib",
            "os.",
            "shutil",
            "pathlib",
            "subprocess",
            "sys.",
            "eval(",
            "exec(",
        ]
        for kw in forbidden_keywords:
            if kw in code:
                return f"Error: 权限受限，禁止使用 '{kw}'。"

        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output
        try:
            # 提供基础环境
            safe_globals = {
                "math": math,
                "re": re,
                "json": json,
                "base64": base64,
                "datetime": datetime,
                "itertools": itertools,
                "print": print,
                "range": range,
                "len": len,
                "int": int,
                "str": str,
                "float": float,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "pow": pow,
                "round": round,
                "enumerate": enumerate,
                "zip": zip,
                "sorted": sorted,
                "reversed": reversed,
            }
            loc = {}
            exec(code, safe_globals, loc)
            stdout_val = redirected_output.getvalue()
            result_summary = f"Stdout:\n{stdout_val}\n"
            if loc:
                # 过滤掉全局变量，只保留执行中产生的局部变量
                vars_summary = {
                    k: str(v)
                    for k, v in loc.items()
                    if k not in safe_globals and not k.startswith("__")
                }
                if vars_summary:
                    result_summary += (
                        f"Variables:\n{json.dumps(vars_summary, ensure_ascii=False)}"
                    )
            return result_summary
        except Exception as e:
            return f"Stdout:\n{redirected_output.getvalue()}\nError: {e}"
        finally:
            sys.stdout = old_stdout

    def _search_searxng_raw(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """通过 SearxNG 拉取原始搜索结果。"""
        if not query.strip():
            return []
        try:
            q = quote(query)
            url = f"{self.searxng_base}search?q={q}&format=json"
            response = requests.get(url, timeout=self.searxng_timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])[: (top_k or self.web_top_k)]
        except Exception:
            return []

    def _fetch_web_original_text(self, url: str) -> str:
        """抓取网页响应原文，不清洗、不截断、不改写。"""
        if not url.strip():
            return ""
        try:
            resp = requests.get(
                url,
                timeout=min(self.searxng_timeout, 30),
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0 Safari/537.36"
                    )
                },
            )
            resp.raise_for_status()
            return resp.text
        except Exception:
            return ""

    def _build_keyword_candidates(
        self,
        query: str,
        intent: str = "",
        question_context: Optional[Dict[str, Any]] = None,
        question_image_inputs: Optional[List[Tuple[str, str]]] = None,
    ) -> List[str]:
        """生成多轮关键词候选，支持小模型改写 + 规则兜底。"""
        candidates: List[str] = []
        normalized_query = query.strip()
        if not normalized_query:
            return candidates

        rewrite_prompt = (
            "你是检索改写助手。请把输入问题改写成适合中文网络检索的关键词短语。"
            '返回 JSON，格式为 {"queries":["..."]}。'
            "要求：保留协议名、关键术语、场景词；不要输出解释。"
            f"\nintent: {intent or 'general'}"
            f"\nquestion: {normalized_query}"
            f"\n完整题目上下文:\n{json.dumps(question_context or {}, ensure_ascii=False)}"
        )

        try:
            rewrite_messages = self._build_image_message(
                rewrite_prompt,
                question_image_inputs or [],
            )
            res = self._call_ai(
                self.simple_client,
                self.simple_model,
                rewrite_messages,
                use_tools=False,
                response_format={"type": "json_object"},
            )
            content = (
                res.choices[0].message.content
                if res and hasattr(res, "choices")
                else ""
            )
            if content:
                data = self._parse_json_safe(content)
                llm_queries = data.get("queries", []) if isinstance(data, dict) else []
                if isinstance(llm_queries, list):
                    for item in llm_queries:
                        text = str(item).strip()
                        if text:
                            candidates.append(text)
        except Exception:
            pass

        fallback = [
            normalized_query,
            f"计算机网络 {normalized_query}",
            f"{normalized_query} 协议 原理",
            f"{normalized_query} 题目 解析",
        ]
        for item in fallback:
            if item not in candidates:
                candidates.append(item)
        return candidates

    def _filter_web_results(
        self,
        query: str,
        raw_results: List[Dict[str, Any]],
        min_relevance: Optional[float] = None,
        question_context: Optional[Dict[str, Any]] = None,
        question_image_inputs: Optional[List[Tuple[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """用简单模型过滤网页结果，只保留高相关事实。"""
        if not raw_results:
            return []

        payload = []
        for r in raw_results:
            payload.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                }
            )

        filter_prompt = f"""你是检索质量过滤器。请从候选网页中提取与问题直接相关的事实。
问题：{query}

完整题目上下文：
{json.dumps(question_context or {}, ensure_ascii=False)}

候选结果：
{json.dumps(payload, ensure_ascii=False)}

只输出 JSON 对象：
{{
  "items": [
    {{"title":"...","url":"...","fact":"...","relevance":0.0}}
  ]
}}

规则：
1. relevance 取值 0~1。
2. 只保留能直接支撑答案的事实；不相关内容删除。
3. fact 必须是简洁陈述句，不要写建议或分析。
"""

        try:
            filter_messages = self._build_image_message(
                filter_prompt,
                question_image_inputs or [],
            )
            res = self._call_ai(
                self.simple_client,
                self.simple_model,
                filter_messages,
                use_tools=False,
                response_format={"type": "json_object"},
            )
            content = (
                res.choices[0].message.content
                if res and hasattr(res, "choices")
                else ""
            )
            if not content:
                return []

            data = self._parse_json_safe(content)
            items = data.get("items", []) if isinstance(data, dict) else []
            if not isinstance(items, list):
                return []

            threshold = (
                min_relevance
                if min_relevance is not None
                else self.web_filter_min_relevance
            )
            cleaned: List[Dict[str, Any]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                relevance = item.get("relevance", 0)
                try:
                    score = float(relevance)
                except Exception:
                    score = 0.0
                if score < threshold:
                    continue
                fact = str(item.get("fact", "")).strip()
                if not fact:
                    continue
                source = next(
                    (
                        r
                        for r in raw_results
                        if str(r.get("url", "")).strip()
                        == str(item.get("url", "")).strip()
                    ),
                    {},
                )
                snippet = str(source.get("content", "") or "")
                original_text = self._fetch_web_original_text(
                    str(item.get("url", "")).strip()
                )
                cleaned.append(
                    {
                        "title": str(item.get("title", "")).strip(),
                        "url": str(item.get("url", "")).strip(),
                        "content": original_text or snippet,
                        "search_snippet": snippet,
                        "relevance": score,
                    }
                )
            return cleaned
        except Exception:
            return []

    def _search_knowledge(
        self,
        query: str,
        question_id: str = "",
        intent: str = "",
        local_top_k: Optional[int] = None,
        web_top_k: Optional[int] = None,
        require_citation: bool = True,
        question_context: Optional[Dict[str, Any]] = None,
        question_image_inputs: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """统一检索：本地知识库 + 联网检索 + 结构化来源。"""
        keyword_candidates = self._build_keyword_candidates(
            query,
            intent=intent,
            question_context=question_context,
            question_image_inputs=question_image_inputs,
        )
        local_hits: List[Dict[str, Any]] = []
        seen_local = set()
        for local_query in [query] + keyword_candidates:
            for hit in self.reference_index.query(
                local_query, top_k=local_top_k or self.local_top_k
            ):
                meta = hit.get("metadata", {}) if isinstance(hit, dict) else {}
                key = (
                    meta.get("source_path", ""),
                    meta.get("page"),
                    meta.get("chunk_id"),
                    meta.get("image_path"),
                    str(hit.get("text", "")),
                )
                if key in seen_local:
                    continue
                seen_local.add(key)
                local_hits.append(hit)

        best_web_records: List[Dict[str, Any]] = []
        used_keywords: List[str] = []
        for keyword in keyword_candidates:
            used_keywords.append(keyword)
            raw_results = self._search_searxng_raw(
                keyword, top_k=web_top_k or self.web_top_k
            )
            filtered = self._filter_web_results(
                query,
                raw_results,
                question_context=question_context,
                question_image_inputs=question_image_inputs,
            )
            if filtered:
                best_web_records = filtered
                break

        web_file = None
        if best_web_records:
            web_file = self.reference_index.ingest_web_records(query, best_web_records)

        citations: List[Dict[str, Any]] = []
        evidence: List[Dict[str, Any]] = []
        local_facts: List[str] = []
        for idx, hit in enumerate(local_hits, start=1):
            meta = hit.get("metadata", {}) if isinstance(hit, dict) else {}
            content_type = str(meta.get("content_type", "") or "")
            image_path = str(meta.get("image_path", "") or "")
            fact = str(hit.get("text", ""))
            evidence_content = (
                "" if content_type == "page_image" and image_path else fact
            )
            local_facts.append(evidence_content)
            evidence.append(
                {
                    "id": f"L{idx}",
                    "type": "local",
                    "source_path": meta.get("source_path", ""),
                    "page": meta.get("page"),
                    "image_path": image_path,
                    "content_type": content_type,
                    "score": hit.get("score"),
                    "content": evidence_content,
                }
            )
            if require_citation:
                citations.append(
                    {
                        "id": f"L{idx}",
                        "source_path": meta.get("source_path", ""),
                        "page": meta.get("page"),
                        "image_path": meta.get("image_path"),
                        "content_type": meta.get("content_type", "text_chunk"),
                        "retrieval_score": hit.get("score"),
                    }
                )

        web_facts: List[str] = []
        for idx, item in enumerate(best_web_records, start=1):
            snippet = str(item.get("content", "") or "")
            web_facts.append(snippet)
            evidence.append(
                {
                    "id": f"W{idx}",
                    "type": "web",
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "relevance": item.get("relevance", 0),
                    "content": snippet,
                    "search_snippet": item.get("search_snippet", ""),
                }
            )
            if require_citation:
                citations.append(
                    {
                        "id": f"W{idx}",
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "relevance": item.get("relevance", 0),
                    }
                )

        return {
            "question_id": question_id,
            "query": query,
            "keywords": used_keywords,
            "local_facts": local_facts,
            "web_facts": web_facts,
            "evidence": evidence,
            "citations": citations,
            "web_ingest_file": web_file,
            "has_evidence": bool(evidence),
        }

    def _search_knowledge_context(
        self,
        query: str,
        question_id: str = "",
        question_context: Optional[Dict[str, Any]] = None,
        question_image_inputs: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[str, List[Tuple[str, str]], bool]:
        """把检索结果整理成答题上下文，并收集可附加的图片文件。"""
        result = self._search_knowledge(
            query=query,
            question_id=question_id,
            question_context=question_context,
            question_image_inputs=question_image_inputs,
        )
        local_lines = []
        web_lines = []
        image_inputs: List[Tuple[str, str]] = []
        seen_images = set()
        for item in result.get("evidence", []):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "") or "")
            if item.get("type") == "local":
                source = str(item.get("source_path", "") or "本地知识库")
                page = item.get("page")
                content_type = str(item.get("content_type", "") or "").strip()
                page_text = f" 第{page}页" if page else ""
                type_text = f" 类型：{content_type}" if content_type else ""
                image_path = str(item.get("image_path", "") or "").strip()
                if image_path and os.path.exists(image_path) and image_path not in seen_images:
                    seen_images.add(image_path)
                    image_label = f"检索证据 {item.get('id')}：{source}{page_text}"
                    image_inputs.append((image_label, image_path))
                if content_type == "page_image" and image_path:
                    local_lines.append(
                        f"- [{item.get('id')}] 来源：{source}{page_text}{type_text}\n  图片证据：已作为图片附件提供。"
                    )
                else:
                    local_lines.append(
                        f"- [{item.get('id')}] 来源：{source}{page_text}{type_text}\n  原始切片内容：\n{content}"
                    )
            elif item.get("type") == "web":
                title = str(item.get("title", "") or "网页结果").strip()
                url = str(item.get("url", "") or "").strip()
                web_lines.append(
                    f"- [{item.get('id')}] 来源：{title} {url}\n  原始网页内容：\n{content}"
                )

        local_part = "\n".join(local_lines)
        web_part = "\n".join(web_lines)
        if not local_part:
            local_part = "- 无本地命中"
        if not web_part:
            web_part = "- 无联网命中"

        return (
            f"[检索问题] {query}\n"
            f"[本地知识库]\n{local_part}\n"
            f"[联网原文]\n{web_part}"
        ), image_inputs, bool(result.get("has_evidence"))

    def _search_knowledge_text(self, query: str, question_id: str = "") -> str:
        """把结构化检索结果压缩成提示词可用文本。"""
        text, _, _ = self._search_knowledge_context(
            query=query, question_id=question_id
        )
        return text

    def _handle_tool_calls(
        self, tool_calls: Any
    ) -> Tuple[List[Dict[str, str]], List[Tuple[str, str]]]:
        """执行工具调用并返回结果列表与可追加给模型的图片证据。"""
        results = []
        image_inputs: List[Tuple[str, str]] = []
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except:
                args = {}
            print(f"  [执行工具] {func_name}: {args}")

            if func_name == "python_interpreter":
                result = self._execute_python(args.get("code", ""))
            elif func_name in {"search", "search_web"}:
                question_context = (
                    args.get("question_context")
                    if isinstance(args.get("question_context"), dict)
                    else self._active_question_context
                )
                search_payload = self._search_knowledge(
                    query=args.get("query", ""),
                    question_id=str(args.get("question_id", "")),
                    intent=str(args.get("intent", "")),
                    local_top_k=args.get("local_top_k"),
                    web_top_k=args.get("web_top_k"),
                    require_citation=bool(args.get("require_citation", True)),
                    question_context=question_context,
                    question_image_inputs=self._active_question_image_inputs,
                )
                for item in search_payload.get("evidence", []):
                    if not isinstance(item, dict):
                        continue
                    image_path = str(item.get("image_path", "") or "").strip()
                    if image_path and os.path.exists(image_path):
                        source = str(item.get("source_path", "") or "检索证据")
                        page = item.get("page")
                        page_text = f" 第{page}页" if page else ""
                        image_inputs.append(
                            (f"工具检索证据 {item.get('id')}：{source}{page_text}", image_path)
                        )
                result = json.dumps(search_payload, ensure_ascii=False)
            else:
                result = "未知工具"

            results.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": func_name,
                    "content": result,
                }
            )
            print(f"  [工具结果] 已返回完整内容，长度 {len(result)} 字符")
        return results, image_inputs

    def _call_ai(
        self,
        client: OpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        use_tools: bool = True,
        **kwargs: Any,
    ) -> Any:
        """封装 AI 调用，支持工具自动处理及网络错误重试"""
        quiet = bool(kwargs.pop("quiet", False))
        current_messages = messages.copy()
        response: Any = None

        for _ in range(10):  # 最多 10 轮工具交互
            call_params = {
                "model": model,
                "messages": current_messages,
                "timeout": None,
            }
            if use_tools:
                call_params["tools"] = self.tools

            call_params.update(kwargs)

            # 网络重试逻辑
            max_retries = 1000
            max_api_status_retries = 3
            last_err = None
            for retry in range(max_retries + 1):
                wait_thread = None
                stop_wait_event = None
                first_chunk_event = None
                try:
                    stream_params = dict(call_params)
                    stream_params["stream"] = True
                    stream_params["stream_options"] = {"include_usage": True}
                    request_start = time.perf_counter()
                    first_chunk_event = threading.Event()
                    stop_wait_event = threading.Event()

                    def _wait_first_chunk_indicator() -> None:
                        while not stop_wait_event.is_set():
                            wait_elapsed = time.perf_counter() - request_start
                            print(
                                f"\r  [流式] 等待首包 {wait_elapsed:.2f}s",
                                end="",
                                flush=True,
                            )
                            if first_chunk_event.wait(0.2):
                                break

                    if not quiet:
                        wait_thread = threading.Thread(
                            target=_wait_first_chunk_indicator,
                            daemon=True,
                        )
                        wait_thread.start()

                    stream = client.chat.completions.create(**stream_params)

                    content_parts: List[str] = []
                    tool_calls_data: Dict[int, Dict[str, Any]] = {}
                    usage = None
                    first_chunk_at: Optional[float] = None
                    stream_chars = 0

                    for chunk in stream:
                        if getattr(chunk, "usage", None) is not None:
                            usage = chunk.usage

                        chunk_choices = getattr(chunk, "choices", None) or []
                        if not chunk_choices:
                            continue

                        choice = chunk_choices[0]
                        delta = getattr(choice, "delta", None)
                        if delta is None:
                            continue

                        delta_content = getattr(delta, "content", None)
                        delta_tool_calls = getattr(delta, "tool_calls", None)
                        has_payload = bool(delta_content) or bool(delta_tool_calls)
                        if has_payload and first_chunk_at is None:
                            first_chunk_at = time.perf_counter()
                            first_chunk_event.set()

                        if delta_content:
                            content_parts.append(delta_content)
                            stream_chars += len(delta_content)

                        if delta_tool_calls:
                            for tc in delta_tool_calls:
                                idx = int(getattr(tc, "index", 0) or 0)
                                entry = tool_calls_data.setdefault(
                                    idx,
                                    {
                                        "id": "",
                                        "function": {
                                            "name": "",
                                            "arguments": "",
                                        },
                                    },
                                )
                                tc_id = getattr(tc, "id", None)
                                if tc_id:
                                    entry["id"] = tc_id

                                tc_func = getattr(tc, "function", None)
                                if tc_func is not None:
                                    tc_name = getattr(tc_func, "name", None)
                                    if tc_name:
                                        entry["function"]["name"] += tc_name

                                    tc_args = getattr(tc_func, "arguments", None)
                                    if tc_args:
                                        entry["function"]["arguments"] += tc_args

                        if first_chunk_at is not None and not quiet:
                            elapsed_stream = time.perf_counter() - first_chunk_at
                            print(
                                f"\r  [流式] 已接收 {stream_chars} 字符，流式耗时 {elapsed_stream:.2f}s",
                                end="",
                                flush=True,
                            )

                    finished_at = time.perf_counter()
                    wait_first = (
                        (first_chunk_at - request_start)
                        if first_chunk_at is not None
                        else (finished_at - request_start)
                    )
                    stream_elapsed = (
                        (finished_at - first_chunk_at)
                        if first_chunk_at is not None
                        else 0
                    )

                    completion_tokens = (
                        getattr(usage, "completion_tokens", None)
                        if usage is not None
                        else None
                    )
                    prompt_tokens = (
                        getattr(usage, "prompt_tokens", None)
                        if usage is not None
                        else None
                    )

                    stop_wait_event.set()
                    first_chunk_event.set()
                    if wait_thread is not None:
                        wait_thread.join(timeout=0.5)

                    if not quiet:
                        print(
                            f"\r  [流式统计] 首包等待 {wait_first:.2f}s | 流式接收 {stream_elapsed:.2f}s | 输出长度 {stream_chars} 字符 | prompt_tokens={prompt_tokens} | completion_tokens={completion_tokens}"
                        )

                    tool_calls = None
                    if tool_calls_data:
                        tool_calls = []
                        for idx in sorted(tool_calls_data.keys()):
                            tc = tool_calls_data[idx]
                            tool_calls.append(
                                SimpleNamespace(
                                    id=tc["id"],
                                    function=SimpleNamespace(
                                        name=tc["function"]["name"],
                                        arguments=tc["function"]["arguments"],
                                    ),
                                )
                            )

                    message = SimpleNamespace(
                        content="".join(content_parts) if content_parts else None,
                        tool_calls=tool_calls,
                    )
                    response = SimpleNamespace(
                        choices=[SimpleNamespace(message=message)],
                        usage=usage,
                    )
                    break
                except (
                    APIConnectionError,
                    InternalServerError,
                    requests.exceptions.RequestException,
                ) as e:
                    if stop_wait_event is not None:
                        stop_wait_event.set()
                    if first_chunk_event is not None:
                        first_chunk_event.set()
                    if wait_thread is not None and wait_thread.is_alive():
                        wait_thread.join(timeout=0.5)
                    last_err = e
                    if retry < max_retries:
                        wait_time = (retry + 1) * 2
                        print(
                            f"  [网络错误] {e}，正在进行第 {retry + 1} 次重试 ({wait_time}s)..."
                        )
                        time.sleep(wait_time)
                    else:
                        print(f"  [严重错误] 已达最大重试次数，调用失败。")
                        raise last_err
                except APIStatusError as e:
                    if stop_wait_event is not None:
                        stop_wait_event.set()
                    if first_chunk_event is not None:
                        first_chunk_event.set()
                    if wait_thread is not None and wait_thread.is_alive():
                        wait_thread.join(timeout=0.5)
                    last_err = e
                    if retry < max_api_status_retries:
                        wait_time = (retry + 1) * 2
                        print(
                            f"  [AI请求错误] {e}，正在进行第 {retry + 1}/{max_api_status_retries} 次重试 ({wait_time}s)..."
                        )
                        time.sleep(wait_time)
                    else:
                        print(
                            f"  [严重错误] AI请求重试 {max_api_status_retries} 次后仍失败。"
                        )
                        raise last_err

            if not response:
                return None
            msg = response.choices[0].message

            if msg.tool_calls:
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            for tool_call in msg.tool_calls
                        ],
                    }
                )
                tool_results, tool_image_inputs = self._handle_tool_calls(msg.tool_calls)
                current_messages.extend(tool_results)
                if tool_image_inputs:
                    image_message = self._build_image_message(
                        "以下是本轮工具检索命中的图片证据，请结合上一条工具结果中的来源使用；图片证据以附件本身为准。",
                        tool_image_inputs,
                    )[0]
                    current_messages.append(image_message)
            else:
                return response
        return response

    def _build_image_message(
        self, prompt: str, image_inputs: List[Any]
    ) -> List[Dict[str, Any]]:
        """构造带可选图片的 user 消息，支持字符串路径或(标签, 路径)元组"""
        normalized_inputs: List[Tuple[str, str]] = []
        for item in image_inputs:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                label = str(item[0]).strip() or "截图"
                path = str(item[1]).strip()
            else:
                path = str(item).strip()
                label = f"题目截图：{os.path.basename(path)}"

            if path and os.path.exists(path):
                normalized_inputs.append((label, path))

        if not normalized_inputs:
            return [{"role": "user", "content": prompt}]

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for label, p in normalized_inputs:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        return [{"role": "user", "content": content}]

    def _collect_question_image_inputs(
        self,
        questions: List[Dict[str, Any]],
        question_image_map: Dict[str, str],
    ) -> List[Tuple[str, str]]:
        """按题号收集题目截图，供多模态提示词使用"""
        images: List[Tuple[str, str]] = []
        for q in questions:
            qid = str(q.get("id", "")).strip()
            if not qid:
                continue
            q_img = question_image_map.get(qid, "")
            if q_img:
                images.append((f"第 {qid} 题题目截图", q_img))
        return images

    def _call_ai_for_question(
        self,
        question_context: Dict[str, Any],
        question_image_inputs: List[Tuple[str, str]],
        client: OpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        use_tools: bool = True,
        **kwargs: Any,
    ) -> Any:
        old_context = self._active_question_context
        old_images = self._active_question_image_inputs
        self._active_question_context = question_context
        self._active_question_image_inputs = question_image_inputs
        try:
            return self._call_ai(
                client,
                model,
                messages,
                use_tools=use_tools,
                **kwargs,
            )
        finally:
            self._active_question_context = old_context
            self._active_question_image_inputs = old_images

    def _extract_page_lines(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """提取页面文本行及其坐标，用于题目截图定位"""
        page_dict = page.get_text("dict")
        lines: List[Dict[str, Any]] = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                x0 = min(s.get("bbox", [0, 0, 0, 0])[0] for s in spans)
                y0 = min(s.get("bbox", [0, 0, 0, 0])[1] for s in spans)
                x1 = max(s.get("bbox", [0, 0, 0, 0])[2] for s in spans)
                y1 = max(s.get("bbox", [0, 0, 0, 0])[3] for s in spans)
                lines.append({"text": text, "bbox": (x0, y0, x1, y1)})
        lines.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        return lines

    def _merge_pixmaps(self, pixmaps: List[fitz.Pixmap]) -> fitz.Pixmap:
        """垂直合并多个 Pixmap，使用 numpy 处理"""
        if not pixmaps:
            return None
        if len(pixmaps) == 1:
            return pixmaps[0]

        arrays = []
        for p in pixmaps:
            # 将 pixmap 转换为 numpy 数组 (H, W, C)
            img = np.frombuffer(p.samples, dtype=np.uint8).reshape(
                p.height, p.width, p.n
            )
            arrays.append(img)

        max_w = max(a.shape[1] for a in arrays)
        padded_arrays = []
        for a in arrays:
            if a.shape[1] < max_w:
                # 如果宽度不一致，右侧填充白色
                pad = (
                    np.ones(
                        (a.shape[0], max_w - a.shape[1], a.shape[2]), dtype=np.uint8
                    )
                    * 255
                )
                a = np.hstack([a, pad])
            padded_arrays.append(a)

        merged_array = np.vstack(padded_arrays)
        # 从 numpy 数组还原回 fitz.Pixmap
        return fitz.Pixmap(
            pixmaps[0].colorspace,
            merged_array.shape[1],
            merged_array.shape[0],
            merged_array.tobytes(),
            pixmaps[0].alpha,
        )

    def generate_problem_screenshots(
        self, pdf_path: str, parts: Dict[str, Any]
    ) -> Dict[str, Dict[str, str]]:
        """按题号从 PDF 裁剪题目截图，保存到 problems 目录 (支持跨页)"""
        out_dir = "problems"
        os.makedirs(out_dir, exist_ok=True)

        choice_ids = [str(q.get("id", "")).strip() for q in parts.get("choice", [])]
        short_ids = [
            str(q.get("id", "")).strip() for q in parts.get("short_answer", [])
        ]
        prog_ids = [str(q.get("id", "")).strip() for q in parts.get("programming", [])]

        all_ids = [qid for qid in (choice_ids + short_ids + prog_ids) if qid]
        id_set = set(all_ids)
        if not id_set:
            return {"choice": {}, "short_answer": {}, "programming": {}}

        doc = fitz.open(pdf_path)
        starts: List[Dict[str, Any]] = []
        seen = set()

        # 根据“数字+分隔符”定位题目起始行
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            for line in self._extract_page_lines(page):
                m = re.match(r"^\s*(\d{1,3})\s*[\.、．\)]\s*", line["text"])
                if not m:
                    continue
                qid = m.group(1)
                if qid in id_set and qid not in seen:
                    starts.append(
                        {
                            "id": qid,
                            "page": page_idx,
                            "y": line["bbox"][1],
                        }
                    )
                    seen.add(qid)

        starts.sort(key=lambda item: (item["page"], item["y"]))

        def _get_clip_pixmap(page: fitz.Page, y0: float, y1: float) -> fitz.Pixmap:
            """截取指定高度范围的页面 Pixmap，并尝试去除页眉页脚空白"""
            lines = self._extract_page_lines(page)
            # 过滤属于该区域的行，跳过明显是页眉（顶部 50 单位）或页脚（底部 50 单位）的内容（如果它们跨页了）
            relevant_lines = []
            for ln in lines:
                ly0, ly1 = ln["bbox"][1], ln["bbox"][3]
                # 如果是中间页，忽略顶部和底部的页眉页脚（大致估算 55 单位）
                is_header = ly1 < 60
                is_footer = ly0 > page.rect.height - 60

                if ly0 >= y0 - 4 and ly1 <= y1 + 4:
                    # 只有当这不是唯一的行时，才跳过页眉页脚（防止题目本身就在页眉位置，虽然罕见）
                    if not (is_header or is_footer):
                        relevant_lines.append(ln)

            if relevant_lines:
                # 进一步缩紧边界
                real_y0 = max(y0, min(ln["bbox"][1] for ln in relevant_lines) - 8)
                real_y1 = min(y1, max(ln["bbox"][3] for ln in relevant_lines) + 8)
            else:
                real_y0, real_y1 = y0, y1

            if real_y1 <= real_y0:
                # 如果没有有效内容，返回一个极小的空白区域以防崩溃，或返回 None
                return None

            clip = fitz.Rect(0, real_y0, page.rect.width, real_y1)
            return page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)

        pdf_prefix = os.path.splitext(os.path.basename(pdf_path))[0]
        id_to_path: Dict[str, str] = {}
        for idx, s in enumerate(starts):
            start_page = s["page"]
            start_y = s["y"]

            if idx + 1 < len(starts):
                end_page = starts[idx + 1]["page"]
                end_y = starts[idx + 1]["y"]
            else:
                # 最后一题，到 PDF 末尾
                end_page = len(doc) - 1
                end_y = doc[end_page].rect.height

            # 收集所有跨页片段
            segments = []
            for p_idx in range(start_page, end_page + 1):
                page = doc[p_idx]
                # 如果是第一页，从题目开始算；否则从页顶算
                y0 = start_y if p_idx == start_page else 0
                # 如果是最后一页，到下一题开始算；否则到页底算
                y1 = end_y if p_idx == end_page else page.rect.height

                if y1 > y0 + 2:  # 忽略过小的片段
                    pix = _get_clip_pixmap(page, y0, y1)
                    if pix:
                        segments.append(pix)

            if segments:
                out_name = f"{pdf_prefix}_{s['id']}.png"
                out_path = os.path.join(out_dir, out_name)
                final_pix = self._merge_pixmaps(segments)
                final_pix.save(out_path)
                id_to_path[s["id"]] = out_path

        doc.close()

        return {
            "choice": {qid: id_to_path.get(qid, "") for qid in choice_ids},
            "short_answer": {qid: id_to_path.get(qid, "") for qid in short_ids},
            "programming": {qid: id_to_path.get(qid, "") for qid in prog_ids},
        }

    def _extract_question_ids_from_pdf(self, pdf_path: str) -> List[str]:
        """直接从 PDF 文本中提取所有题号，无需 AI 模型"""
        question_ids = []
        try:
            doc = fitz.open(pdf_path)
            seen_ids = set()

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                for line in self._extract_page_lines(page):
                    m = re.match(
                        r"^\s*(\d{1,3}(?:_sub_\d+)?)\s*[\.、．\)]\s*", line["text"]
                    )
                    if m:
                        qid = m.group(1)
                        if qid not in seen_ids:
                            question_ids.append(qid)
                            seen_ids.add(qid)

            doc.close()
        except Exception as e:
            print(f">>> [警告] 从 PDF 提取题号失败: {e}")
            return []

        return question_ids

    def _extract_parts_by_fixed_sections(self, pdf_path: str) -> Dict[str, Any]:
        """按固定分区标题解析题目：一、单项选择题 / 二、简答题 / 三、程序设计题。"""
        section_patterns = [
            ("choice", re.compile(r"^\s*(?:一|1)[、.．]\s*单项选择题\s*$")),
            ("short_answer", re.compile(r"^\s*(?:二|2)[、.．]\s*简答题\s*$")),
            ("programming", re.compile(r"^\s*(?:三|3)[、.．]\s*程序设计题\s*$")),
        ]
        q_start_re = re.compile(r"^\s*(\d{1,3}(?:_sub_\d+)?)\s*[\.、．\)]\s*(.*)")

        rows: List[Dict[str, Any]] = []
        doc = fitz.open(pdf_path)
        try:
            for page_idx in range(len(doc)):
                for line in self._extract_page_lines(doc[page_idx]):
                    text = str(line.get("text", ""))
                    if not text.strip():
                        continue
                    rows.append(
                        {
                            "page": page_idx,
                            "y": float(line.get("bbox", [0, 0, 0, 0])[1]),
                            "text": text,
                        }
                    )
        finally:
            doc.close()

        rows.sort(key=lambda item: (item["page"], item["y"]))

        events: List[Dict[str, Any]] = []
        current_section = ""
        for idx, row in enumerate(rows):
            text = row["text"].strip()
            for section_name, pattern in section_patterns:
                if pattern.match(text):
                    current_section = section_name
                    events.append({"kind": "section", "section": section_name, "idx": idx})
                    break
            else:
                m = q_start_re.match(text)
                if m and current_section:
                    events.append(
                        {
                            "kind": "question",
                            "section": current_section,
                            "idx": idx,
                            "id": m.group(1),
                        }
                    )

        question_events = [e for e in events if e["kind"] == "question"]
        section_events = [e for e in events if e["kind"] == "section"]
        parts: Dict[str, Any] = {"choice": [], "short_answer": [], "programming": []}

        for pos, event in enumerate(question_events):
            section = str(event["section"])
            start_idx = int(event["idx"])
            next_question_idx = (
                int(question_events[pos + 1]["idx"])
                if pos + 1 < len(question_events)
                else len(rows)
            )
            next_section_idx = next(
                (
                    int(se["idx"])
                    for se in section_events
                    if int(se["idx"]) > start_idx
                ),
                len(rows),
            )
            end_idx = min(next_question_idx, next_section_idx)
            question_text = "\n".join(row["text"] for row in rows[start_idx:end_idx])
            parts[section].append({"id": str(event["id"]), "question": question_text})

        return parts

    def _parts_look_misclassified(self, parts: Any) -> bool:
        """识别旧缓存中的坏分类：所有题都被塞进 choice，简答/程序为空。"""
        if not isinstance(parts, dict):
            return True
        choices = parts.get("choice")
        shorts = parts.get("short_answer")
        programs = parts.get("programming")
        if not isinstance(choices, list) or not isinstance(shorts, list) or not isinstance(programs, list):
            return True
        return len(choices) > 0 and len(shorts) == 0 and len(programs) == 0

    def _extract_homework_name_from_pdf(self, pdf_path: str) -> str:
        """用 AI 从 PDF 首页提取作业名称"""
        try:
            doc = fitz.open(pdf_path)
            first_page = doc[0]
            lines = self._extract_page_lines(first_page)
            doc.close()

            first_lines = "\n".join([line["text"] for line in lines[:10]])

            prompt = f"""请从以下 PDF 首页文本中提取作业名称。只需返回一个作业名称字符串，形如"第x课xxx"。

文本内容：
{first_lines}

返回格式：只输出作业名称，不要其他内容。"""

            messages = [{"role": "user", "content": prompt}]
            response = self._call_ai(
                self.simple_client,
                self.simple_model,
                messages,
                use_tools=False,
            )

            if response and hasattr(response, "choices"):
                homework_name = response.choices[0].message.content.strip()
                return homework_name if homework_name else "未命名作业"
        except Exception as e:
            print(f">>> [警告] 提取作业名称失败: {e}")

        return "未命名作业"

    def parse_pdf(
        self, pdf_path: str
    ) -> Tuple[str, Dict[str, Any], Dict[str, Dict[str, str]]]:
        """解析PDF并利用LLM提取四个部分的内容"""
        print(">>> 正在准备 PDF 多模态输入...")
        parse_pdf_image_inputs = self._prepare_parse_pdf_image_inputs(pdf_path)
        if not parse_pdf_image_inputs:
            raise ValueError("PDF 多模态输入准备失败，无法解析作业结构")
        print(f">>> 已准备 {len(parse_pdf_image_inputs)} 张 PDF 页面图片")
        for label, path in parse_pdf_image_inputs[:3]:  # 只打印前3张
            print(f"    - {label}: {path} (exists: {os.path.exists(path)})")
        if len(parse_pdf_image_inputs) > 3:
            print(f"    ... 及其他 {len(parse_pdf_image_inputs) - 3} 张")

        print(">>> 正在按固定分区标题解析题目结构...")
        parts = self._extract_parts_by_fixed_sections(pdf_path)
        all_question_ids = [
            str(q.get("id", ""))
            for key in ("choice", "short_answer", "programming")
            for q in parts.get(key, [])
            if isinstance(q, dict)
        ]
        if not all_question_ids:
            raise ValueError("未能按固定格式解析到题目，请检查 PDF 分区标题")
        print(
            f">>> 题目分类: 选择题 {len(parts['choice'])} 道，"
            f"简答题 {len(parts['short_answer'])} 道，"
            f"程序设计题 {len(parts['programming'])} 道"
        )

        print(">>> 正在生成题目截图到 problems 目录...")
        screenshots = self.generate_problem_screenshots(pdf_path, parts)

        print(">>> 正在用 AI 提取作业名称...")
        homework_name = self._extract_homework_name_from_pdf(pdf_path)

        print("\n" + "=" * 30)
        print("PDF 解析完成：")
        print(f"作业名称: {homework_name}")
        print(f"题目总数: {len(all_question_ids)} 道")
        print("题目截图目录: problems")
        print("=" * 30)

        return homework_name, parts, screenshots

    def _prepare_parse_pdf_image_inputs(
        self,
        pdf_path: str,
    ) -> List[Tuple[str, str]]:
        """将待解析作业 PDF 转成页面图片，供多模态结构解析使用。支持完整PDF提取。"""
        if not os.path.exists(pdf_path):
            return []

        out_dir = os.path.join("problems", "_parse_mm")
        os.makedirs(out_dir, exist_ok=True)

        inputs: List[Tuple[str, str]] = []
        try:
            doc = fitz.open(pdf_path)
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            page_count = len(doc)  # 提取完整PDF，无页数限制
            print(f">>> [PDF多模态] 正在转换 {page_count} 页 PDF 为图片...")

            if page_count > 100:
                print(
                    f">>> [提示] PDF 页数较多 ({page_count} 页)，可能需要较长时间和更多 API token"
                )

            for i in range(page_count):
                page = doc[i]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
                img_path = os.path.join(out_dir, f"{pdf_name}_parse_p{i + 1}.png")
                pix.save(img_path)
                inputs.append((f"作业PDF第{i + 1}页", img_path))
                if i == 0:
                    print(f">>> [PDF多模态] 第一张图片已保存: {img_path}")
                if (i + 1) % 50 == 0:
                    print(f">>> [PDF多模态] 已处理 {i + 1} 页...")
            doc.close()
            print(f">>> [PDF多模态] 共生成 {len(inputs)} 张图片")
        except Exception as e:
            print(f">>> [警告] 作业 PDF 多模态转换失败: {e}")
            return []

        return inputs

    def solve_choice_questions(
        self,
        choices_list: List[Dict[str, Any]],
        image_map: Dict[str, str],
        reference_materials_text: str = "无",
        reference_image_inputs: Optional[List[Tuple[str, str]]] = None,
        on_question_pass: Optional[Callable[[str, str], None]] = None,
    ) -> List[str]:
        """使用复杂模型解决选择题 (CoT + 每题搜索 + 循环审阅)"""
        if not choices_list:
            return [""] * self.CHOICE_QUESTION_LIMIT

        student_name = self.config["student_info"]["name"]
        final_ans = [""] * self.CHOICE_QUESTION_LIMIT
        pending_choices = choices_list.copy()
        reference_image_inputs = reference_image_inputs or []

        max_rounds = 10
        for round_idx in range(max_rounds):
            if not pending_choices:
                break

            print(
                f"\n>>> 选择题处理第 {round_idx + 1} 轮 (剩余 {len(pending_choices)} 题)..."
            )

            still_pending = []
            for q in pending_choices:
                qid = str(q.get("id"))
                print(f"  [搜索中] 第 {qid} 题...")
                search_res, search_image_inputs, has_search_evidence = (
                    self._search_knowledge_context(
                        f"计算机网络 {q.get('question')}",
                        question_id=qid,
                        question_context=q,
                        question_image_inputs=self._collect_question_image_inputs(
                            [q], image_map
                        ),
                    )
                )
                if not has_search_evidence:
                    print(f"  [题号 {qid}] 未检索到相关证据，跳过本轮答题")
                    q_with_feedback = q.copy()
                    q_with_feedback["feedback"] = "上一轮未检索到相关证据，请继续检索后再作答。"
                    still_pending.append(q_with_feedback)
                    continue

                prompt = f"""你是一个计算机网络助教。请解决以下选择题。
主题：计算机网络

要求使用 Chain-of-Thought (CoT) 模式：
1. 深入分析题目背景，并在 <thought> 标签内明确列出【考察知识点】。
2. 优先参考提供的【参考资料】与【参考背景信息】辅助推导。
3. 展现分步骤推导逻辑。
4. 最终答案必须写在 <answer> 标签内，且只能包含大写字母选项（如 A、AB、ACD），不得包含中文、标点、括号、前缀文本（如“答案：”）。

输出格式硬约束（必须全部满足）：
1. 只能输出一个 JSON 对象，首字符必须是 {{，末字符必须是 }}。
2. 禁止输出 Markdown、禁止输出代码块标记（如 ```json）、禁止输出任何解释文字。
3. JSON 顶层键必须是 "results"，且为数组。
4. 每个元素必须包含：
   - "id": 与输入题号一致
   - "analysis": "<thought>...推导...</thought>"
   - "answer": "<answer>AB</answer>"（AB 仅为示例，必须为合法选项字母组合）

示例：
{{
  "results": [
    {{
      "id": "1",
      "analysis": "<thought>【考察知识点】：...\n【推导逻辑】：...</thought>",
      "answer": "<answer>A</answer>"
    }}
  ]
}}

【参考资料】（启动时加载的参考文件）：
{reference_materials_text}

【参考背景信息】（当前题搜索结果）：
{search_res}

题目内容：
{json.dumps(q, ensure_ascii=False)}
"""
                solve_messages = self._build_image_message(
                    prompt,
                    self._collect_question_image_inputs([q], image_map)
                    + search_image_inputs
                    + reference_image_inputs,
                )

                response = self._call_ai_for_question(
                    q,
                    self._collect_question_image_inputs([q], image_map),
                    self.complex_client,
                    self.complex_model,
                    solve_messages,
                    response_format={"type": "json_object"},
                )

                if not response or not hasattr(response, "choices"):
                    still_pending.append(q)
                    continue
                content = response.choices[0].message.content
                if not content:
                    still_pending.append(q)
                    continue
                try:
                    data = self._parse_json_safe(content)
                except Exception as e:
                    print(f"  [题号 {qid}] JSON 解析失败: {e}")
                    still_pending.append(q)
                    continue

                # 防御性解析：兼容单题对象、results 数组和历史 ans 数组。
                res = None
                if isinstance(data, dict):
                    if "results" in data and isinstance(data.get("results"), list):
                        for item in data.get("results", []):
                            if isinstance(item, dict) and str(item.get("id")) == qid:
                                res = item
                                break
                    elif "ans" in data and isinstance(data.get("ans"), list):
                        for item in data.get("ans", []):
                            if isinstance(item, dict) and str(item.get("id")) == qid:
                                res = item
                                break
                    elif str(data.get("id", qid)) == qid:
                        res = data
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and str(item.get("id")) == qid:
                            res = item
                            break

                if not res:
                    still_pending.append(q)
                    continue

                # 严格提取答案：必须有 <answer>...</answer> 标签
                raw_answer = str(res.get("answer", ""))
                ans_match = re.search(r"<answer>(.*?)</answer>", raw_answer, re.S)

                # 如果无法提取到标签，跳过此题进入下一轮
                if not ans_match:
                    print(f"  [题号 {qid}] 答案缺少 <answer> 标签，进入下一轮重试")
                    still_pending.append(q)
                    continue

                answer_str = ans_match.group(1).strip()

                # 进一步验证答案是否符合规范（仅字母，长度合理）
                clean_ans = self._normalize_choice_answer(answer_str)
                if not clean_ans:
                    print(
                        f"  [题号 {qid}] 答案规范不符（{answer_str}），进入下一轮重试"
                    )
                    still_pending.append(q)
                    continue

                # 思路提取（可以宽松一些，因为思路主要用于展示和复习）
                thought_match = re.search(
                    r"<thought>(.*?)</thought>", str(res.get("analysis", "")), re.S
                )
                thought_str = thought_match.group(1).strip() if thought_match else ""

                review_prompt = f"""作为审阅专家，请严谨核查此选择题。
题目：{q.get('question')}
思路：{thought_str}
答案：{answer_str}
【参考资料】（启动时加载的参考文件）：
{reference_materials_text}

【参考背景信息】：{search_res}

核查要求（务必严谨）：
1. 【事实核查是核心】：你的主要任务是判断推导思路是否符合计算机网络协议和逻辑事实。
2. 【有目的的工具使用】：仅在需要验证具体数据、计算结果或协议细节时使用工具。禁止盲目搜索题目或执行无关代码。
3. 【思路校验】：检查“考察知识点”是否准确，推导过程是否存在逻辑跳跃或错误。
4. 【参考原题】：只有在确认网上存在高度匹配的原题时，才参考其标准答案。
5. 【身份契合度】：确认文风符合大二学生 {student_name} 的真实水平，去 AI 化。

输出要求：若思路正确且事实无误，输出中必须包含 "PASS"。否则，请指出具体的事实错误或逻辑漏洞。
"""
                review_messages = self._build_image_message(
                    review_prompt,
                    self._collect_question_image_inputs([q], image_map)
                    + search_image_inputs
                    + reference_image_inputs,
                )
                rev_res = self._call_ai_for_question(
                    q,
                    self._collect_question_image_inputs([q], image_map),
                    self.simple_client,
                    self.simple_model,
                    review_messages,
                    use_tools=True,
                )

                if not rev_res or not hasattr(rev_res, "choices"):
                    still_pending.append(q)
                    continue
                rev_content = rev_res.choices[0].message.content
                if rev_content and "PASS" in rev_content.strip().upper():
                    # 直接使用前面已验证的 clean_ans（冗余检查已在提取时完成）
                    print(f"  [题号 {qid}] 审阅通过: {clean_ans}")
                    try:
                        base_qid = qid.split("-")[0]
                        idx = int(base_qid) - 1
                        if 0 <= idx < self.CHOICE_QUESTION_LIMIT:
                            final_ans[idx] = clean_ans
                    except:
                        pass
                    if on_question_pass:
                        on_question_pass(qid, clean_ans)
                else:
                    reason = rev_content.strip() if rev_content else "未通过"
                    print(f"  [题号 {qid}] 审阅未通过: {reason}")
                    q_with_feedback = q.copy()
                    q_with_feedback["feedback"] = reason  # 修复：不污染题目本身
                    still_pending.append(q_with_feedback)

            pending_choices = still_pending

        return final_ans

    def solve_short_answers(
        self,
        short_answer_list: List[Dict[str, Any]],
        image_map: Dict[str, str],
        reference_materials_text: str = "无",
        reference_image_inputs: Optional[List[Tuple[str, str]]] = None,
        on_question_pass: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """使用复杂模型解决简答题，并由简单模型审阅（CoT 循环反馈机制）"""
        if not short_answer_list:
            return []

        student_name = self.config["student_info"]["name"]
        pending_questions = short_answer_list.copy()
        final_results_map = {}
        best_answers_per_q = {}  # 记录每题的最好答案
        no_improvement_rounds = 0  # 连续无进展轮数
        reference_image_inputs = reference_image_inputs or []

        max_rounds = 10
        for round_idx in range(max_rounds):
            if not pending_questions:
                break

            round_improved = False

            print(
                f"\n>>> 简答题处理第 {round_idx + 1} 轮 (剩余 {len(pending_questions)} 题)..."
            )
            still_pending = []
            for q in pending_questions:
                qid = str(q.get("id"))
                print(f"  [搜索中] 第 {qid} 题...")
                search_res, search_image_inputs, has_search_evidence = (
                    self._search_knowledge_context(
                        f"计算机网络 {q.get('question')}",
                        question_id=qid,
                        question_context=q,
                        question_image_inputs=self._collect_question_image_inputs(
                            [q], image_map
                        ),
                    )
                )
                if not has_search_evidence:
                    print(f"  [题号 {qid}] 未检索到相关证据，跳过本轮答题")
                    q_with_feedback = q.copy()
                    q_with_feedback["feedback"] = "上一轮未检索到相关证据，请继续检索后再作答。"
                    still_pending.append(q_with_feedback)
                    continue

                solve_prompt = f"""你是一个大二学生 {student_name}。正在完成计算机网络作业。
要求使用 CoT 模式进行推理：
1. 在 <thought> 标签内首先明确列出该题目的【考察知识点】，结合【参考资料】、【参考背景信息】和之前的【反馈意见】（如果有）进行逻辑推导。
2. 将最终给出的回答写在 <answer> 标签内。
3. 回答要求（像学生交给老师的作业答案）：
    - 语气自然、认真、书面化，像大学生写给任课老师看的课程作业，不要写成公文、论文摘要或审稿意见。
     - 格式规则：
         * 正文一律使用完整的中文段落，不能用 Markdown 加粗（**）、斜体（*）、无序列表（-）、有序列表、代码块等。
         * 如果答案里确实有多个事物需要逐项对比或分类列举（例如比较两种协议的字段、列出多个阶段的参数），才在段落之后插入一个标准 Markdown 表格；没有对比/分类需求就不用表格，直接写段落。
         * 表格格式：第一行是表头，第二行是分隔行（如 |---|---|），之后是数据行，每行都要有 | 开头和结尾。
     - 可读性硬约束：
         * 先给结论，再解释原因；不要先铺垫一大段抽象定义。
         * 句子尽量短，每句只表达一个核心意思，避免长串并列从句。
         * 专业术语可以直接使用，不要求逐个解释；但禁止堆砌生僻词、空话套话和夸张修辞。
         * 必须正确使用中文标点。每个分句都要有逗号、句号、分号等停顿标记，禁止整段只靠空格或连词硬拼。
         * 单句过长时必须主动断句。不要连续输出超长复合句，避免“一句话塞满整段信息”。
     - 文字风格：自然流畅，允许少量口语化连接词（如“可以理解为”“这里的关键是”），但结论必须准确；避免大量堆砌"首先/其次/再次/最后"或"一、二、三、四"式的机械分点；少用括号解释，禁止写成长段华丽但信息密度低的句子。
     - 内容要正确、完整，把原理和原因讲清楚。
   - 仅限中文，除了必要的英文专业术语。

输出格式硬约束（必须全部满足）：
1. 只能输出一个 JSON 对象，首字符必须是 {{，末字符必须是 }}。
2. 禁止输出 Markdown、禁止输出代码块标记（如 ```json）、禁止输出任何解释文字。
3. JSON 顶层键必须是 "answers"，且为数组。
4. 每个元素必须包含：
   - "id": 与输入题号一致
   - "analysis": "<thought>...推导...</thought>"
   - "answer": "<answer>最终回答内容</answer>"

示例：
{{
  "answers": [
    {{
      "id": "1",
      "analysis": "<thought>【考察知识点】：...\n【分析】：...</thought>",
      "answer": "<answer>...</answer>"
    }}
  ]
}}
【参考资料】（启动时加载的参考文件）：
{reference_materials_text}

【参考背景信息】（当前题搜索结果）：
{search_res}

待处理题目（包含题目和可能的反馈意见）：
{json.dumps(q, ensure_ascii=False)}
"""
                title = q.get("question", "")
                solve_messages = self._build_image_message(
                    solve_prompt,
                    self._collect_question_image_inputs([q], image_map)
                    + search_image_inputs
                    + reference_image_inputs,
                )
                response = self._call_ai_for_question(
                    q,
                    self._collect_question_image_inputs([q], image_map),
                    self.complex_client,
                    self.complex_model,
                    solve_messages,
                    response_format={"type": "json_object"},
                )

                if not response or not hasattr(response, "choices"):
                    still_pending.append(q)
                    continue
                content = response.choices[0].message.content
                if not content:
                    still_pending.append(q)
                    continue

                try:
                    data = self._parse_json_safe(content)
                except Exception as e:
                    print(f"  [题号 {qid}] JSON 解析失败: {e}")
                    still_pending.append(q)
                    continue

                res = None
                if isinstance(data, dict):
                    if "answers" in data and isinstance(data.get("answers"), list):
                        for item in data.get("answers", []):
                            if isinstance(item, dict) and str(item.get("id")) == qid:
                                res = item
                                break
                    elif "results" in data and isinstance(data.get("results"), list):
                        for item in data.get("results", []):
                            if isinstance(item, dict) and str(item.get("id")) == qid:
                                res = item
                                break
                    elif str(data.get("id", qid)) == qid:
                        res = data
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and str(item.get("id")) == qid:
                            res = item
                            break

                if not res:
                    still_pending.append(q)
                    continue

                raw_ans = str(res.get("answer", ""))
                ans_match = re.search(r"<answer>(.*?)</answer>", raw_ans, re.S)
                ans = ans_match.group(1).strip() if ans_match else raw_ans

                review_prompt = f"""你是一个计算机网络审阅专家。请严谨审阅以下简答题答案。
题目：{title}
回答：{ans}
【参考资料】（启动时加载的参考文件）：
{reference_materials_text}

【参考背景信息】：{search_res}

审阅标准（核心优先 - 允许小缺陷）：
1. 【知识点与逻辑】（最关键）：回答是否准确命中该题的计算机网络核心知识点？推导逻辑是否正确？允许有小的计算或细节错误，但原理必须对。
2. 【可读性】（次关键）：是否能看懂？允许有小的标点缺陷或断句不完美，只要能读、有逻辑就可以。严格反对：无标点堆砌、华丽但无信息、机械套话。
3. 【文风审查】：文风应自然、清楚、像学生提交给老师的课程作业答案。重点检查是否“可读易懂”：
    - 是否先给结论再解释；
    - 是否存在过长句、术语堆叠、空话套话；
    - 是否出现华丽修辞但缺少有效信息。
    - 标点是否完整、断句是否清晰；若出现无标点堆砌或超长句连写，判为不通过。
    若存在晦涩难懂、学术腔过重、机械分点或括号解释过多，判为不通过。
4. 【格式要求】：正文必须是完整中文段落，禁止使用加粗、斜体、列表、代码块等 Markdown 语法。如果答案里包含多项对比或分类列举，应在段落后出现标准 Markdown 表格（含表头行和分隔行）；如果没有使用表格但内容明显需要对比，请指出。

输出硬约束（必须遵守）：
1. 若该答案可接受（允许小缺陷），你的回复必须包含大写字符串 "PASS"。
2. 若该答案不可接受，你的回复必须不包含 "PASS"，并给出一句具体错误原因。
3. 禁止输出模糊结论（例如“基本可以”“差不多”）。
"""
                review_messages = self._build_image_message(
                    review_prompt,
                    self._collect_question_image_inputs([q], image_map)
                    + search_image_inputs
                    + reference_image_inputs,
                )
                rev_res = self._call_ai_for_question(
                    q,
                    self._collect_question_image_inputs([q], image_map),
                    self.simple_client,
                    self.simple_model,
                    review_messages,
                    use_tools=True,
                )

                if not rev_res or not hasattr(rev_res, "choices"):
                    still_pending.append(q)
                    continue
                rev_content = rev_res.choices[0].message.content

                # 更新最佳答案记录
                if qid not in best_answers_per_q or len(ans) > len(
                    best_answers_per_q[qid][0]
                ):
                    best_answers_per_q[qid] = (ans, q.get("id", ""), title)
                    round_improved = True

                # 通过判定唯一标准：审阅回复中包含 PASS
                is_pass = bool(rev_content and "PASS" in rev_content.upper())

                if is_pass:
                    print(f"  [题号 {qid}] 审阅通过")
                    passed_result = {
                        "index": q.get("id", ""),
                        "title": title,
                        "answer": ans,
                    }
                    final_results_map[qid] = passed_result
                    if on_question_pass:
                        on_question_pass(qid, passed_result)
                else:
                    reason = rev_content.strip() if rev_content else "审阅未通过"
                    print(f"  [题号 {qid}] 审阅未通过: {reason[:100]}")
                    q_with_feedback = q.copy()
                    q_with_feedback["feedback"] = reason  # 修复
                    still_pending.append(q_with_feedback)

            pending_questions = still_pending

            # 早停机制：连续3轮无进展则用最佳答案强制通过
            if not round_improved:
                no_improvement_rounds += 1
            else:
                no_improvement_rounds = 0

            if no_improvement_rounds >= 3:
                print(f"\n>>> [提示] 连续 3 轮无进展，使用最佳答案强制通过剩余题目...")
                for qid, (best_ans, q_index, q_title) in best_answers_per_q.items():
                    if qid not in final_results_map:
                        final_results_map[qid] = {
                            "index": q_index,
                            "title": q_title,
                            "answer": best_ans,
                        }
                break

        # 整理结果
        results = []
        for q in short_answer_list:
            qid = str(q.get("id"))
            if qid in final_results_map:
                results.append(final_results_map[qid])
            else:
                results.append(
                    {
                        "index": q.get("id", ""),
                        "title": q.get("question", ""),
                        "answer": "（未通过审阅）",
                    }
                )

        return results

    def handle_programming(self, prog_list: List[Dict[str, Any]]) -> str:
        """处理程序设计题"""
        if not prog_list:
            return ""

        with open("project_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()

        gitee_links = []
        for p in prog_list:
            print("\n" + "=" * 20)
            print(f"处理程序设计题 [{p.get('id', '?')}]:")
            print(p.get("question", ""))
            print("=" * 20)
            user_prompt = (
                f"系统提示词: {system_prompt}\n当前题目: {p.get('question', '')}"
            )
            print(f"生成的提示词已准备好供参考: \n--BEGIN--\n{user_prompt}\n--END--")
            gitee_link = input(
                f"\n请输入第 {p.get('id', '?')} 题项目完成后的 Gitee 链接: "
            ).strip()
            if not gitee_link.startswith("http"):
                gitee_link = "尚未提供有效链接"
            gitee_links.append(f"{p.get('id', '?')}:{gitee_link}")

        return "\n".join(gitee_links)

    def _clean_markdown(self, text: str) -> str:
        """强力去除文本中的 Markdown 语法，返回纯文本"""
        if not text:
            return ""
        # 如果不是字符串（例如是 Subdoc 对象），直接返回
        if not isinstance(text, str):
            return text

        # 1. 去除代码块标识
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        # 2. 去除加粗/斜体
        text = re.sub(r"\*\*+(.*?)\*\*+", r"\1", text)
        text = re.sub(r"\*+(.*?)\*+", r"\1", text)
        text = re.sub(r"__+(.*?)__+", r"\1", text)
        text = re.sub(r"_+(.*?)_+", r"\1", text)
        # 3. 去除标题符号
        text = re.sub(r"^#+\s+", "", text, flags=re.M)
        # 4. 去除行内代码
        text = re.sub(r"`(.*?)`", r"\1", text)
        # 5. 去除链接
        text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
        # 6. 去除列表符号 (仅去除行首的 * 或 - )
        text = re.sub(r"^[\s\t]*[\*\-\+]\s+", "", text, flags=re.M)
        # 7. 去除数字列表开头的点
        text = re.sub(r"^[\s\t]*\d+\.\s+", "", text, flags=re.M)
        return text.strip()

    def _text_to_subdoc(self, tpl: DocxTemplate, text: str):
        """将包含 Markdown 表格的文本转换为 Subdoc 对象以插入原生 Word 内容"""

        def _is_table_line(line: str) -> bool:
            s = line.strip()
            return s.count("|") >= 2

        def _is_separator_line(line: str) -> bool:
            # 兼容 --- / :--- / ---: / :---: 等 Markdown 分隔行
            s = line.strip()
            if not s:
                return False
            if s.startswith("|"):
                s = s[1:]
            if s.endswith("|"):
                s = s[:-1]
            cells = [c.strip() for c in s.split("|")]
            if not cells:
                return False
            for c in cells:
                if not c:
                    return False
                if not re.fullmatch(r":?-{3,}:?", c):
                    return False
            return True

        def _split_row_cells(line: str) -> List[str]:
            s = line.strip()
            if s.startswith("|"):
                s = s[1:]
            if s.endswith("|"):
                s = s[:-1]
            return [c.strip() for c in s.split("|")]

        sd = tpl.new_subdoc()
        if not text or not isinstance(text, str):
            return sd

        lines = text.splitlines()
        i = 0
        while i < len(lines):
            cur = lines[i].rstrip()

            # 探测 Markdown 表格块：当前行像表格，且下一行是分隔线
            if (
                i + 1 < len(lines)
                and _is_table_line(cur)
                and _is_separator_line(lines[i + 1])
            ):
                table_lines = [cur, lines[i + 1].rstrip()]
                j = i + 2
                while j < len(lines):
                    nxt = lines[j].rstrip()
                    if not nxt.strip() or not _is_table_line(nxt):
                        break
                    table_lines.append(nxt)
                    j += 1

                rows_data: List[List[str]] = []
                for raw in table_lines:
                    if _is_separator_line(raw):
                        continue
                    cells = _split_row_cells(raw)
                    if any(c.strip() for c in cells):
                        rows_data.append(cells)

                if rows_data:
                    num_rows = len(rows_data)
                    num_cols = max(len(r) for r in rows_data)
                    table = sd.add_table(rows=num_rows, cols=num_cols)
                    table.style = "Table Grid"
                    for r_idx, row_cells in enumerate(rows_data):
                        padded = row_cells + [""] * (num_cols - len(row_cells))
                        for c_idx, cell_text in enumerate(padded):
                            table.cell(r_idx, c_idx).text = self._clean_markdown(
                                cell_text
                            )
                    sd.add_paragraph("")
                else:
                    fallback_text = self._clean_markdown("\n".join(table_lines))
                    if fallback_text:
                        sd.add_paragraph(fallback_text)

                i = j
                continue

            cleaned = self._clean_markdown(cur)
            if cleaned:
                sd.add_paragraph(cleaned)
            i += 1

        return sd

    def generate_docx(self, homework_name: str, context: Dict[str, Any]):
        """生成最终的 docx 文件，包含表格支持与兜底清理"""
        tpl = DocxTemplate("template.docx")

        # 不污染原始 context，避免后续反馈阶段 json.dumps(context) 遇到 Subdoc 序列化错误
        render_context = deepcopy(context)

        # 对渲染上下文中的简答题内容进行处理
        # 模板使用 {{p q.answer }} 段落级替换，必须始终传入 Subdoc 对象
        if "questions" in render_context:
            render_context["questions"] = self._build_render_questions_without_title(
                render_context["questions"]
            )
            for q in render_context["questions"]:
                ans_text = q.get("answer", "")
                q["answer"] = self._text_to_subdoc(tpl, ans_text)

        tpl.render(render_context)
        tpl_any = cast(Any, tpl)
        safe_name = re.sub(r'[\\/:*?"<>|]', "_", homework_name)
        output_name = f"{safe_name}.docx"

        # 常见场景：目标 docx 正在被 Word 占用，先重试再降级到新文件名。
        max_retries = 5
        retry_interval_sec = 1.5
        last_permission_err: Optional[PermissionError] = None
        for attempt in range(1, max_retries + 1):
            try:
                tpl_any.save(output_name)
                return output_name
            except PermissionError as e:
                last_permission_err = e
                if attempt < max_retries:
                    print(
                        f">>> [警告] 保存失败（文件可能被占用）: {output_name}，{retry_interval_sec}s 后重试 "
                        f"({attempt}/{max_retries})"
                    )
                    time.sleep(retry_interval_sec)

        fallback_name = f"{safe_name}_{time.strftime('%Y%m%d_%H%M%S')}.docx"
        try:
            tpl_any.save(fallback_name)
            print(f">>> [提示] 原文件仍被占用，已改为新文件名输出: {fallback_name}")
            return fallback_name
        except PermissionError:
            if last_permission_err is not None:
                raise last_permission_err
            raise

    def _parse_json_safe(self, content: str) -> Any:
        """安全解析 JSON，自动处理代码块标记包裹的情况"""
        if not content:
            return None
        # 尝试提取 ```json ... ``` 中间的内容
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.S)
        if match:
            content = match.group(1).strip()
        return json.loads(content)

    def _prepare_reference_pdf_image_inputs(
        self,
        reference_pdf_paths: List[str],
        max_pages_per_pdf: int = 6,
    ) -> List[Tuple[str, str]]:
        """将参考 PDF 转为多模态图片输入（每个 PDF 限制前若干页）。"""
        inputs: List[Tuple[str, str]] = []
        if not reference_pdf_paths:
            return inputs

        out_dir = os.path.join("problems", "_reference_mm")
        os.makedirs(out_dir, exist_ok=True)

        for pdf_path in reference_pdf_paths:
            if not os.path.exists(pdf_path):
                print(f">>> [警告] 参考 PDF 不存在，已忽略: {pdf_path}")
                continue

            try:
                doc = fitz.open(pdf_path)
                pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                page_count = min(len(doc), max_pages_per_pdf)
                for i in range(page_count):
                    page = doc[i]
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
                    img_path = os.path.join(out_dir, f"{pdf_name}_p{i + 1}.png")
                    pix.save(img_path)
                    inputs.append((f"参考PDF《{pdf_name}》第{i + 1}页", img_path))
                doc.close()
            except Exception as e:
                print(f">>> [警告] 参考 PDF 多模态转换失败: {pdf_path}, {e}")

        return inputs

    def _read_md_text(self, md_path: str) -> str:
        """读取 Markdown 文本内容"""
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            max_chars = 40000
            if len(text) > max_chars:
                return text[:max_chars]
            return text
        except Exception as e:
            print(f">>> [警告] 读取参考 Markdown 文本失败: {e}")
            return ""

    def _prepare_reference_materials(self, reference_md_paths: List[str]) -> str:
        """加载参考 Markdown，并整理为每题都可复用的参考资料文本。"""
        blocks: List[str] = []

        def _append_block(kind: str, src_path: str, content: str):
            if not content:
                return
            blocks.append(
                f"[参考文件] 类型: {kind}\n路径: {src_path}\n内容:\n{content}"
            )

        for p in reference_md_paths:
            if not os.path.exists(p):
                print(f">>> [警告] 参考 Markdown 不存在，已忽略: {p}")
                continue
            _append_block("md", p, self._read_md_text(p))

        if not blocks:
            return "无"

        joined = "\n\n".join(blocks)
        max_chars = 120000
        if len(joined) > max_chars:
            return joined[:max_chars]
        return joined

    def _normalize_choice_answer(self, value: Any) -> str:
        """规范化选择题答案：仅保留字母，去空格后长度必须小于 10。"""
        if not isinstance(value, str):
            return ""

        clean_val = re.sub(r"[^A-Za-z]", "", value).upper()
        if not clean_val:
            return ""
        if len(clean_val) >= 10:
            return ""
        return clean_val

    def _has_valid_choice_cache(self, values: Any) -> bool:
        """检查缓存中的选择题答案列表是否全部满足格式约束。"""
        if not isinstance(values, list):
            return False

        for val in values:
            if not val:
                continue
            if not self._normalize_choice_answer(val):
                return False
        return True

    def _build_render_questions_without_title(
        self, questions: Any
    ) -> List[Dict[str, Any]]:
        """为 Word 渲染准备简答题结构：仅保留 index 与 answer。"""
        normalized: List[Dict[str, Any]] = []
        if not isinstance(questions, list):
            return normalized

        for q in questions:
            if not isinstance(q, dict):
                continue
            q_dict = cast(Dict[str, Any], q)
            normalized.append(
                {
                    "index": q_dict.get("index", ""),
                    "answer": self._clean_markdown(str(q_dict.get("answer", ""))),
                }
            )

        return normalized

    def _validate_adjusted_context_with_screenshots(
        self,
        candidate_context: Any,
        current_context: Dict[str, Any],
        parts: Dict[str, Any],
        screenshots: Dict[str, Dict[str, str]],
    ) -> Tuple[bool, str]:
        """按截图题目划分校验反馈 JSON；不满足则要求重试。"""
        if not isinstance(candidate_context, dict):
            return False, "返回内容不是 JSON 对象"

        cand = cast(Dict[str, Any], candidate_context)

        if "ans" not in cand or not isinstance(cand.get("ans"), list):
            return False, "缺少 ans 列表"
        if "questions" not in cand or not isinstance(cand.get("questions"), list):
            return False, "缺少 questions 列表"

        candidate_ans = cast(List[Any], cand.get("ans", []))
        current_ans = cast(List[Any], current_context.get("ans", []))
        if len(candidate_ans) != len(current_ans):
            return (
                False,
                f"ans 长度不匹配，期望 {len(current_ans)}，实际 {len(candidate_ans)}",
            )

        # 选择题题号优先来自截图映射；若无截图则回退到解析题目。
        raw_choice_ids = list((screenshots.get("choice") or {}).keys())
        if not raw_choice_ids:
            raw_choice_ids = [
                str(q.get("id", "")).strip()
                for q in cast(List[Any], parts.get("choice", []))
                if isinstance(q, dict)
            ]

        allowed_choice_indexes = set()
        for qid in raw_choice_ids:
            try:
                idx = int(str(qid).split("-")[0]) - 1
                if 0 <= idx < len(candidate_ans):
                    allowed_choice_indexes.add(idx)
            except Exception:
                continue

        for i, val in enumerate(candidate_ans):
            if not val:
                continue
            if i not in allowed_choice_indexes:
                return False, f"ans 第 {i + 1} 项不属于截图中的选择题"
            if not self._normalize_choice_answer(val):
                return False, f"ans 第 {i + 1} 项格式非法"

        candidate_questions = cast(List[Any], cand.get("questions", []))
        current_questions = cast(List[Any], current_context.get("questions", []))
        if len(candidate_questions) != len(current_questions):
            return (
                False,
                f"questions 长度不匹配，期望 {len(current_questions)}，实际 {len(candidate_questions)}",
            )

        # 简答题题号优先来自截图映射；若无截图则回退到解析题目。
        expected_short_ids = list((screenshots.get("short_answer") or {}).keys())
        if not expected_short_ids:
            expected_short_ids = [
                str(q.get("id", "")).strip()
                for q in cast(List[Any], parts.get("short_answer", []))
                if isinstance(q, dict)
            ]

        if len(expected_short_ids) != len(candidate_questions):
            return (
                False,
                "questions 数量与截图中的简答题数量不一致",
            )

        expected_index_seq = [str(v) for v in expected_short_ids]
        for i, q in enumerate(candidate_questions):
            if not isinstance(q, dict):
                return False, f"questions 第 {i + 1} 项不是对象"
            q_dict = cast(Dict[str, Any], q)
            idx_val = str(q_dict.get("index", "")).strip()
            if idx_val != expected_index_seq[i]:
                return (
                    False,
                    f"questions 第 {i + 1} 项题号不匹配，期望 {expected_index_seq[i]}，实际 {idx_val}",
                )
            if "answer" not in q_dict:
                return False, f"questions 第 {i + 1} 项缺少 answer"

        return True, "PASS"

    def _guard_context_update(
        self,
        original_context: Dict[str, Any],
        candidate_context: Any,
    ) -> Dict[str, Any]:
        """对模型返回的作业 JSON 做兜底约束：只允许变更允许变更的字段。"""
        if not isinstance(candidate_context, dict):
            return original_context
        candidate_dict = cast(Dict[str, Any], candidate_context)

        guarded = dict(original_context)

        # 第一部分元信息与第三部分 Git 地址禁止改动
        for fixed_key in [
            "homework_name",
            "class_name",
            "student_id",
            "name",
            "gitee_info",
        ]:
            guarded[fixed_key] = original_context.get(fixed_key, "")

        # 第二部分-选择题：仅允许按原索引更新答案，长度与语义保持不变
        original_ans = cast(List[Any], original_context.get("ans", []))
        candidate_ans = cast(List[Any], candidate_dict.get("ans", []))
        merged_ans = original_ans.copy()
        for i in range(min(len(original_ans), len(candidate_ans))):
            val = candidate_ans[i]
            clean_val = self._normalize_choice_answer(val)
            if clean_val:
                merged_ans[i] = clean_val
        guarded["ans"] = merged_ans

        # 第二部分-简答题：只允许修改 answer，index/title 强制保留原值
        original_questions = cast(List[Any], original_context.get("questions", []))
        candidate_questions = cast(List[Any], candidate_dict.get("questions", []))
        merged_questions: List[Dict[str, Any]] = []
        for i, oq in enumerate(original_questions):
            if not isinstance(oq, dict):
                continue
            oq_dict = cast(Dict[str, Any], oq)

            new_answer: str = str(oq_dict.get("answer", ""))
            if i < len(candidate_questions) and isinstance(
                candidate_questions[i], dict
            ):
                cand_q = cast(Dict[str, Any], candidate_questions[i])
                cand_answer = cand_q.get("answer", new_answer)
                if cand_answer is not None:
                    new_answer = str(cand_answer)

            merged_questions.append(
                {
                    "index": oq_dict.get("index", ""),
                    "title": oq_dict.get("title", ""),
                    "answer": self._clean_markdown(new_answer),
                }
            )
        guarded["questions"] = merged_questions

        return guarded

    def _get_cache_path(self, pdf_path: str) -> str:
        """生成缓存文件路径"""
        cache_dir = ".homework_cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        # 基于PDF文件名生成缓存文件
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        return os.path.join(cache_dir, f"{pdf_name}.cache.json")

    def _load_cache(
        self,
        pdf_path: str,
        input_paths: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """加载缓存，若缓存比所有输入文件都新则返回缓存数据"""
        cache_path = self._get_cache_path(pdf_path)

        # 检查缓存文件是否存在
        if not os.path.exists(cache_path):
            return None

        cache_mtime = os.path.getmtime(cache_path)

        # 缓存必须晚于所有输入文件（目标 PDF + 参考文件）
        all_inputs = [pdf_path] + (input_paths or [])
        latest_input_mtime: Optional[float] = None
        latest_input_file = ""
        for p in all_inputs:
            if not p or not os.path.exists(p):
                continue
            mtime = os.path.getmtime(p)
            if latest_input_mtime is None or mtime > latest_input_mtime:
                latest_input_mtime = mtime
                latest_input_file = p

        if latest_input_mtime is not None and cache_mtime <= latest_input_mtime:
            print(
                f">>> [缓存] 缓存已过期（输入文件已更新）: {latest_input_file}，将重新处理"
            )
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            print(f">>> [缓存] 成功加载缓存: {cache_path}")
            return cache_data
        except Exception as e:
            print(f">>> [缓存] 加载缓存失败: {e}，将重新处理")
            return None

    def _save_cache(self, pdf_path: str, cache_data: Dict[str, Any]) -> None:
        """保存缓存数据"""
        cache_path = self._get_cache_path(pdf_path)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            print(f">>> [缓存] 已保存缓存: {cache_path}")
        except Exception as e:
            print(f">>> [缓存] 保存缓存失败: {e}")

    def run(
        self,
        pdf_path: str,
    ):
        print(f">>> 开始解析 PDF: {pdf_path}")

        print(">>> [知识库] 检查 references 目录索引状态...")
        changed = self.reference_index.ensure_up_to_date()
        if changed:
            print(">>> [知识库] 检测到参考文件变动，已重建索引")
        else:
            print(">>> [知识库] 索引已是最新")

        all_input_paths = [pdf_path]

        # 尝试加载缓存
        cache_data = self._load_cache(pdf_path, all_input_paths)
        if cache_data is None:
            cache_data = {}

        parsed_homework_name = cache_data.get("parsed_homework_name")
        parsed_parts = cache_data.get("parsed_parts")
        parsed_screenshots = cache_data.get("parsed_screenshots")

        has_parsed_cache = (
            isinstance(parsed_homework_name, str)
            and isinstance(parsed_parts, dict)
            and isinstance(parsed_screenshots, dict)
            and not self._parts_look_misclassified(parsed_parts)
        )

        if has_parsed_cache:
            print(">>> [缓存] 使用首次解析结果（作业名称、题目结构与截图映射）")
            homework_name = parsed_homework_name
            parts = parsed_parts
            screenshots = parsed_screenshots
            if not cache_data.get("screenshots_dir"):
                cache_data["screenshots_dir"] = "problems"
                self._save_cache(pdf_path, cache_data)
        else:
            homework_name, parts, screenshots = self.parse_pdf(pdf_path)
            cache_data["parsed_homework_name"] = homework_name
            cache_data["parsed_parts"] = parts
            cache_data["parsed_screenshots"] = screenshots
            cache_data["screenshots_dir"] = "problems"
            cache_data["parsed_cached_at"] = time.time()
            for stale_key in (
                "choice_ans",
                "choice_cached_at",
                "choice_answers_by_id",
                "short_answers",
                "short_answers_cached_at",
                "short_answers_by_id",
                "programming_info",
                "programming_cached_at",
            ):
                cache_data.pop(stale_key, None)
            self._save_cache(pdf_path, cache_data)

        print(f">>> 作业名称: {homework_name}")
        if cache_data.get("screenshots_dir"):
            print(f">>> 题目截图目录: {cache_data.get('screenshots_dir')}")

        reference_image_inputs: List[Tuple[str, str]] = []
        reference_materials_text = (
            "参考资料由统一 search 工具动态检索，"
            "来源于 references/ 本地知识库与联网原文。"
        )

        # 处理选择题（按题号缓存）
        choice_answers_by_id = cache_data.get("choice_answers_by_id")
        if not isinstance(choice_answers_by_id, dict):
            choice_answers_by_id = {}
            cache_data["choice_answers_by_id"] = choice_answers_by_id

        # 兼容旧整段缓存：迁移成按题号缓存。
        if (
            cache_data.get("choice_ans")
            and self._has_valid_choice_cache(cache_data.get("choice_ans"))
        ):
            legacy_ans = cache_data.get("choice_ans", [])
            for q in parts.get("choice", []):
                qid = str(q.get("id", "")).strip()
                try:
                    idx = int(qid.split("-")[0]) - 1
                except Exception:
                    continue
                if qid and qid not in choice_answers_by_id and 0 <= idx < len(legacy_ans):
                    val = self._normalize_choice_answer(str(legacy_ans[idx]))
                    if val:
                        choice_answers_by_id[qid] = {
                            "answer": val,
                            "cached_at": cache_data.get("choice_cached_at", time.time()),
                        }

        ans = [""] * self.CHOICE_QUESTION_LIMIT
        pending_choices: List[Dict[str, Any]] = []
        for q in parts.get("choice", []):
            qid = str(q.get("id", "")).strip()
            cached = choice_answers_by_id.get(qid)
            cached_answer = ""
            if isinstance(cached, dict):
                cached_answer = self._normalize_choice_answer(str(cached.get("answer", "")))
            elif isinstance(cached, str):
                cached_answer = self._normalize_choice_answer(cached)
            if cached_answer:
                try:
                    idx = int(qid.split("-")[0]) - 1
                    if 0 <= idx < self.CHOICE_QUESTION_LIMIT:
                        ans[idx] = cached_answer
                except Exception:
                    pass
            else:
                pending_choices.append(q)

        if pending_choices:
            print(
                f">>> 正在处理选择题...（缓存命中 {len(parts.get('choice', [])) - len(pending_choices)} 道，待处理 {len(pending_choices)} 道）"
            )

            def _cache_choice(qid: str, answer: str) -> None:
                choice_answers_by_id[qid] = {
                    "answer": answer,
                    "cached_at": time.time(),
                }
                cache_data["choice_answers_by_id"] = choice_answers_by_id
                self._save_cache(pdf_path, cache_data)

            new_ans = self.solve_choice_questions(
                pending_choices,
                screenshots.get("choice", {}),
                reference_materials_text,
                reference_image_inputs,
                on_question_pass=_cache_choice,
            )
            for i, val in enumerate(new_ans):
                if val:
                    ans[i] = val
        else:
            print(">>> [缓存] 所有选择题均已按题号缓存")

        cache_data["choice_ans"] = ans
        cache_data["choice_cached_at"] = time.time()
        self._save_cache(pdf_path, cache_data)

        # 处理简答题（按题号缓存）
        short_answers_by_id = cache_data.get("short_answers_by_id")
        if not isinstance(short_answers_by_id, dict):
            short_answers_by_id = {}
            cache_data["short_answers_by_id"] = short_answers_by_id

        if isinstance(cache_data.get("short_answers"), list):
            for item in cache_data.get("short_answers", []):
                if not isinstance(item, dict):
                    continue
                qid = str(item.get("index", "")).strip()
                if qid and qid not in short_answers_by_id and item.get("answer"):
                    short_answers_by_id[qid] = {
                        "result": item,
                        "cached_at": cache_data.get("short_answers_cached_at", time.time()),
                    }

        pending_short: List[Dict[str, Any]] = []
        for q in parts.get("short_answer", []):
            qid = str(q.get("id", "")).strip()
            cached = short_answers_by_id.get(qid)
            cached_result = cached.get("result") if isinstance(cached, dict) else None
            if not isinstance(cached_result, dict) or "answer" not in cached_result:
                pending_short.append(q)

        if pending_short:
            print(
                f">>> 正在处理简答题...（缓存命中 {len(parts.get('short_answer', [])) - len(pending_short)} 道，待处理 {len(pending_short)} 道）"
            )

            def _cache_short(qid: str, result: Dict[str, Any]) -> None:
                short_answers_by_id[qid] = {
                    "result": result,
                    "cached_at": time.time(),
                }
                cache_data["short_answers_by_id"] = short_answers_by_id
                self._save_cache(pdf_path, cache_data)

            _ = self.solve_short_answers(
                pending_short,
                screenshots.get("short_answer", {}),
                reference_materials_text,
                reference_image_inputs,
                on_question_pass=_cache_short,
            )
        else:
            print(">>> [缓存] 所有简答题均已按题号缓存")

        questions = []
        for q in parts.get("short_answer", []):
            qid = str(q.get("id", "")).strip()
            cached = short_answers_by_id.get(qid)
            cached_result = cached.get("result") if isinstance(cached, dict) else None
            if isinstance(cached_result, dict):
                questions.append(cached_result)
            else:
                questions.append(
                    {
                        "index": qid,
                        "title": q.get("question", ""),
                        "answer": "（未通过审阅）",
                    }
                )

        cache_data["short_answers"] = questions
        cache_data["short_answers_cached_at"] = time.time()
        self._save_cache(pdf_path, cache_data)

        # 处理程序设计题（检查缓存）
        if "programming_info" in cache_data and cache_data.get("programming_info"):
            print(">>> [缓存] 使用缓存的程序设计题答案")
            gitee_info = cache_data["programming_info"]
        else:
            print(">>> 正在处理程序设计题...")
            gitee_info = self.handle_programming(parts["programming"])
            cache_data["programming_info"] = gitee_info
            cache_data["programming_cached_at"] = time.time()
            self._save_cache(pdf_path, cache_data)

        context: Dict[str, Any] = {
            "homework_name": homework_name,
            "class_name": self.config["student_info"]["class"],
            "student_id": self.config["student_info"]["id"],
            "name": self.config["student_info"]["name"],
            "ans": ans,
            "questions": questions,
            "gitee_info": gitee_info,
        }

        # 保存最终完整缓存
        cache_data["final_context"] = context
        cache_data["completed_at"] = time.time()
        self._save_cache(pdf_path, cache_data)

        output_file = self.generate_docx(homework_name, context)
        print(f"\n[成功] 作业已生成: {output_file}")

        while True:
            feedback = input(
                "\n请输入反馈 (输入 'OK' 确认并退出, 或输入修改意见): "
            ).strip()
            if feedback.upper() == "OK":
                print("作业已确认，程序退出。")
                break
            else:
                print(f">>> 正在根据反馈修改作业: {feedback}")
                adjustment_prompt = f"""用户对生成的作业提出了修改意见："{feedback}"
请根据意见调整当前的作业内容。
当前内容：{json.dumps(context, ensure_ascii=False)}

要求：
1. 严格输出调整后的完整 JSON。
2. 只能输出一个 JSON 对象，首字符必须是 {{，末字符必须是 }}。
3. 禁止输出 Markdown、禁止输出代码块标记（如 ```json）、禁止输出解释文字。
4. 选择题 ans 的每一项只能是大写字母选项组合（如 A、AB、ACD），不得包含中文、标点和前缀文本。
5. 简答题文字必须更易懂：先给结论，再给理由；句子简洁；允许轻微口语化表达，但不能牺牲专业准确性。
6. 简答题必须使用正常中文标点并清晰断句，禁止无标点堆砌长段文字。
7. questions 每一项必须保留 index、title、answer 三个字段（title 仅用于审阅与缓存）。
"""
                max_adjust_retries = 5
                last_reason = ""
                adjusted_ok = False
                for retry in range(1, max_adjust_retries + 1):
                    retry_prompt = adjustment_prompt
                    if last_reason:
                        retry_prompt += (
                            "\n\n上一次返回的 JSON 不满足题目划分校验，请严格修复后重试。"
                            f"\n不通过原因：{last_reason}"
                        )

                    adj_res = self._call_ai(
                        self.complex_client,
                        self.complex_model,
                        [{"role": "user", "content": retry_prompt}],
                        response_format={"type": "json_object"},
                    )
                    if not adj_res or not hasattr(adj_res, "choices"):
                        last_reason = "模型未返回有效响应"
                        print(
                            f">>> [警告] 反馈修复第 {retry}/{max_adjust_retries} 次失败: {last_reason}"
                        )
                        continue

                    adj_content = adj_res.choices[0].message.content
                    if not adj_content:
                        last_reason = "模型返回内容为空"
                        print(
                            f">>> [警告] 反馈修复第 {retry}/{max_adjust_retries} 次失败: {last_reason}"
                        )
                        continue

                    try:
                        adjusted_context = self._parse_json_safe(adj_content)
                    except Exception as e:
                        last_reason = f"JSON 解析失败: {e}"
                        print(
                            f">>> [警告] 反馈修复第 {retry}/{max_adjust_retries} 次失败: {last_reason}"
                        )
                        continue

                    valid, reason = self._validate_adjusted_context_with_screenshots(
                        adjusted_context,
                        context,
                        parts,
                        screenshots,
                    )
                    if not valid:
                        last_reason = reason
                        print(
                            f">>> [警告] 反馈修复第 {retry}/{max_adjust_retries} 次未通过题目划分校验: {reason}"
                        )
                        continue

                    context = self._guard_context_update(context, adjusted_context)
                    adjusted_ok = True
                    break

                if not adjusted_ok:
                    print(
                        f">>> [警告] 反馈修复失败，已保留原内容。最后原因: {last_reason or '未知原因'}"
                    )
                    continue

                output_file = self.generate_docx(homework_name, context)
                print(f"\n[成功] 已根据反馈重新生成作业: {output_file}")

        # 运行期间新增的 web 资料在此统一重建一次索引，避免运行中重复全量 embedding。
        rebuilt = self.reference_index.finalize_pending_updates()
        if rebuilt:
            print(">>> [知识库] 已完成结束阶段统一重建")

        output_file = self.generate_docx(homework_name, context)
        print(f"\n[成功] 作业已生成: {output_file}")


if __name__ == "__main__":
    import argparse
    import sys

    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
    if not pdf_files:
        print("未找到 PDF 文件。")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="自动化解析并生成作业文档")
    parser.add_argument(
        "target_pdf",
        nargs="?",
        default=pdf_files[0],
        help="待处理作业 PDF 路径（默认使用当前目录首个 PDF）",
    )
    parser.add_argument(
        "--references-dir",
        default="",
        help="统一参考资料目录（默认使用 config.toml 中 retrieval.references_dir）",
    )
    args = parser.parse_args()

    target_pdf = args.target_pdf

    print("\n>>> 命令行参数解析结果:")
    print(f">>> target_pdf: {target_pdf}")
    if args.references_dir:
        print(f">>> references_dir override: {args.references_dir}")

    try:
        automator = HomeworkAutomator()
        if args.references_dir:
            automator.reference_index.references_dir = Path(args.references_dir)
            automator.reference_index.web_dir = (
                automator.reference_index.references_dir
                / automator.reference_index.web_dir_name
            )
            automator.reference_index.references_dir.mkdir(parents=True, exist_ok=True)
            automator.reference_index.web_dir.mkdir(parents=True, exist_ok=True)
        automator.run(target_pdf)
    except Exception as e:
        print(f"\n[错误] 运行失败: {e}")
