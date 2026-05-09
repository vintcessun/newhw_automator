import base64
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import fitz
import requests
from pydantic import Field, PrivateAttr

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from unstructured.partition.auto import partition


class LlamaCppMultimodalEmbedding(BaseEmbedding):
    """通过 llama.cpp /v1/embeddings（OpenAI 兼容接口）获取向量。
    文本与图片均走同一端点：
    - 文本：input = "{text_instruct}{text}"
    - 图片文档：input 为视觉占位符，图像数据通过 image_data 侧载传入
    """

    embed_base_url: str = Field(default="http://localhost:8080")
    embed_api_key: str = Field(default="sk-no-key")
    embed_model_name: str = Field(default="")
    request_timeout: int = Field(default=120)
    text_instruct: str = Field(default="Represent this document for retrieval: ")
    image_to_text_fn: Optional[Callable[[str], str]] = Field(default=None)
    image_input_token: str = Field(
        default="<|vision_start|><|image_pad|><|vision_end|>"
    )
    embedding_cache_path: str = Field(default=".homework_cache/embedding_cache.json")
    image_desc_cache_path: str = Field(
        default=".image_desc_cache/image_desc_cache.json"
    )
    embedding_workers: int = Field(default=1)
    verbose: bool = Field(default=True)

    _embedding_cache: Dict[str, List[float]] = PrivateAttr(default_factory=dict)
    _image_desc_cache: Dict[str, str] = PrivateAttr(default_factory=dict)
    _cache_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _embedding_cache_dirty: bool = PrivateAttr(default=False)
    _image_desc_cache_dirty: bool = PrivateAttr(default=False)
    _embed_done: int = PrivateAttr(default=0)
    _embed_total: int = PrivateAttr(default=0)
    _progress_log: bool = PrivateAttr(default=False)
    _embed_start_time: float = PrivateAttr(default=0.0)
    _progress_last_print: float = PrivateAttr(default=0.0)
    _progress_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._load_caches()

    def _load_caches(self) -> None:
        emb_raw = self._load_json_dict(self.embedding_cache_path)
        img_raw = self._load_json_dict(self.image_desc_cache_path)

        emb_cache: Dict[str, List[float]] = {}
        for k, v in emb_raw.items():
            if isinstance(v, list) and v:
                try:
                    emb_cache[str(k)] = [float(x) for x in v]
                except Exception:
                    continue

        img_cache: Dict[str, str] = {}
        for k, v in img_raw.items():
            if isinstance(v, str) and v.strip():
                img_cache[str(k)] = v

        self._embedding_cache = emb_cache
        self._image_desc_cache = img_cache

    def _load_json_dict(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            return {}
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return {}
            return {str(k): v for k, v in raw.items()}
        except Exception:
            return {}

    def _atomic_save_json_dict(self, path: str, payload: Dict[str, Any]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=str(p.parent)
        ) as f:
            json.dump(payload, f, ensure_ascii=False)
            temp_name = f.name
        os.replace(temp_name, p)

    def _cache_key(self, text_input: str) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.embed_model_name.encode("utf-8"))
        hasher.update(b"\n")
        hasher.update(text_input.encode("utf-8", errors="ignore"))
        return hasher.hexdigest()

    def _image_desc_key(self, image_path: str) -> str:
        p = Path(image_path)
        try:
            st = p.stat()
            base = f"{p.as_posix()}|{st.st_mtime_ns}|{st.st_size}"
        except Exception:
            base = p.as_posix()
        return hashlib.sha256(base.encode("utf-8", errors="ignore")).hexdigest()

    def _flush_caches(self) -> None:
        with self._cache_lock:
            if self._embedding_cache_dirty:
                self._atomic_save_json_dict(
                    self.embedding_cache_path, self._embedding_cache
                )
                self._embedding_cache_dirty = False
            if self._image_desc_cache_dirty:
                self._atomic_save_json_dict(
                    self.image_desc_cache_path, self._image_desc_cache
                )
                self._image_desc_cache_dirty = False

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _request_embedding(
        self, payload: Dict[str, Any], label: str, image_b64_len: int = 0
    ) -> List[float]:
        """POST /v1/embeddings 并返回向量，带进度输出。"""
        headers = {
            "Authorization": f"Bearer {self.embed_api_key}",
            "Content-Type": "application/json",
        }
        # <=0 表示无限等待
        timeout: Optional[float] = (
            None if self.request_timeout <= 0 else float(self.request_timeout)
        )

        request_start = time.perf_counter()
        stop_event = threading.Event()

        def _progress() -> None:
            while not stop_event.wait(0.5):
                print(
                    f"\r  [Embedding/{label}] 等待响应 {time.perf_counter() - request_start:.1f}s ...",
                    end="",
                    flush=True,
                )

        t = threading.Thread(target=_progress, daemon=True)
        if self.verbose:
            t.start()
        try:
            resp = requests.post(
                f"{self.embed_base_url}/v1/embeddings",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        finally:
            if self.verbose:
                stop_event.set()
                t.join(timeout=1.0)

        elapsed = time.perf_counter() - request_start
        resp.raise_for_status()
        raw = resp.content
        data = resp.json()
        raw_embedding = data["data"][0]["embedding"]
        if not isinstance(raw_embedding, list):
            raise ValueError("embedding response malformed: embedding is not a list")
        embedding = [float(x) for x in raw_embedding]
        vector_dim = len(embedding)

        extra_info = ""
        if image_b64_len > 0:
            extra_info = f" | 图片base64 {image_b64_len} chars"

        if self.verbose:
            print(
                f"\r  [Embedding/{label}] 完成 | 耗时 {elapsed:.2f}s | 响应 {len(raw)} B | 向量维度 {vector_dim}{extra_info}"
            )
        return embedding

    def _embed_text(self, text: str) -> List[float]:
        """纯文本向量化。"""
        text_input = f"{self.text_instruct}{text}"
        cache_key = self._cache_key(text_input)
        with self._cache_lock:
            cached = self._embedding_cache.get(cache_key)
        if isinstance(cached, list) and cached:
            if self.verbose:
                print(f"  [Embedding/文本] 命中缓存 key={cache_key[:10]}...")
            return cached

        payload: Dict[str, Any] = {
            "model": self.embed_model_name,
            "input": text_input,
        }
        vector = self._request_embedding(payload, label="文本")
        with self._cache_lock:
            self._embedding_cache[cache_key] = vector
            self._embedding_cache_dirty = True
        self._flush_caches()
        return vector

    def _embed_image(self, image_path: str, context_text: str = "") -> List[float]:
        """图片先转文字，再走文本 embedding。"""
        description = ""
        image_key = self._image_desc_key(image_path)
        with self._cache_lock:
            cached_desc = self._image_desc_cache.get(image_key, "")
        if cached_desc:
            description = cached_desc
            if self.verbose:
                print(f">>> [Embedding/图片转文字] 命中缓存 key={image_key[:10]}...")
        elif callable(self.image_to_text_fn):
            start = time.perf_counter()
            try:
                description = (self.image_to_text_fn(image_path) or "").strip()
            except Exception as exc:
                if self.verbose:
                    print(f">>> [Embedding/图片转文字] 调用失败: {exc}")
                description = ""
            elapsed = time.perf_counter() - start
            if self.verbose:
                print(
                    f">>> [Embedding/图片转文字] 完成 | 耗时 {elapsed:.2f}s | 描述长度 {len(description)}"
                )
            if description:
                with self._cache_lock:
                    self._image_desc_cache[image_key] = description
                    self._image_desc_cache_dirty = True
                self._flush_caches()

        # 优先使用图片描述；若失败则退回页面文本，避免索引中断
        if description:
            text_for_embedding = f"[图片描述]\n{description}"
        elif context_text.strip():
            text_for_embedding = context_text
        else:
            text_for_embedding = f"[图片文件] {os.path.basename(image_path)}"

        return self._embed_text(text_for_embedding)

    def _get_text_embedding(self, text: str) -> List[float]:
        """LlamaIndex 回调：文档向量。若含 PDF 页面图片则优先图片向量化。"""
        if "[PDF Page Snapshot]" in text:
            m = re.search(r"image_path:\s*(\S+)", text)
            if m:
                img_path = m.group(1)
                if os.path.exists(img_path):
                    try:
                        result = self._embed_image(img_path, context_text=text)
                        with self._progress_lock:
                            self._embed_done += 1
                        return result
                    except Exception as exc:
                        print(f">>> [Embedding] 图片向量化失败，降级为文本: {exc}")
        result = self._embed_text(text)
        with self._progress_lock:
            self._embed_done += 1
        return result

    def _get_query_embedding(self, query: str) -> List[float]:
        """LlamaIndex 回调：查询向量（纯文本）。"""
        return self._embed_text(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量文档向量化：支持并发提升吞吐。"""
        if not texts:
            return []

        workers = max(1, int(self.embedding_workers or 1))
        if workers <= 1 or len(texts) <= 1:
            return [self._get_text_embedding(t) for t in texts]

        results: List[Optional[List[float]]] = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._get_text_embedding, text): idx
                for idx, text in enumerate(texts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        # 到这里 results 不应有 None，保留兜底避免类型问题。
        return [r if isinstance(r, list) else [] for r in results]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def set_verbose(self, verbose: bool) -> None:
        self.verbose = bool(verbose)

    def reset_embed_counter(self, total: int) -> None:
        self._embed_done = 0
        self._embed_total = total
        self._embed_start_time = time.perf_counter()

    def get_embed_progress(self) -> tuple:
        return (self._embed_done, self._embed_total)

    def prewarm_image_description(self, image_path: str) -> bool:
        """预生成图片描述并写入缓存，返回是否发生实际转写。
        返回 False 表示缓存已命中或无可用转写函数/失败。
        """
        if not image_path or not os.path.exists(image_path):
            return False
        image_key = self._image_desc_key(image_path)
        with self._cache_lock:
            if self._image_desc_cache.get(image_key):
                return False
        if not callable(self.image_to_text_fn):
            return False
        try:
            description = (self.image_to_text_fn(image_path) or "").strip()
        except Exception as exc:
            print(f">>> [Embedding/图片预热] 调用失败 {image_path}: {exc}")
            return False
        if not description:
            return False
        with self._cache_lock:
            self._image_desc_cache[image_key] = description
            self._image_desc_cache_dirty = True
        self._flush_caches()
        return True


class ReferenceIndexManager:
    SUPPORTED_SUFFIX = {
        ".md",
        ".markdown",
        ".txt",
        ".pdf",
        ".docx",
        ".doc",
        ".rst",
        ".html",
        ".htm",
    }

    def __init__(
        self,
        retrieval_cfg: Dict[str, Any],
        image_to_text_fn: Optional[Callable[[str], str]] = None,
        low_quality_chunk_from_source_fn: Optional[
            Callable[[str, Dict[str, Any]], str]
        ] = None,
    ):
        self.references_dir = Path(retrieval_cfg.get("references_dir", "references"))
        self.web_dir_name = str(retrieval_cfg.get("web_dir", "web"))
        self.web_dir = Path(self.web_dir_name)
        self.qdrant_path = str(retrieval_cfg.get("qdrant_path", ".qdrant"))
        self.collection_name = str(
            retrieval_cfg.get("collection_name", "newhw_reference_knowledge")
        )
        self.embedding_model = str(retrieval_cfg.get("embedding_model", ""))
        self.embedding_base_url = str(
            retrieval_cfg.get("embedding_base_url", "http://localhost:8080")
        )
        self.embedding_api_key = str(
            retrieval_cfg.get("embedding_api_key", "sk-no-key")
        )
        self.embedding_timeout = int(retrieval_cfg.get("embedding_timeout", 120))
        self.embedding_workers = int(retrieval_cfg.get("embedding_workers", 6))
        self.image_to_text_workers = int(retrieval_cfg.get("image_to_text_workers", 8))
        self.image_to_text_fn = image_to_text_fn
        self.low_quality_chunk_from_source_fn = low_quality_chunk_from_source_fn
        self.embedding_text_instruct = str(
            retrieval_cfg.get(
                "embedding_text_instruct", "Represent this document for retrieval: "
            )
        )
        self.embedding_image_input_token = str(
            retrieval_cfg.get(
                "embedding_image_input_token",
                "<|vision_start|><|image_pad|><|vision_end|>",
            )
        )
        self.chunk_size = int(retrieval_cfg.get("chunk_size", 900))
        self.chunk_overlap = int(retrieval_cfg.get("chunk_overlap", 120))
        self.min_chunk_chars = int(retrieval_cfg.get("min_chunk_chars", 80))
        self.merge_target_chars = int(
            retrieval_cfg.get("merge_target_chars", max(1200, self.chunk_size * 2))
        )
        self.max_chunk_chars = int(
            retrieval_cfg.get("max_chunk_chars", max(2200, self.chunk_size * 3))
        )
        self.repair_low_quality_with_llm = bool(
            retrieval_cfg.get("repair_low_quality_with_llm", True)
        )
        self.low_quality_repair_min_chars = int(
            retrieval_cfg.get("low_quality_repair_min_chars", 40)
        )
        self.low_quality_repair_max_chars = int(
            retrieval_cfg.get("low_quality_repair_max_chars", 3200)
        )
        self.local_top_k = int(retrieval_cfg.get("local_top_k", 6))

        self.cache_dir = Path(".homework_cache")
        self.state_path = self.cache_dir / "reference_index_state.json"
        self.parse_cache_dir = self.cache_dir / "ref_parse"
        self.chunk_cache_dir = self.cache_dir / "ref_chunks"
        self.image_desc_cache_dir = Path(".image_desc_cache")
        self.page_image_dir = Path("problems") / "_reference_mm"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.parse_cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_desc_cache_dir.mkdir(parents=True, exist_ok=True)
        self.references_dir.mkdir(parents=True, exist_ok=True)
        self.web_dir.mkdir(parents=True, exist_ok=True)
        self.page_image_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_ready = False
        Settings.text_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        self.qdrant_client = QdrantClient(path=self.qdrant_path)
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
        )
        self.index: Optional[VectorStoreIndex] = None
        self.pending_rebuild = False

        # 某些 PDF 结构树损坏会触发 MuPDF 底层错误输出，关闭显示避免污染日志。
        try:
            fitz.TOOLS.mupdf_display_errors(False)
            fitz.TOOLS.mupdf_display_warnings(False)
        except Exception:
            pass

    def ensure_up_to_date(self, force_rebuild: bool = False) -> bool:
        self._ensure_embedding_model()
        print(">>> [知识库] 正在扫描 references 目录文件签名...")
        signatures = self._scan_signatures()
        print(f">>> [知识库] 扫描完成，共 {len(signatures)} 个文件")
        state = self._load_state()
        collection_exists = self._collection_exists()
        has_changed = (
            force_rebuild
            or state.get("signatures") != signatures
            or not collection_exists
        )
        if has_changed:
            print(">>> [知识库] 检测到变更，开始重建索引...")
            self._rebuild_full_index(signatures)
            print(">>> [知识库] 索引重建完成")
        else:
            print(">>> [知识库] 索引已是最新，加载现有索引...")
            self._load_index()
            print(">>> [知识库] 索引加载完成")
        return has_changed

    def _collection_exists(self) -> bool:
        try:
            return bool(self.qdrant_client.collection_exists(self.collection_name))
        except Exception:
            return False

    def query(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        self._ensure_embedding_model()
        if self.index is None:
            self._load_index()
        if self.index is None:
            return []

        retriever = self.index.as_retriever(similarity_top_k=top_k or self.local_top_k)
        nodes = retriever.retrieve(query)
        results: List[Dict[str, Any]] = []
        for node in nodes:
            score = getattr(node, "score", None)
            text = node.node.get_content() if getattr(node, "node", None) else ""
            metadata = dict(getattr(node.node, "metadata", {}) or {})
            results.append(
                {
                    "score": float(score) if score is not None else None,
                    "text": text,
                    "metadata": metadata,
                }
            )
        return results

    def ingest_web_records(
        self, query: str, records: List[Dict[str, Any]]
    ) -> Optional[str]:
        if not records:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]", "_", query)[:48]
        file_name = f"web_{ts}_{safe_query or 'query'}.md"
        target = self.web_dir / file_name

        lines = [f"# Web Knowledge: {query}", "", f"- created_at: {ts}", ""]
        for idx, item in enumerate(records, start=1):
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            content = str(item.get("content", ""))
            relevance = str(item.get("relevance", ""))
            lines.extend(
                [
                    f"## Item {idx}",
                    f"- title: {title}",
                    f"- url: {url}",
                    f"- relevance: {relevance}",
                    "",
                    content,
                    "",
                ]
            )

        target.write_text("\n".join(lines), encoding="utf-8")
        # 运行中仅落盘，延迟到流程结束时统一重建，避免反复全量 embedding。
        self.pending_rebuild = True
        return str(target)

    def finalize_pending_updates(self) -> bool:
        """在流程结束时执行一次统一重建。"""
        if not self.pending_rebuild:
            return False
        print(">>> [知识库] 检测到运行期新增 web 资料，正在结束阶段统一重建索引...")
        self.ensure_up_to_date(force_rebuild=False)
        self.pending_rebuild = False
        return True

    def _ensure_embedding_model(self) -> None:
        if self._embedding_ready:
            return
        try:
            embed_model = LlamaCppMultimodalEmbedding(
                embed_base_url=self.embedding_base_url,
                embed_api_key=self.embedding_api_key,
                embed_model_name=self.embedding_model,
                request_timeout=self.embedding_timeout,
                embedding_workers=self.embedding_workers,
                image_to_text_fn=self.image_to_text_fn,
                text_instruct=self.embedding_text_instruct,
                image_input_token=self.embedding_image_input_token,
                embedding_cache_path=str(self.cache_dir / "embedding_cache.json"),
                image_desc_cache_path=str(
                    self.image_desc_cache_dir / "image_desc_cache.json"
                ),
            )
            Settings.embed_model = embed_model
            self._embedding_ready = True
            print(
                f">>> [Embedding] 已初始化 LlamaCppMultimodalEmbedding: {self.embedding_base_url}"
            )
        except Exception as exc:
            from llama_index.core.embeddings import MockEmbedding

            print(
                f">>> [警告] LlamaCppMultimodalEmbedding 初始化失败，已降级为 MockEmbedding。原因: {exc}"
            )
            Settings.embed_model = MockEmbedding(embed_dim=1024)
            self._embedding_ready = True

    def _load_index(self) -> None:
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
        )
        try:
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        except Exception:
            self.index = None

    def _rebuild_full_index(self, signatures: Dict[str, str]) -> None:
        rebuild_start = time.perf_counter()

        def _progress(ratio: float, stage: str) -> None:
            ratio = max(0.0, min(1.0, ratio))
            bar_len = 28
            filled = int(bar_len * ratio)
            bar = "#" * filled + "-" * (bar_len - filled)
            line = f"\r>>> [知识库重建] [{bar}] {ratio * 100:6.2f}% | {stage}"
            print(line, end="", flush=True)

        _progress(0.0, "准备中")
        all_files = self._scan_reference_files()
        total_files = len(all_files)
        nodes: List[TextNode] = []
        doc_count = 0
        parse_hits = 0
        chunk_hits = 0
        _progress(0.05, f"解析/切块 0/{total_files}")
        for i, file_path in enumerate(all_files, start=1):
            rel_path = file_path.as_posix()
            try:
                content_hash = self._content_hash(file_path)
            except Exception as exc:
                print(f">>> [知识库] 读取失败，跳过: {rel_path}，原因: {exc}")
                continue

            cached_nodes = self._load_chunk_cache(content_hash)
            if cached_nodes is not None:
                nodes.extend(cached_nodes)
                chunk_hits += 1
                # parse 层未必命中，但既然块缓存有效就直接用，doc_count 无法准确反推
                _progress(
                    0.05 + 0.55 * (i / total_files),
                    f"解析/切块 {i}/{total_files} (块缓存命中)",
                )
                continue

            docs = self._load_parse_cache(content_hash)
            if docs is not None:
                parse_hits += 1
            else:
                docs = self._build_documents_for_file(file_path)
                self._save_parse_cache(content_hash, rel_path, docs)

            doc_count += len(docs)
            file_nodes = self._build_nodes_for_documents(docs)
            self._save_chunk_cache(content_hash, rel_path, file_nodes)
            nodes.extend(file_nodes)
            _progress(
                0.05 + 0.55 * (i / total_files),
                f"解析/切块 {i}/{total_files}",
            )

        total_nodes = len(nodes)
        _progress(0.62, f"准备向量化 {total_nodes} 块")
        print()  # 换行，向量化进度条独占一行
        print(
            f">>> [知识库] 解析缓存命中 {parse_hits}/{total_files}，"
            f"切块缓存命中 {chunk_hits}/{total_files}"
        )

        if self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.delete_collection(self.collection_name)

        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
        )

        # 重置计数器
        embed_model = getattr(Settings, "embed_model", None)
        if isinstance(embed_model, LlamaCppMultimodalEmbedding):
            embed_model.reset_embed_counter(total_nodes)

        # 先一次性把 PDF 页面快照的图片统一转文字并写入缓存，
        # 避免后续 embedding 阶段频繁在 VLM 与 embedding 之间来回切换。
        if isinstance(embed_model, LlamaCppMultimodalEmbedding):
            image_paths: List[str] = []
            seen_imgs: set[str] = set()
            for n in nodes:
                text = n.get_content() if hasattr(n, "get_content") else ""
                if "[PDF Page Snapshot]" not in text:
                    continue
                m = re.search(r"image_path:\s*(\S+)", text)
                if not m:
                    continue
                p = m.group(1)
                if p and p not in seen_imgs and os.path.exists(p):
                    seen_imgs.add(p)
                    image_paths.append(p)

            total_imgs = len(image_paths)
            if total_imgs > 0:
                self._set_embedding_verbose(False)
                prewarm_start = time.perf_counter()
                prewarm_stop = threading.Event()
                state = {"idx": 0, "generated": 0, "cached": 0}
                state_lock = threading.Lock()
                bar_len = 28

                def _prewarm_bar() -> None:
                    while not prewarm_stop.wait(0.5):
                        with state_lock:
                            idx = state["idx"]
                            gen = state["generated"]
                            hit = state["cached"]
                        ratio = idx / total_imgs if total_imgs else 0.0
                        filled = int(bar_len * ratio)
                        bar = "#" * filled + "-" * (bar_len - filled)
                        elapsed = time.perf_counter() - prewarm_start
                        print(
                            f"\r  [图片预热] [{bar}] {idx}/{total_imgs} | 新生成 {gen} 缓存命中 {hit} | {elapsed:.1f}s   ",
                            end="",
                            flush=True,
                        )

                prewarm_hb = threading.Thread(target=_prewarm_bar, daemon=True)
                prewarm_hb.start()

                def _prewarm_one(img_path: str) -> None:
                    did_generate = embed_model.prewarm_image_description(img_path)
                    with state_lock:
                        state["idx"] += 1
                        if did_generate:
                            state["generated"] += 1
                        else:
                            state["cached"] += 1

                try:
                    with ThreadPoolExecutor(
                        max_workers=self.image_to_text_workers
                    ) as _pool:
                        list(_pool.map(_prewarm_one, image_paths))
                finally:
                    prewarm_stop.set()
                    prewarm_hb.join(timeout=1.0)
                    elapsed = time.perf_counter() - prewarm_start
                    with state_lock:
                        gen = state["generated"]
                        hit = state["cached"]
                    print(
                        f"\r  [图片预热] [{'#' * bar_len}] {total_imgs}/{total_imgs} | 新生成 {gen} 缓存命中 {hit} | 完成 | {elapsed:.2f}s   ",
                        flush=True,
                    )
                    print()

        self._set_embedding_verbose(False)
        embed_start = time.perf_counter()
        stop_event = threading.Event()

        def _embed_bar() -> None:  # noqa: E306
            bar_len = 28
            while not stop_event.wait(0.5):
                _done = (
                    embed_model.get_embed_progress()[0]
                    if isinstance(embed_model, LlamaCppMultimodalEmbedding)
                    else 0
                )
                elapsed = time.perf_counter() - embed_start
                ratio = (_done / total_nodes) if total_nodes else 0.0
                filled = int(bar_len * ratio)
                bar = "#" * filled + "-" * (bar_len - filled)
                # 用尾部空格填充清除上次更长的行，不换行
                print(
                    f"\r  [向量化] [{bar}] {_done}/{total_nodes} 块 | {elapsed:.1f}s   ",
                    end="",
                    flush=True,
                )

        heartbeat = threading.Thread(target=_embed_bar, daemon=True)
        heartbeat.start()
        try:
            self.index = VectorStoreIndex(
                nodes,
                vector_store=self.vector_store,
                show_progress=False,
            )
        finally:
            stop_event.set()
            heartbeat.join(timeout=1.0)
            self._set_embedding_verbose(True)
            _done = (
                embed_model.get_embed_progress()[0]
                if isinstance(embed_model, LlamaCppMultimodalEmbedding)
                else total_nodes
            )
            elapsed = time.perf_counter() - embed_start
            print(
                f"\r  [向量化] [{'#' * 28}] {_done}/{total_nodes} 块 | 完成 | {elapsed:.2f}s   ",
                flush=True,
            )
            print()

        _progress(0.95, "保存状态")
        self._save_state(
            {
                "updated_at": int(time.time()),
                "signatures": signatures,
                "doc_count": doc_count,
                "node_count": total_nodes,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
            }
        )
        _progress(1.0, "完成")
        print()
        total_elapsed = time.perf_counter() - rebuild_start
        print(
            f">>> [知识库] 重建完成 | 文件 {total_files} | 文档块 {total_nodes} | 总耗时 {total_elapsed:.2f}s"
        )

    def _set_embedding_verbose(self, verbose: bool) -> None:
        model = getattr(Settings, "embed_model", None)
        if isinstance(model, LlamaCppMultimodalEmbedding):
            model.set_verbose(verbose)

    def _build_documents_for_file(self, file_path: Path) -> List[Document]:
        suffix = file_path.suffix.lower()
        rel_path = str(file_path.as_posix())
        if suffix == ".pdf":
            # PDF 仅按页建块，不再追加更细粒度文本切块。
            return self._build_pdf_page_documents(file_path, rel_path)

        docs: List[Document] = []

        chunks = self._partition_text_chunks(file_path)
        for idx, chunk in enumerate(chunks, start=1):
            text = chunk.strip()
            if not text:
                continue
            docs.append(
                Document(
                    text=text,
                    metadata={
                        "source_path": rel_path,
                        "chunk_id": idx,
                        "content_type": "text_chunk",
                    },
                )
            )
        return docs

    def _build_nodes_for_documents(self, docs: List[Document]) -> List[TextNode]:
        nodes: List[TextNode] = []
        for doc in docs:
            metadata = dict(getattr(doc, "metadata", {}) or {})
            content_type = str(metadata.get("content_type", "") or "")
            # 页级 PDF 文档保持 1 页 1 块，避免被 SentenceSplitter 二次切分。
            if content_type == "page_image":
                text = str(getattr(doc, "text", "") or "").strip()
                if text:
                    nodes.append(TextNode(text=text, metadata=metadata))
                continue
            nodes.extend(Settings.text_splitter.get_nodes_from_documents([doc]))
        return nodes

    def _build_pdf_page_documents(
        self, file_path: Path, rel_path: str
    ) -> List[Document]:
        docs: List[Document] = []
        skipped_pages = 0
        try:
            pdf = fitz.open(str(file_path))
        except Exception:
            return docs

        try:
            for page_index in range(len(pdf)):
                try:
                    page = pdf[page_index]
                except Exception as exc:
                    skipped_pages += 1
                    print(
                        f">>> [参考索引] 跳过损坏页: {rel_path} 第{page_index + 1}页，原因: {exc}"
                    )
                    continue

                try:
                    page_text = page.get_text("text")
                except Exception as exc:
                    skipped_pages += 1
                    print(
                        f">>> [参考索引] 提取文本失败: {rel_path} 第{page_index + 1}页，原因: {exc}"
                    )
                    page_text = ""

                image_name = f"{file_path.stem}_p{page_index + 1}.png"
                image_path = self.page_image_dir / image_name
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
                    pix.save(str(image_path))
                    image_value = image_path.as_posix()
                except Exception as exc:
                    print(
                        f">>> [参考索引] 页面截图失败: {rel_path} 第{page_index + 1}页，原因: {exc}"
                    )
                    image_value = ""

                docs.append(
                    Document(
                        text=(
                            f"[PDF Page Snapshot]\n"
                            f"source: {rel_path}\n"
                            f"page: {page_index + 1}\n"
                            f"image_path: {image_value}\n"
                            f"content:\n{page_text}"
                        ),
                        metadata={
                            "source_path": rel_path,
                            "page": page_index + 1,
                            "image_path": image_value,
                            "content_type": "page_image",
                        },
                    )
                )
        finally:
            pdf.close()

        if skipped_pages > 0:
            print(
                f">>> [参考索引] 文件 {rel_path} 共跳过 {skipped_pages} 页（其余页已继续处理）"
            )

        return docs

    def _partition_text_chunks(self, file_path: Path) -> List[str]:
        suffix = file_path.suffix.lower()

        # Markdown / 纯文本：按标题切分，不经过 unstructured
        if suffix in {".md", ".markdown", ".txt", ".rst"}:
            return self._split_text_by_heading(file_path)

        try:
            elements = partition(filename=str(file_path))
            chunks = [str(e).strip() for e in elements if str(e).strip()]
            if chunks:
                return self._coalesce_text_chunks(chunks, file_path=file_path)
        except Exception:
            pass

        try:
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
            return self._coalesce_text_chunks([raw], file_path=file_path)
        except Exception:
            return []

    def _normalize_text(self, text: str) -> str:
        # 删除大部分控制字符，保留换行和制表，避免把乱码噪声送入向量库。
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _is_low_quality_chunk(self, text: str) -> bool:
        t = self._normalize_text(text)
        if not t:
            return True
        if len(t) < max(10, self.min_chunk_chars // 4):
            return True

        printable = sum(ch.isprintable() or ch in "\n\t" for ch in t)
        printable_ratio = printable / max(1, len(t))
        if printable_ratio < 0.85:
            return True

        noise = sum(
            1
            for ch in t
            if ord(ch) < 32
            or (
                not ch.isalnum()
                and not ch.isspace()
                and not ("\u4e00" <= ch <= "\u9fff")
            )
        )
        noise_ratio = noise / max(1, len(t))
        if noise_ratio > 0.45:
            return True

        return False

    def _coalesce_text_chunks(self, chunks: List[str], file_path: Path) -> List[str]:
        merged: List[str] = []
        buffer = ""
        total_chunks = len(chunks)

        def flush() -> None:
            nonlocal buffer
            if not buffer.strip():
                buffer = ""
                return
            merged.append(buffer)
            buffer = ""

        for idx, raw in enumerate(chunks, start=1):
            text = str(raw)
            if not text.strip():
                continue

            if not buffer:
                buffer = text
            elif len(buffer) + 2 + len(text) <= self.max_chunk_chars:
                buffer = f"{buffer}\n\n{text}"
            else:
                flush()
                buffer = text

            if len(buffer) >= self.merge_target_chars:
                flush()

        flush()
        return merged

    def _rewrite_low_quality_chunk_from_source(
        self,
        file_path: Path,
        raw_chunk: str,
        chunk_index: int,
        total_chunks: int,
    ) -> str:
        return ""

    def _split_text_by_heading(self, file_path: Path) -> List[str]:
        """按 Markdown 标题（# / ## / ...）将文件切分为语义块。
        同一标题下的内容若超过 chunk_size 字符则进一步切成多个子块。
        """
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        if not text.strip():
            return []

        # 按标题行拆分，保留标题行本身
        heading_pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
        parts = heading_pattern.split(text)

        # parts 结构：[前导内容, 标题1, 内容1, 标题2, 内容2, ...]
        sections: List[str] = []
        # 前导内容（无标题区域）
        if parts[0].strip():
            sections.append(parts[0].strip())

        i = 1
        while i < len(parts) - 1:
            heading = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            section = f"{heading}\n{body}".strip() if body else heading
            if section:
                sections.append(section)
            i += 2

        if not sections:
            # 无标题，整体作为一块
            return [text.strip()]

        # 超长块按 chunk_size 再拆
        result: List[str] = []
        for sec in sections:
            if len(sec) <= self.chunk_size:
                result.append(sec)
            else:
                # 按换行符切成段落，再聚合
                lines = sec.splitlines(keepends=True)
                buf = ""
                for line in lines:
                    if len(buf) + len(line) > self.chunk_size and buf.strip():
                        result.append(buf.strip())
                        buf = line
                    else:
                        buf += line
                if buf.strip():
                    result.append(buf.strip())

        return result

    def _fallback_pdf_text(self, file_path: Path) -> List[str]:
        chunks: List[str] = []
        try:
            pdf = fitz.open(str(file_path))
            for i in range(len(pdf)):
                try:
                    text = pdf[i].get_text("text").strip()
                except Exception:
                    text = ""
                if text:
                    chunks.append(text)
            pdf.close()
        except Exception:
            return []
        return chunks

    def _scan_reference_files(self) -> List[Path]:
        files: List[Path] = []
        if not self.references_dir.exists():
            return files
        for path in self.references_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.SUPPORTED_SUFFIX:
                continue
            files.append(path)
        return sorted(files)

    def _scan_signatures(self) -> Dict[str, str]:
        signatures: Dict[str, str] = {}
        for path in self._scan_reference_files():
            signatures[path.as_posix()] = self._file_signature(path)
        return signatures

    def _file_signature(self, file_path: Path) -> str:
        stat = file_path.stat()
        hasher = hashlib.sha256()
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        with file_path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    def _content_hash(self, file_path: Path) -> str:
        """按文件内容计算 sha256，与 mtime 无关，供每文件缓存键使用。"""
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    def _parse_cache_path(self, content_hash: str) -> Path:
        return self.parse_cache_dir / f"{content_hash}.json"

    def _chunk_cache_path(self, content_hash: str) -> Path:
        strategy_sig = (
            f"v3_pdf_page_only_{self.chunk_size}_{self.chunk_overlap}_"
            f"{self.min_chunk_chars}_{self.merge_target_chars}_{self.max_chunk_chars}_"
            f"{int(self.repair_low_quality_with_llm)}"
        )
        return self.chunk_cache_dir / f"{content_hash}_{strategy_sig}.json"

    def _load_parse_cache(self, content_hash: str) -> Optional[List[Document]]:
        path = self._parse_cache_path(content_hash)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            # v2 起仅接受新解析缓存，避免复用历史细粒度 PDF 切块结果。
            if int(payload.get("version", 0)) != 2:
                return None
            raw_docs = payload.get("docs")
            if not isinstance(raw_docs, list):
                return None
            docs: List[Document] = []
            for item in raw_docs:
                if not isinstance(item, dict):
                    return None
                text = item.get("text", "")
                metadata = item.get("metadata") or {}
                if not isinstance(metadata, dict):
                    metadata = {}
                # PDF 页面快照依赖磁盘图片，缺失则整体失效以重建
                img_path = metadata.get("image_path")
                if (
                    metadata.get("content_type") == "page_image"
                    and isinstance(img_path, str)
                    and img_path
                    and not os.path.exists(img_path)
                ):
                    return None
                docs.append(Document(text=str(text), metadata=metadata))
            return docs
        except Exception:
            return None

    def _save_parse_cache(
        self, content_hash: str, rel_path: str, docs: List[Document]
    ) -> None:
        path = self._parse_cache_path(content_hash)
        payload = {
            "version": 2,
            "rel_path": rel_path,
            "docs": [
                {
                    "text": d.get_content() if hasattr(d, "get_content") else str(d),
                    "metadata": dict(getattr(d, "metadata", {}) or {}),
                }
                for d in docs
            ],
        }
        try:
            self._atomic_write_json(path, payload)
        except Exception as exc:
            print(f">>> [知识库] 写入解析缓存失败: {path}，原因: {exc}")

    def _load_chunk_cache(self, content_hash: str) -> Optional[List[TextNode]]:
        path = self._chunk_cache_path(content_hash)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            if int(payload.get("chunk_size", -1)) != self.chunk_size:
                return None
            if int(payload.get("chunk_overlap", -1)) != self.chunk_overlap:
                return None
            if int(payload.get("min_chunk_chars", -1)) != self.min_chunk_chars:
                return None
            if int(payload.get("merge_target_chars", -1)) != self.merge_target_chars:
                return None
            if int(payload.get("max_chunk_chars", -1)) != self.max_chunk_chars:
                return None
            if bool(payload.get("repair_low_quality_with_llm", False)) != bool(
                self.repair_low_quality_with_llm
            ):
                return None
            raw_nodes = payload.get("nodes")
            if not isinstance(raw_nodes, list):
                return None
            nodes: List[TextNode] = []
            for item in raw_nodes:
                if not isinstance(item, dict):
                    return None
                text = item.get("text", "")
                metadata = item.get("metadata") or {}
                if not isinstance(metadata, dict):
                    metadata = {}
                img_path = metadata.get("image_path")
                if (
                    metadata.get("content_type") == "page_image"
                    and isinstance(img_path, str)
                    and img_path
                    and not os.path.exists(img_path)
                ):
                    return None
                nodes.append(TextNode(text=str(text), metadata=metadata))
            return nodes
        except Exception:
            return None

    def _save_chunk_cache(
        self, content_hash: str, rel_path: str, nodes: List[TextNode]
    ) -> None:
        path = self._chunk_cache_path(content_hash)
        payload = {
            "version": 1,
            "rel_path": rel_path,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_chars": self.min_chunk_chars,
            "merge_target_chars": self.merge_target_chars,
            "max_chunk_chars": self.max_chunk_chars,
            "repair_low_quality_with_llm": self.repair_low_quality_with_llm,
            "nodes": [
                {
                    "text": n.get_content() if hasattr(n, "get_content") else str(n),
                    "metadata": dict(getattr(n, "metadata", {}) or {}),
                }
                for n in nodes
            ],
        }
        try:
            self._atomic_write_json(path, payload)
        except Exception as exc:
            print(f">>> [知识库] 写入切块缓存失败: {path}，原因: {exc}")

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=str(path.parent)
        ) as f:
            json.dump(payload, f, ensure_ascii=False)
            temp_name = f.name
        os.replace(temp_name, path)

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self, payload: Dict[str, Any]) -> None:
        self.state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
