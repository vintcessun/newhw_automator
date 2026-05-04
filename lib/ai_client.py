"""AI 客户端配置和工具调用管理"""

import json
import time
import threading
import requests
from typing import Any, Dict, List, Optional
from types import SimpleNamespace
from openai import OpenAI, APIConnectionError, APIStatusError, InternalServerError


class AIClient:
    """管理 AI 客户端和工具调用"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup_clients()
        self.tools = self._init_tools()

    def _setup_clients(self):
        """初始化简单模型和复杂模型的 OpenAI 客户端"""
        simple_cfg = self.config["llm"]["simple"]
        self.simple_client = OpenAI(
            api_key=simple_cfg["api_key"], base_url=simple_cfg["base_url"]
        )
        self.simple_model = simple_cfg["model"]

        complex_cfg = self.config["llm"]["complex"]
        self.complex_client = OpenAI(
            api_key=complex_cfg["api_key"], base_url=complex_cfg["base_url"]
        )
        self.complex_model = complex_cfg["model"]

        searxng_cfg = self.config.get("searxng", {})
        self.searxng_base = searxng_cfg.get("base_url", "http://localhost:8089/")
        self.searxng_timeout = int(searxng_cfg.get("timeout_seconds", 3600))

    def _init_tools(self) -> List[Dict[str, Any]]:
        """初始化工具定义"""
        return [
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
                    "name": "search_web",
                    "description": "通过 SearxNG 搜索互联网获取最新的计算机网络协议知识或相关资料。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索关键词，用空格分割",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def execute_python(self, code: str) -> str:
        """执行 Python 代码并返回输出和返回值，限制文件和网络访问"""
        import sys
        from io import StringIO
        import math, re, json, base64, datetime, itertools

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

    def search_searxng(self, query: str) -> str:
        """通过 SearxNG 搜索信息并返回前5条结果"""
        try:
            url = f"{self.searxng_base}search?q={query}&format=json"
            response = requests.get(url, timeout=self.searxng_timeout)
            data = response.json()
            results = data.get("results", [])[:5]
            if not results:
                return "未找到结果"
            summary = "\n".join(
                [f"- {r.get('title')}: {r.get('content')}" for r in results]
            )
            return summary
        except Exception as e:
            return f"搜索失败: {e}"

    def handle_tool_calls(self, tool_calls: Any) -> List[Dict[str, str]]:
        """执行工具调用并返回结果列表"""
        results = []
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except:
                args = {}
            print(f"  [执行工具] {func_name}: {args}")

            if func_name == "python_interpreter":
                result = self.execute_python(args.get("code", ""))
            elif func_name == "search_web":
                result = self.search_searxng(args.get("query", ""))
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
            print(f"  [工具结果] {result[:100]}...")
        return results

    def call_ai(
        self,
        client: OpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        use_tools: bool = True,
        **kwargs: Any,
    ) -> Any:
        """封装 AI 调用，支持工具自动处理及网络错误重试"""
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

                        if first_chunk_at is not None:
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
                    wait_thread.join(timeout=0.5)

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
                tool_results = self.handle_tool_calls(msg.tool_calls)
                current_messages.extend(tool_results)
            else:
                return response
        return response
