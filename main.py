import os
import toml
import json
import re
import base64
import requests
import time
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Optional, cast
from openai import OpenAI, APIConnectionError, InternalServerError
from docxtpl import DocxTemplate
from pypdf import PdfReader
import fitz
import numpy as np


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
        self.searxng_base = self.config.get("searxng", {}).get(
            "base_url", "http://localhost:8089/"
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

    def _search_searxng(self, query: str) -> str:
        """通过 SearxNG 搜索信息并返回前5条结果"""
        try:
            url = f"{self.searxng_base}search?q={query}&format=json"
            response = requests.get(url, timeout=3600)
            data = response.json()
            results = data.get("results", [])[:5]  # 返回 5 条
            if not results:
                return "未找到结果"
            summary = "\n".join(
                [f"- {r.get('title')}: {r.get('content')}" for r in results]
            )
            return summary
        except Exception as e:
            return f"搜索失败: {e}"

    def _handle_tool_calls(self, tool_calls: Any) -> List[Dict[str, str]]:
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
                result = self._execute_python(args.get("code", ""))
            elif func_name == "search_web":
                result = self._search_searxng(args.get("query", ""))
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

    def _call_ai(
        self,
        client: OpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        use_tools: bool = True,
        **kwargs: Any,
    ) -> Any:
        """封装 AI 调用，支持工具自动处理及网络错误重试"""
        current_messages = messages.copy()

        for _ in range(10):  # 最多 10 轮工具交互
            call_params = {
                "model": model,
                "messages": current_messages,
            }
            if use_tools:
                call_params["tools"] = self.tools

            call_params.update(kwargs)

            # 网络重试逻辑
            max_retries = 1000
            last_err = None
            response = None
            for retry in range(max_retries + 1):
                try:
                    response = client.chat.completions.create(**call_params)
                    break
                except (
                    APIConnectionError,
                    InternalServerError,
                    requests.exceptions.RequestException,
                ) as e:
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

            if not response:
                return None
            msg = response.choices[0].message

            if msg.tool_calls:
                current_messages.append(msg)
                tool_results = self._handle_tool_calls(msg.tool_calls)
                current_messages.extend(tool_results)
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

    def parse_pdf(
        self, pdf_path: str
    ) -> Tuple[str, Dict[str, Any], Dict[str, Dict[str, str]]]:
        """解析PDF并利用LLM提取四个部分的内容"""
        print(">>> 正在提取 PDF 文本...")
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        print(">>> 正在使用 AI 解析作业结构...")
        prompt = f"""你是一个作业解析助手。请阅读以下从 PDF 中提取的乱序或复杂的文本，并将其整理为结构化的 JSON 格式。
要求：
1. 提取“作业名称”（通常是标题）。
2. 将内容分为四个部分：
   - "homework_name": 作业名称字符串，不需要计算机网络等前缀，只需要直接写“第x课xxxx”。
   - "choice": 包含完整题目的列表，列表格式，每个元素包含 "id" (题号) 和 "question" (完整题目及选项)。
   - "short_answer": 包含完整题目的列表，列表格式，每个元素包含 "id" (题号) 和 "question" (完整题目)。
   - "programming": 包含完整题目的列表，列表格式，每个元素包含 "id" (题号) 和 "question" (完整要求/链接提示)。
3. 严格输出 JSON 格式，不要任何 Markdown。

文本内容：
{full_text}
"""
        response = self._call_ai(
            self.simple_client,
            self.simple_model,
            [{"role": "user", "content": prompt}],
            use_tools=False,
            response_format={"type": "json_object"},
        )

        if not response or not hasattr(response, "choices"):
            raise ValueError("AI 解析 PDF 结构失败")
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("AI 解析内容为空")

        data = self._parse_json_safe(content)
        homework_name = data.get("homework_name", "未命名作业")
        parts = {
            "choice": data.get("choice", []),
            "short_answer": data.get("short_answer", []),
            "programming": data.get("programming", []),
        }

        print(">>> 正在生成题目截图到 problems 目录...")
        screenshots = self.generate_problem_screenshots(pdf_path, parts)

        print("\n" + "=" * 30)
        print("AI 解析结果：")
        print(f"作业名称: {homework_name}")
        print(
            f"选择题: {len(parts['choice'])} 道, 简答题: {len(parts['short_answer'])} 道, 编程题: {len(parts['programming'])} 道"
        )
        print("题目截图目录: problems")
        print("=" * 30)

        return homework_name, parts, screenshots

    def solve_choice_questions(
        self,
        choices_list: List[Dict[str, Any]],
        image_map: Dict[str, str],
        reference_materials_text: str = "无",
    ) -> List[str]:
        """使用复杂模型解决选择题 (CoT + 每题搜索 + 循环审阅)"""
        if not choices_list:
            return [""] * self.CHOICE_QUESTION_LIMIT

        student_name = self.config["student_info"]["name"]
        final_ans = [""] * self.CHOICE_QUESTION_LIMIT
        pending_choices = choices_list.copy()

        max_rounds = 10
        for round_idx in range(max_rounds):
            if not pending_choices:
                break

            print(
                f"\n>>> 选择题处理第 {round_idx + 1} 轮 (剩余 {len(pending_choices)} 题)..."
            )

            # 为当前轮次的每道题进行搜索背景调查
            current_batch_context = {}
            for q in pending_choices:
                qid = str(q.get("id"))
                print(f"  [搜索中] 第 {qid} 题...")
                search_res = self._search_searxng(f"计算机网络 {q.get('question')}")
                current_batch_context[qid] = search_res

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

【参考背景信息】（当前轮次搜索结果）：
{json.dumps(current_batch_context, ensure_ascii=False)}

题目内容：
{json.dumps(pending_choices, ensure_ascii=False)}
"""
            solve_messages = self._build_image_message(
                prompt,
                self._collect_question_image_inputs(pending_choices, image_map),
            )

            response = self._call_ai(
                self.complex_client,
                self.complex_model,
                solve_messages,
                response_format={"type": "json_object"},
            )

            if not response or not hasattr(response, "choices"):
                continue
            content = response.choices[0].message.content
            if not content:
                continue
            try:
                data = self._parse_json_safe(content)
            except Exception as e:
                print(f"  [错误] JSON 解析失败: {e}")
                continue

            # 防御性解析
            results_data = []
            if isinstance(data, list):
                results_data = data
            elif isinstance(data, dict):
                results_data = data.get("results", [])
                if not results_data and "ans" in data:
                    results_data = data.get("ans", [])

            results_map = {}
            for r in results_data:
                if isinstance(r, dict):
                    qid = str(r.get("id"))
                    results_map[qid] = r

            still_pending = []
            for q in pending_choices:
                qid = str(q.get("id"))
                res = results_map.get(qid)
                if not res:
                    still_pending.append(q)
                    continue

                thought_match = re.search(
                    r"<thought>(.*?)</thought>", res.get("analysis", ""), re.S
                )
                raw_answer = str(res.get("answer", ""))
                ans_match = re.search(r"<answer>(.*?)</answer>", raw_answer, re.S)
                thought_str = thought_match.group(1).strip() if thought_match else ""
                answer_str = (
                    ans_match.group(1).strip() if ans_match else raw_answer.strip()
                )

                review_prompt = f"""作为审阅专家，请严谨核查此选择题。
题目：{q.get('question')}
思路：{thought_str}
答案：{answer_str}
【参考资料】（启动时加载的参考文件）：
{reference_materials_text}

【参考背景信息】：{current_batch_context.get(qid, "无")}

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
                    self._collect_question_image_inputs([q], image_map),
                )
                rev_res = self._call_ai(
                    self.simple_client,
                    self.simple_model,
                    review_messages,
                    use_tools=True,
                )

                if not rev_res or not hasattr(rev_res, "choices"):
                    continue
                rev_content = rev_res.choices[0].message.content
                if rev_content and "PASS" in rev_content.strip().upper():
                    # 兜底检查
                    clean_ans = re.sub(r"[^A-Za-z ]", "", answer_str).upper().strip()
                    if not clean_ans:
                        print(f"  [题号 {qid}] 答案格式非法，进入下一轮重试")
                        still_pending.append(q)
                        continue

                    print(f"  [题号 {qid}] 审阅通过: {clean_ans}")
                    try:
                        base_qid = qid.split("-")[0]
                        idx = int(base_qid) - 1
                        if 0 <= idx < self.CHOICE_QUESTION_LIMIT:
                            final_ans[idx] = clean_ans
                    except:
                        pass
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
    ) -> List[Dict[str, Any]]:
        """使用复杂模型解决简答题，并由简单模型审阅（CoT 循环反馈机制）"""
        if not short_answer_list:
            return []

        student_name = self.config["student_info"]["name"]
        pending_questions = short_answer_list.copy()
        final_results_map = {}
        best_answers_per_q = {}  # 记录每题的最好答案
        no_improvement_rounds = 0  # 连续无进展轮数

        max_rounds = 10
        for round_idx in range(max_rounds):
            if not pending_questions:
                break

            round_improved = False

            print(
                f"\n>>> 简答题处理第 {round_idx + 1} 轮 (剩余 {len(pending_questions)} 题)..."
            )
            # 为当前轮次的每道题进行搜索背景调查
            current_batch_context = {}
            for q in pending_questions:
                qid = str(q.get("id"))
                print(f"  [搜索中] 第 {qid} 题...")
                search_res = self._search_searxng(f"计算机网络 {q.get('question')}")
                current_batch_context[qid] = search_res

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

【参考背景信息】（当前轮次搜索结果）：
{json.dumps(current_batch_context, ensure_ascii=False)}

待处理题目（包含题目和可能的反馈意见）：
{json.dumps(pending_questions, ensure_ascii=False)}
"""
            solve_messages = self._build_image_message(
                solve_prompt,
                self._collect_question_image_inputs(pending_questions, image_map),
            )
            response = self._call_ai(
                self.complex_client,
                self.complex_model,
                solve_messages,
                response_format={"type": "json_object"},
            )

            if not response or not hasattr(response, "choices"):
                continue
            content = response.choices[0].message.content
            if not content:
                continue

            try:
                data = self._parse_json_safe(content)
            except Exception as e:
                print(f"  [错误] JSON 解析失败: {e}")
                continue

            current_answers = {}
            # 防御性解析
            ans_data = []
            if isinstance(data, list):
                ans_data = data
            elif isinstance(data, dict):
                ans_data = data.get("answers", [])
                if not ans_data and "results" in data:
                    ans_data = data.get("results", [])

            for a in ans_data:
                if not isinstance(a, dict):
                    continue
                qid = str(a.get("id"))
                raw_ans = a.get("answer", "")
                ans_match = re.search(r"<answer>(.*?)</answer>", raw_ans, re.S)
                current_answers[qid] = (
                    ans_match.group(1).strip() if ans_match else raw_ans
                )

            # 2. 简单模型审阅
            still_pending = []
            for q in pending_questions:
                qid = str(q.get("id"))
                title = q.get("question", "")
                ans = current_answers.get(qid, "（未生成回答）")

                review_prompt = f"""你是一个计算机网络审阅专家。请严谨审阅以下简答题答案。
题目：{title}
回答：{ans}
【参考资料】（启动时加载的参考文件）：
{reference_materials_text}

【参考背景信息】：{current_batch_context.get(qid, "暂无相关背景资料")}

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
                    self._collect_question_image_inputs([q], image_map),
                )
                rev_res = self._call_ai(
                    self.simple_client,
                    self.simple_model,
                    review_messages,
                    use_tools=True,
                )

                if not rev_res or not hasattr(rev_res, "choices"):
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
                    final_results_map[qid] = {
                        "index": q.get("id", ""),
                        "title": title,
                        "answer": ans,
                    }
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

    def _read_pdf_text(self, pdf_path: str) -> str:
        """读取整份 PDF 文本"""
        try:
            reader = PdfReader(pdf_path)
            chunks = []
            for page in reader.pages:
                chunks.append(page.extract_text() or "")
            text = "\n".join(chunks).strip()
            max_chars = 40000
            if len(text) > max_chars:
                return text[:max_chars]
            return text
        except Exception as e:
            print(f">>> [警告] 读取参考 PDF 文本失败: {e}")
            return ""

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

    def _prepare_reference_materials(
        self,
        reference_pdf_paths: List[str],
        reference_md_paths: List[str],
    ) -> str:
        """加载参考 PDF/MD，并整理为每题都可复用的参考资料文本"""
        blocks: List[str] = []

        def _append_block(kind: str, src_path: str, content: str):
            if not content:
                return
            blocks.append(
                f"[参考文件] 类型: {kind}\n路径: {src_path}\n内容:\n{content}"
            )

        for p in reference_pdf_paths:
            if not os.path.exists(p):
                print(f">>> [警告] 参考 PDF 不存在，已忽略: {p}")
                continue
            _append_block("pdf", p, self._read_pdf_text(p))

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
            if not isinstance(val, str):
                continue
            clean_val = re.sub(r"[^A-Za-z ]", "", val).upper().strip()
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

    def _load_cache(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """加载缓存，若缓存比PDF新则返回缓存数据"""
        cache_path = self._get_cache_path(pdf_path)

        # 检查缓存文件是否存在
        if not os.path.exists(cache_path):
            return None

        # 检查PDF是否存在及修改时间
        if not os.path.exists(pdf_path):
            return None

        pdf_mtime = os.path.getmtime(pdf_path)
        cache_mtime = os.path.getmtime(cache_path)

        # 缓存如果比PDF新则有效
        if cache_mtime <= pdf_mtime:
            print(f">>> [缓存] 缓存已过期（PDF 已更新），将重新处理")
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
        reference_pdf_paths: Optional[List[str]] = None,
        reference_md_paths: Optional[List[str]] = None,
    ):
        print(f">>> 开始解析 PDF: {pdf_path}")
        homework_name, parts, screenshots = self.parse_pdf(pdf_path)
        print(f">>> 作业名称: {homework_name}")

        reference_pdf_paths = reference_pdf_paths or []
        reference_md_paths = reference_md_paths or []
        reference_materials_text = self._prepare_reference_materials(
            reference_pdf_paths,
            reference_md_paths,
        )
        if reference_materials_text != "无":
            print(
                f">>> 已加载参考资料: PDF {len(reference_pdf_paths)} 个, MD {len(reference_md_paths)} 个"
            )
            print(">>> 后续每次题目生成与审阅都会附带这些参考资料作为上下文...")
        else:
            print(">>> 未加载到有效参考资料，将仅使用题目截图与检索背景进行作答。")

        # 尝试加载缓存
        cache_data = self._load_cache(pdf_path)
        if cache_data is None:
            cache_data = {}

        # 处理选择题（检查缓存）
        if "choice_ans" in cache_data and cache_data.get("choice_ans"):
            print(">>> [缓存] 使用缓存的选择题答案")
            ans = cache_data["choice_ans"]
        else:
            print(">>> 正在处理选择题...")
            ans = self.solve_choice_questions(
                parts["choice"],
                screenshots.get("choice", {}),
                reference_materials_text,
            )
            cache_data["choice_ans"] = ans
            cache_data["choice_cached_at"] = time.time()
            self._save_cache(pdf_path, cache_data)

        # 处理简答题（检查缓存）
        if "short_answers" in cache_data and cache_data.get("short_answers"):
            print(">>> [缓存] 使用缓存的简答题答案")
            questions = cache_data["short_answers"]
        else:
            print(">>> 正在处理简答题...")
            questions = self.solve_short_answers(
                parts["short_answer"],
                screenshots.get("short_answer", {}),
                reference_materials_text,
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
"""
                adj_res = self._call_ai(
                    self.complex_client,
                    self.complex_model,
                    [{"role": "user", "content": adjustment_prompt}],
                    response_format={"type": "json_object"},
                )
                if not adj_res or not hasattr(adj_res, "choices"):
                    continue
                adj_content = adj_res.choices[0].message.content
                if adj_content:
                    try:
                        adjusted_context = self._parse_json_safe(adj_content)
                        context = self._guard_context_update(context, adjusted_context)
                    except Exception as e:
                        print(f">>> [警告] 反馈修复 JSON 解析失败，保留原内容: {e}")
                        continue
                    output_file = self.generate_docx(homework_name, context)
                    print(f"\n[成功] 已根据反馈重新生成作业: {output_file}")

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
"""
                adj_res = self._call_ai(
                    self.complex_client,
                    self.complex_model,
                    [{"role": "user", "content": adjustment_prompt}],
                    response_format={"type": "json_object"},
                )
                if not adj_res or not hasattr(adj_res, "choices"):
                    continue
                adj_content = adj_res.choices[0].message.content
                if adj_content:
                    try:
                        adjusted_context = self._parse_json_safe(adj_content)
                        context = self._guard_context_update(context, adjusted_context)
                    except Exception as e:
                        print(f">>> [警告] 反馈修复 JSON 解析失败，保留原内容: {e}")
                        continue
                    output_file = self.generate_docx(homework_name, context)
                    print(f"\n[成功] 已根据反馈重新生成作业: {output_file}")


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
        "--reference-pdf",
        action="append",
        default=[],
        help="参考答案 PDF 路径（可多次传入，将在每次题目生成与审阅时附带）",
    )
    parser.add_argument(
        "--reference-md",
        action="append",
        default=[],
        help="参考 Markdown 路径（可多次传入，将在每次题目生成与审阅时附带）",
    )
    args = parser.parse_args()

    target_pdf = args.target_pdf
    reference_pdfs = args.reference_pdf
    reference_mds = args.reference_md

    print("\n>>> 命令行参数解析结果:")
    print(f">>> target_pdf: {target_pdf}")
    print(f">>> reference_pdfs ({len(reference_pdfs)}):")
    for i, p in enumerate(reference_pdfs, start=1):
        print(f"    {i}. {p}")
    if not reference_pdfs:
        print("    (无)")

    print(f">>> reference_mds ({len(reference_mds)}):")
    for i, p in enumerate(reference_mds, start=1):
        print(f"    {i}. {p}")
    if not reference_mds:
        print("    (无)")

    try:
        automator = HomeworkAutomator()
        automator.run(target_pdf, reference_pdfs, reference_mds)
    except Exception as e:
        print(f"\n[错误] 运行失败: {e}")
