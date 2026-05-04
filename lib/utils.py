"""工具函数集合"""

import os
import json
import re
import base64
from typing import Any, Dict, List, Tuple


def build_image_message(prompt: str, image_inputs: List[Any]) -> List[Dict[str, Any]]:
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


def collect_question_image_inputs(
    questions: List[Dict[str, Any]], question_image_map: Dict[str, str]
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


def parse_json_safe(content: str) -> Any:
    """安全解析 JSON，自动处理代码块标记包裹的情况"""
    if not content:
        return None
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.S)
    if match:
        content = match.group(1).strip()
    return json.loads(content)


def normalize_choice_answer(value: Any) -> str:
    """规范化选择题答案：仅保留字母，去空格后长度必须小于 10"""
    if not isinstance(value, str):
        return ""

    clean_val = re.sub(r"[^A-Za-z]", "", value).upper()
    if not clean_val:
        return ""
    if len(clean_val) >= 10:
        return ""
    return clean_val


def clean_markdown(text: str) -> str:
    """强力去除文本中的 Markdown 语法，返回纯文本"""
    if not text:
        return ""
    if not isinstance(text, str):
        return text

    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"\*\*+(.*?)\*\*+", r"\1", text)
    text = re.sub(r"\*+(.*?)\*+", r"\1", text)
    text = re.sub(r"__+(.*?)__+", r"\1", text)
    text = re.sub(r"_+(.*?)_+", r"\1", text)
    text = re.sub(r"^#+\s+", "", text, flags=re.M)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"^[\s\t]*[\*\-\+]\s+", "", text, flags=re.M)
    text = re.sub(r"^[\s\t]*\d+\.\s+", "", text, flags=re.M)
    return text.strip()


def read_md_text(md_path: str) -> str:
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


def prepare_reference_materials(reference_md_paths: List[str]) -> str:
    """加载参考 Markdown，并整理为每题都可复用的参考资料文本"""
    blocks: List[str] = []

    for p in reference_md_paths:
        if not os.path.exists(p):
            print(f">>> [警告] 参考 Markdown 不存在，已忽略: {p}")
            continue
        content = read_md_text(p)
        if content:
            blocks.append(f"[参考文件] 类型: md\n路径: {p}\n内容:\n{content}")

    if not blocks:
        return "无"

    joined = "\n\n".join(blocks)
    max_chars = 120000
    if len(joined) > max_chars:
        return joined[:max_chars]
    return joined
