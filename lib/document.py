"""Word 文档生成"""

import os
import re
import time
from typing import Any, Dict, List, Optional, cast
from copy import deepcopy
from docxtpl import DocxTemplate
from .utils import clean_markdown


class DocumentGenerator:
    """生成 Word 文档"""

    def __init__(self):
        pass

    def text_to_subdoc(self, tpl: DocxTemplate, text: str):
        """将包含 Markdown 表格的文本转换为 Subdoc 对象以插入原生 Word 内容"""

        def _is_table_line(line: str) -> bool:
            s = line.strip()
            return s.count("|") >= 2

        def _is_separator_line(line: str) -> bool:
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
                            table.cell(r_idx, c_idx).text = clean_markdown(cell_text)
                    sd.add_paragraph("")
                else:
                    fallback_text = clean_markdown("\n".join(table_lines))
                    if fallback_text:
                        sd.add_paragraph(fallback_text)

                i = j
                continue

            cleaned = clean_markdown(cur)
            if cleaned:
                sd.add_paragraph(cleaned)
            i += 1

        return sd

    def build_render_questions_without_title(
        self, questions: Any
    ) -> List[Dict[str, Any]]:
        """为 Word 渲染准备简答题结构：仅保留 index 与 answer"""
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
                    "answer": clean_markdown(str(q_dict.get("answer", ""))),
                }
            )

        return normalized

    def generate_docx(self, homework_name: str, context: Dict[str, Any]) -> str:
        """生成最终的 docx 文件，包含表格支持与兜底清理"""
        tpl = DocxTemplate("template.docx")

        render_context = deepcopy(context)

        if "questions" in render_context:
            render_context["questions"] = self.build_render_questions_without_title(
                render_context["questions"]
            )
            for q in render_context["questions"]:
                ans_text = q.get("answer", "")
                q["answer"] = self.text_to_subdoc(tpl, ans_text)

        tpl.render(render_context)
        tpl_any = cast(Any, tpl)
        safe_name = re.sub(r'[\\/:*?"<>|]', "_", homework_name)
        output_name = f"{safe_name}.docx"

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
