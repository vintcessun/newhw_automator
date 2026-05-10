"""PDF 处理和题目截图生成"""

import os
import re
from typing import Any, Dict, List, Tuple
import fitz
import numpy as np


class PDFProcessor:
    """处理 PDF 的解析和截图生成"""

    def __init__(self, simple_client, simple_model):
        self.simple_client = simple_client
        self.simple_model = simple_model

    def extract_page_lines(self, page: fitz.Page) -> List[Dict[str, Any]]:
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

    def merge_pixmaps(self, pixmaps: List[fitz.Pixmap]) -> fitz.Pixmap:
        """垂直合并多个 Pixmap，使用 numpy 处理"""
        if not pixmaps:
            return None
        if len(pixmaps) == 1:
            return pixmaps[0]

        arrays = []
        for p in pixmaps:
            img = np.frombuffer(p.samples, dtype=np.uint8).reshape(
                p.height, p.width, p.n
            )
            arrays.append(img)

        max_w = max(a.shape[1] for a in arrays)
        padded_arrays = []
        for a in arrays:
            if a.shape[1] < max_w:
                pad = (
                    np.ones(
                        (a.shape[0], max_w - a.shape[1], a.shape[2]), dtype=np.uint8
                    )
                    * 255
                )
                a = np.hstack([a, pad])
            padded_arrays.append(a)

        merged_array = np.vstack(padded_arrays)
        return fitz.Pixmap(
            pixmaps[0].colorspace,
            merged_array.shape[1],
            merged_array.shape[0],
            merged_array.tobytes(),
            pixmaps[0].alpha,
        )

    def extract_question_ids_from_pdf(self, pdf_path: str) -> List[str]:
        """直接从 PDF 文本中提取所有题号，无需 AI 模型"""
        question_ids = []
        try:
            doc = fitz.open(pdf_path)
            seen_ids = set()

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                for line in self.extract_page_lines(page):
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

    def _base_question_number(self, qid: Any) -> int:
        m = re.match(r"^\s*(\d{1,3})", str(qid))
        return int(m.group(1)) if m else -1

    def extract_parts_by_fixed_sections(self, pdf_path: str) -> Dict[str, Any]:
        """按固定分区标题解析题目，并按连续递增题号过滤正文编号。"""
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
                for line in self.extract_page_lines(doc[page_idx]):
                    text = str(line.get("text", "")).strip()
                    if not text:
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
        expected_question_number = None
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
                    q_number = self._base_question_number(m.group(1))
                    if q_number < 0:
                        continue
                    if (
                        expected_question_number is not None
                        and q_number != expected_question_number
                    ):
                        continue
                    events.append(
                        {
                            "kind": "question",
                            "section": current_section,
                            "idx": idx,
                            "id": m.group(1),
                        }
                    )
                    expected_question_number = q_number + 1

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

    def extract_homework_name_from_pdf(self, pdf_path: str, call_ai_func) -> str:
        """用 AI 从 PDF 首页提取作业名称"""
        try:
            doc = fitz.open(pdf_path)
            first_page = doc[0]
            lines = self.extract_page_lines(first_page)
            doc.close()

            first_lines = "\n".join([line["text"] for line in lines[:10]])

            prompt = f"""请从以下 PDF 首页文本中提取作业名称。只需返回一个作业名称字符串，形如"第x课xxx"。

文本内容：
{first_lines}

返回格式：只输出作业名称，不要其他内容。"""

            messages = [{"role": "user", "content": prompt}]
            response = call_ai_func(
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

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            for line in self.extract_page_lines(page):
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
            """截取指定高度范围的页面 Pixmap"""
            lines = self.extract_page_lines(page)
            relevant_lines = []
            for ln in lines:
                ly0, ly1 = ln["bbox"][1], ln["bbox"][3]
                is_header = ly1 < 60
                is_footer = ly0 > page.rect.height - 60

                if ly0 >= y0 - 4 and ly1 <= y1 + 4:
                    if not (is_header or is_footer):
                        relevant_lines.append(ln)

            if relevant_lines:
                real_y0 = max(y0, min(ln["bbox"][1] for ln in relevant_lines) - 8)
                real_y1 = min(y1, max(ln["bbox"][3] for ln in relevant_lines) + 8)
            else:
                real_y0, real_y1 = y0, y1

            if real_y1 <= real_y0:
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
                end_page = len(doc) - 1
                end_y = doc[end_page].rect.height

            segments = []
            for p_idx in range(start_page, end_page + 1):
                page = doc[p_idx]
                y0 = start_y if p_idx == start_page else 0
                y1 = end_y if p_idx == end_page else page.rect.height

                if y1 > y0 + 2:
                    pix = _get_clip_pixmap(page, y0, y1)
                    if pix:
                        segments.append(pix)

            if segments:
                out_name = f"{pdf_prefix}_{s['id']}.png"
                out_path = os.path.join(out_dir, out_name)
                final_pix = self.merge_pixmaps(segments)
                final_pix.save(out_path)
                id_to_path[s["id"]] = out_path

        doc.close()

        return {
            "choice": {qid: id_to_path.get(qid, "") for qid in choice_ids},
            "short_answer": {qid: id_to_path.get(qid, "") for qid in short_ids},
            "programming": {qid: id_to_path.get(qid, "") for qid in prog_ids},
        }

    def prepare_parse_pdf_image_inputs(self, pdf_path: str) -> List[Tuple[str, str]]:
        """将 PDF 转成页面图片，供多模态结构解析使用"""
        if not os.path.exists(pdf_path):
            return []

        out_dir = os.path.join("problems", "_parse_mm")
        os.makedirs(out_dir, exist_ok=True)

        inputs: List[Tuple[str, str]] = []
        try:
            doc = fitz.open(pdf_path)
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            page_count = len(doc)
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

    def prepare_reference_pdf_image_inputs(
        self, reference_pdf_paths: List[str], max_pages_per_pdf: int = 6
    ) -> List[Tuple[str, str]]:
        """将参考 PDF 转为多模态图片输入"""
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
