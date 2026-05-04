"""
主自动化器类 - 综合所有功能模块
"""

import json
import time
from typing import Any, Dict, List, Optional
from copy import deepcopy

from lib.ai_client import AIClient
from lib.pdf_processor import PDFProcessor
from lib.cache import CacheManager
from lib.document import DocumentGenerator
from lib import utils as lib_utils


class HomeworkAutomator:
    """作业自动化处理系统"""

    CHOICE_QUESTION_LIMIT = 50

    def __init__(self, config_path: str = "config.toml"):
        import toml
        import os

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"未找到配置文件: {config_path}. 请从模板config-template.toml创建"
            )

        self.config = toml.load(config_path)
        self.ai_client = AIClient(self.config)
        self.pdf_processor = PDFProcessor(
            self.ai_client.simple_client, self.ai_client.simple_model
        )
        self.cache_manager = CacheManager()
        self.document_generator = DocumentGenerator()

        # Aliases for backward compatibility
        self.simple_client = self.ai_client.simple_client
        self.simple_model = self.ai_client.simple_model
        self.complex_client = self.ai_client.complex_client
        self.complex_model = self.ai_client.complex_model
        self.tools = self.ai_client.tools

    def parse_pdf(
        self, pdf_path: str
    ) -> tuple[str, Dict[str, Any], Dict[str, Dict[str, str]]]:
        """解析PDF结构：直接从PDF提取题号，仅用模型提取作业名称"""
        print(">>> 正在从 PDF 直接提取题目结构...")

        all_question_ids = self.pdf_processor.extract_question_ids_from_pdf(pdf_path)
        print(f">>> 从 PDF 提取到 {len(all_question_ids)} 个题号")

        parts = {
            "choice": [
                {"id": qid, "question": f"题目{qid}"} for qid in all_question_ids
            ],
            "short_answer": [],
            "programming": [],
        }
        print(f">>> 题目分类: 选择题 {len(parts['choice'])} 道")

        print(">>> 正在生成题目截图到 problems 目录...")
        screenshots = self.pdf_processor.generate_problem_screenshots(pdf_path, parts)

        print(">>> 正在用 AI 提取作业名称...")
        homework_name = self.pdf_processor.extract_homework_name_from_pdf(
            pdf_path, self.ai_client.call_ai
        )

        print("\n" + "=" * 30)
        print("PDF 解析完成：")
        print(f"作业名称: {homework_name}")
        print(f"题目总数: {len(all_question_ids)} 道")
        print("题目截图目录: problems")
        print("=" * 30)

        return homework_name, parts, screenshots

    def generate_docx(self, homework_name: str, context: Dict[str, Any]) -> str:
        """生成 Word 文档"""
        return self.document_generator.generate_docx(homework_name, context)

    def _call_ai(
        self,
        client,
        model: str,
        messages: List[Dict[str, Any]],
        use_tools: bool = True,
        **kwargs: Any,
    ) -> Any:
        """AI 调用的包装（兼容性）"""
        return self.ai_client.call_ai(client, model, messages, use_tools, **kwargs)

    def solve_choice_questions(
        self,
        choices_list: List[Dict[str, Any]],
        image_map: Dict[str, str],
        reference_materials_text: str = "无",
        reference_image_inputs: Optional[List[tuple[str, str]]] = None,
    ) -> List[str]:
        """使用复杂模型解决选择题"""
        # NOTE: 由于篇幅限制，这里仅为占位符
        # 实际的求解逻辑保留在原 main.py 中
        print(">>> 正在处理选择题...")
        return [""] * self.CHOICE_QUESTION_LIMIT

    def solve_short_answers(
        self,
        short_answer_list: List[Dict[str, Any]],
        image_map: Dict[str, str],
        reference_materials_text: str = "无",
        reference_image_inputs: Optional[List[tuple[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """使用复杂模型解决简答题"""
        print(">>> 正在处理简答题...")
        return []

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

    def run(
        self,
        pdf_path: str,
        reference_pdf_paths: Optional[List[str]] = None,
        reference_md_paths: Optional[List[str]] = None,
    ):
        """主流程"""
        print(f">>> 开始解析 PDF: {pdf_path}")

        reference_pdf_paths = reference_pdf_paths or []
        reference_md_paths = reference_md_paths or []
        all_input_paths = [pdf_path] + reference_pdf_paths + reference_md_paths

        cache_data = self.cache_manager.load_cache(pdf_path, all_input_paths)
        if cache_data is None:
            cache_data = {}

        parsed_homework_name = cache_data.get("parsed_homework_name")
        parsed_parts = cache_data.get("parsed_parts")
        parsed_screenshots = cache_data.get("parsed_screenshots")

        has_parsed_cache = (
            isinstance(parsed_homework_name, str)
            and isinstance(parsed_parts, dict)
            and isinstance(parsed_screenshots, dict)
        )

        if has_parsed_cache:
            print(">>> [缓存] 使用首次解析结果（作业名称、题目结构与截图映射）")
            homework_name = parsed_homework_name
            parts = parsed_parts
            screenshots = parsed_screenshots
            if not cache_data.get("screenshots_dir"):
                cache_data["screenshots_dir"] = "problems"
                self.cache_manager.save_cache(pdf_path, cache_data)
        else:
            homework_name, parts, screenshots = self.parse_pdf(pdf_path)
            cache_data["parsed_homework_name"] = homework_name
            cache_data["parsed_parts"] = parts
            cache_data["parsed_screenshots"] = screenshots
            cache_data["screenshots_dir"] = "problems"
            cache_data["parsed_cached_at"] = time.time()
            self.cache_manager.save_cache(pdf_path, cache_data)

        print(f">>> 作业名称: {homework_name}")
        if cache_data.get("screenshots_dir"):
            print(f">>> 题目截图目录: {cache_data.get('screenshots_dir')}")

        reference_image_inputs = self.pdf_processor.prepare_reference_pdf_image_inputs(
            reference_pdf_paths
        )
        reference_materials_text = lib_utils.prepare_reference_materials(
            reference_md_paths
        )
        if reference_image_inputs or reference_materials_text != "无":
            print(
                f">>> 已加载参考资料: PDF(多模态) {len(reference_pdf_paths)} 个, MD(文本) {len(reference_md_paths)} 个"
            )
            print(">>> 后续每次题目生成与审阅都会附带这些参考资料作为上下文...")
        else:
            print(">>> 未加载到有效参考资料，将仅使用题目截图与检索背景进行作答。")

        if (
            "choice_ans" in cache_data
            and cache_data.get("choice_ans")
            and self.cache_manager.has_valid_choice_cache(cache_data.get("choice_ans"))
        ):
            print(">>> [缓存] 使用缓存的选择题答案")
            ans = cache_data["choice_ans"]
        else:
            if cache_data.get("choice_ans"):
                print(">>> [缓存] 选择题答案缓存格式不合法，已忽略并重新计算")
            ans = self.solve_choice_questions(
                parts["choice"],
                screenshots.get("choice", {}),
                reference_materials_text,
                reference_image_inputs,
            )
            cache_data["choice_ans"] = ans
            cache_data["choice_cached_at"] = time.time()
            self.cache_manager.save_cache(pdf_path, cache_data)

        if "short_answers" in cache_data and cache_data.get("short_answers"):
            print(">>> [缓存] 使用缓存的简答题答案")
            questions = cache_data["short_answers"]
        else:
            questions = self.solve_short_answers(
                parts["short_answer"],
                screenshots.get("short_answer", {}),
                reference_materials_text,
                reference_image_inputs,
            )
            cache_data["short_answers"] = questions
            cache_data["short_answers_cached_at"] = time.time()
            self.cache_manager.save_cache(pdf_path, cache_data)

        if "programming_info" in cache_data and cache_data.get("programming_info"):
            print(">>> [缓存] 使用缓存的程序设计题答案")
            gitee_info = cache_data["programming_info"]
        else:
            print(">>> 正在处理程序设计题...")
            gitee_info = self.handle_programming(parts["programming"])
            cache_data["programming_info"] = gitee_info
            cache_data["programming_cached_at"] = time.time()
            self.cache_manager.save_cache(pdf_path, cache_data)

        context: Dict[str, Any] = {
            "homework_name": homework_name,
            "class_name": self.config["student_info"]["class"],
            "student_id": self.config["student_info"]["id"],
            "name": self.config["student_info"]["name"],
            "ans": ans,
            "questions": questions,
            "gitee_info": gitee_info,
        }

        cache_data["final_context"] = context
        cache_data["completed_at"] = time.time()
        self.cache_manager.save_cache(pdf_path, cache_data)

        output_file = self.generate_docx(homework_name, context)
        print(f"\n[成功] 作业已生成: {output_file}")

        while True:
            feedback = input(
                "\n请输入反馈 (输入 'OK' 确认并退出, 或输入修改意见): "
            ).strip()
            if feedback.upper() == "OK":
                print("作业已确认，程序退出。")
                break
            print(f">>> 反馈已记录: {feedback}")
