"""缓存管理"""

import os
import json
from typing import Any, Dict, List, Optional


class CacheManager:
    """管理作业的缓存"""

    def __init__(self):
        self.cache_dir = ".homework_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_path(self, pdf_path: str) -> str:
        """生成缓存文件路径"""
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        return os.path.join(self.cache_dir, f"{pdf_name}.cache.json")

    def load_cache(
        self,
        pdf_path: str,
        input_paths: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """加载缓存，若缓存比所有输入文件都新则返回缓存数据"""
        cache_path = self.get_cache_path(pdf_path)

        if not os.path.exists(cache_path):
            return None

        cache_mtime = os.path.getmtime(cache_path)

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

    def save_cache(self, pdf_path: str, cache_data: Dict[str, Any]) -> None:
        """保存缓存数据"""
        cache_path = self.get_cache_path(pdf_path)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            print(f">>> [缓存] 已保存缓存: {cache_path}")
        except Exception as e:
            print(f">>> [缓存] 保存缓存失败: {e}")

    def has_valid_choice_cache(self, values: Any) -> bool:
        """检查缓存中的选择题答案列表是否全部满足格式约束"""
        from .utils import normalize_choice_answer

        if not isinstance(values, list):
            return False

        for val in values:
            if not val:
                continue
            if not normalize_choice_answer(val):
                return False
        return True
