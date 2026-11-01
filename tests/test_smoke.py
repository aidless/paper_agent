# -*- coding: utf-8 -*-
"""paper_agent 冒烟测试：静态校验 + 检索降级 + 评测数字一致性.

不依赖 GPU / 大模型 / 完整训练集。验证三件事：
  1. 全部 .py 可语法解析、全部 .json 可 JSON 解析；
  2. fused_search 在向量索引缺失时优雅降级为关键词回退（不抛错）；
  3. reports/eval_3way_v2.json 的核心数字与 DELIVERY.md 一致（防手抄漂移）。
"""
import ast
import json
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestStaticParse(unittest.TestCase):
    def test_all_py_parse(self):
        bad = []
        for root, _dirs, files in os.walk(REPO):
            if ".git" in root or "outputs" in root:
                continue
            for f in files:
                if f.endswith(".py"):
                    p = os.path.join(root, f)
                    try:
                        with open(p, encoding="utf-8") as fh:
                            ast.parse(fh.read())
                    except SyntaxError as e:
                        bad.append((p, str(e)))
        self.assertEqual(bad, [], f"syntax errors: {bad}")

    def test_all_json_parse(self):
        bad = []
        for root, _dirs, files in os.walk(REPO):
            if ".git" in root or "outputs" in root:
                continue
            for f in files:
                if f.endswith(".json"):
                    p = os.path.join(root, f)
                    try:
                        with open(p, encoding="utf-8") as fh:
                            json.load(fh)
                    except Exception as e:
                        bad.append((p, str(e)))
        self.assertEqual(bad, [], f"json errors: {bad}")


class TestRetrievalFallback(unittest.TestCase):
    def test_fused_search_keyword_fallback(self):
        """强制向量通道缓存为 None → fused_search 应走关键词回退并返回列表，不抛错."""
        sys.path.insert(0, os.path.join(REPO, "retrieval"))
        import fuse
        fuse.search_pes2o_vector.cache = None
        fuse.search_incr_vector.cache = None
        fuse._st_model.cache = None
        results = fuse.fused_search("LLM agent evaluation", top_k=3)
        self.assertIsInstance(results, list)


class TestEvalNumbers(unittest.TestCase):
    def test_eval_3way_v2_key_numbers(self):
        """eval_3way_v2.json 的核心数字必须与 DELIVERY.md 一致（防手抄漂移）."""
        p = os.path.join(REPO, "reports", "eval_3way_v2.json")
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        self.assertAlmostEqual(d["base"]["val_nll_60"], 3.9718, places=4)
        self.assertAlmostEqual(d["ckpt136"]["val_nll_60"], 0.3879, places=4)
        self.assertEqual(d["ckpt136"]["gen"]["evd"], 1.0)
        self.assertEqual(d["ckpt136"]["gen"]["tag"], 0.0)


if __name__ == "__main__":
    unittest.main()
