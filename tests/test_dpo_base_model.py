# -*- coding: utf-8 -*-
"""云端可用本地已下载的模型路径覆盖 HF repo id（AMD 实例 hf-mirror 直连受限时）。

huggingface_hub 在集群代理下对 resolve 的 HEAD 请求会卡死（实测僵 16h），
但 curl 带代理下载 6.4MB/s 通。于是先把权重 curl 到本地目录，再让
train_dpo 用 DPO_BASE_MODEL 指向它，绕开 hub 的网络层。
"""
import inspect
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "train"))

import train_dpo  # noqa: E402


class TestResolveBaseModel(unittest.TestCase):
    def test_未设置时用CFG默认(self):
        env = {}
        self.assertEqual(train_dpo.resolve_base_model(env), train_dpo.CFG["base_model"])

    def test_设置后返回本地路径(self):
        env = {"DPO_BASE_MODEL": "/workspace/models/Qwen3-4B"}
        self.assertEqual(train_dpo.resolve_base_model(env), "/workspace/models/Qwen3-4B")

    def test_空串等同未设置(self):
        env = {"DPO_BASE_MODEL": ""}
        self.assertEqual(train_dpo.resolve_base_model(env), train_dpo.CFG["base_model"])


class TestCallSiteUsesResolver(unittest.TestCase):
    def test_main的tokenizer与模型加载都走resolver(self):
        src = inspect.getsource(train_dpo.main)
        # 两处 from_pretrained 都必须用解析后的变量，不许再裸写 CFG["base_model"]
        self.assertIn("base_model = resolve_base_model(", src)
        self.assertNotIn('from_pretrained(CFG["base_model"]', src)

    def test_snapshot记录实际模型路径(self):
        src = inspect.getsource(train_dpo.main)
        self.assertIn('"base_model": base_model', src)


if __name__ == "__main__":
    unittest.main()
