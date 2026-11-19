# -*- coding: utf-8 -*-
"""量化训练的计算精度选择（T4 上 bf16 会直接崩，2026-11-19 云端实测）.

Kaggle T4 上第一次真跑（adapter 已修好、trainable tensors=504）后，
第 2 步就 `CUDA error: an illegal memory access was encountered`。
T4 是 Turing（compute capability 7.5），**硬件不支持 bfloat16**；脚本里
`bnb_4bit_compute_dtype=torch.bfloat16` 在 7.5 上不是「慢一点」，是非法访存。

所以精度必须按**运行时能力**选，不能按「新一代卡都行」的直觉写死。
"""
import inspect
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "train"))

import train_dpo  # noqa: E402


class _FakeTorch:
    """够用的假 torch：只提供两个 dtype 哨兵。"""

    class _D:
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return "torch.%s" % self.name

    bfloat16 = _D("bfloat16")
    float16 = _D("float16")


class TestPickComputeDtype(unittest.TestCase):
    def test_ampere及以上用bfloat16(self):
        for cap in ((8, 0), (8, 6), (8, 9), (9, 0), (12, 0)):
            with self.subTest(cap=cap):
                got = train_dpo.pick_compute_dtype(cap, _FakeTorch)
                self.assertIs(got, _FakeTorch.bfloat16, "capability %s 支持 bf16" % (cap,))

    def test_turing及更老用float16(self):
        # T4 = (7,5) / V100 = (7,0) / 更老一律 fp16：这些卡 bf16 会非法访存
        for cap in ((7, 5), (7, 0), (6, 1), (5, 2)):
            with self.subTest(cap=cap):
                got = train_dpo.pick_compute_dtype(cap, _FakeTorch)
                self.assertIs(got, _FakeTorch.float16, "capability %s 不支持 bf16" % (cap,))

    def test_能力查询失败时保守用float16(self):
        self.assertIs(train_dpo.pick_compute_dtype(None, _FakeTorch), _FakeTorch.float16)


class TestCallSiteUsesCapability(unittest.TestCase):
    def test_计算精度必须按运行时能力选(self):
        src = inspect.getsource(train_dpo.main)
        self.assertIn("pick_compute_dtype(", src)
        self.assertIn("get_device_capability()", src)
        # 写死 bf16 是那次崩溃的直接原因，不许复活
        self.assertNotIn("bnb_4bit_compute_dtype=torch.bfloat16", src)


if __name__ == "__main__":
    unittest.main()
