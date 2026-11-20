# -*- coding: utf-8 -*-
"""Kaggle kernel 依赖须显式处置 torchao 版本冲突（2026-11-19 v7 实测）.

v7 读数（paper_agent/outputs/dpo-kaggle/dsh-dpo-train.log:56/140）：
    model, mode = attach_trainable_adapter(model, sft_ad, peft_mod=peft, ...)
    ImportError: Found an incompatible version of torchao. Found version 0.10.0,
                 but only versions above 0.16.0 are supported

根因不在模型/显存：peft 0.20 的 import_utils.py:150-164 硬检查 torchao >= 0.16，
Kaggle 镜像预装 torchao 0.10.0。而本方案走的是**非量化 fp16 权重**（load_plan
在 capability<8 时 use_4bit=False），整条链路根本不消费 torchao —— 旧版本只是
被「装了却版本低」这件事绊倒。

修法选择：卸掉 torchao（find_spec 返回 None → is_torchao_available() 直接 False，
不抛 ImportError），而不是升级到 0.16+（会连带 torch/torchvision 版本，引入新面）。
本测试只看 kernel 源码是否显式、无条件地卸掉它；真机复验靠 v8 kernel 跑。
"""
import os
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL = os.path.join(REPO, "cloud", "kaggle-dpo", "kernel", "dpo-kernel.py")


def _source():
    with open(KERNEL, encoding="utf-8") as fh:
        return fh.read()


class TestKernelTorchaoDisposition(unittest.TestCase):
    def test_kernel_显式卸载_torchao(self):
        """源码必须出现 uninstall + torchao，且是字面量而非注释/字符串里的空话。"""
        src = _source()
        code_lines = []
        for line in src.splitlines():
            s = line.strip()
            if s.startswith("#"):
                continue
            code_lines.append(s)
        code = "\n".join(code_lines)
        self.assertIn("uninstall", code)
        self.assertIn("torchao", code)

    def test_卸载在训练之前(self):
        """必须在 `== 训练 ==` 打印之前完成，否则 get_peft_model 已经抛错。"""
        src = _source()
        self.assertLess(src.index("uninstall"), src.index('print("== 训练 ==")'))

    def test_卸载是check_为真还是显式忽略错误(self):
        """torchao 可能压根没装；uninstall 会非 0，kernel 不能以 check=True 炸掉。"""
        src = _source()
        line = next(l for l in src.splitlines() if "uninstall" in l and not l.strip().startswith("#"))
        self.assertNotIn("check=True", line)


if __name__ == "__main__":
    unittest.main()
