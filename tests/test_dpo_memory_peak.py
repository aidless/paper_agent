# -*- coding: utf-8 -*-
"""非量化路径的 DPO 显存峰值控制（2026-11-19 v8 云端 OOM 实测）.

v8 读数（dsh-dpo-train.log）：
    [adapter] mode=fresh_lora / trainable tensors=504
    [data] pairs=400
    torch.OutOfMemoryError: Tried to allocate 20.00 MiB. GPU 0 has a total
    capacity of 14.56 GiB of which 20.81 MiB is free. ... 14.19 GiB is
    allocated by PyTorch, and 144.50 MiB is reserved but unallocated.
栈在 `training_step → accelerator.backward`，不是精度（v7 的 torchao 已解）。

非量化 fp16 4B 权重约 8GB；DPO 每条样本跑 chosen/rejected × policy/ref 四次
前向，梯度检查点若用 reentrant=True 会把每层输入整段留在图里，峰值再吃数 GB。
两条有证据的降峰措施：
  1. gradient_checkpointing_kwargs={"use_reentrant": False}：非重入版只保留
     张量形状/设备元数据，backward 时重算，峰值显著低；
  2. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True：OOM 文本自己建议的
     碎片治理（v8 reserved-but-unallocated 144.5 MiB 不大，但成本为零）。
"""
import inspect
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "train"))

import train_dpo  # noqa: E402

KERNEL = os.path.join(REPO, "cloud", "kaggle-dpo", "kernel", "dpo-kernel.py")


def _kernel_source():
    with open(KERNEL, encoding="utf-8") as fh:
        return fh.read()


class TestDpoMemoryPeak(unittest.TestCase):
    def test_非量化路径使用非重入梯度检查点(self):
        src = inspect.getsource(train_dpo.main)
        self.assertIn('gradient_checkpointing_kwargs={"use_reentrant": False}', src)

    def test_kernel_设置显存碎片治理(self):
        src = _kernel_source()
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", src)
        self.assertIn("expandable_segments:True", src)

    def test_精度旋钮没被动过(self):
        """OOM 不是精度问题，bf16/fp16 的既有判据不许被本次改动顺带改掉。"""
        self.assertIs(train_dpo.pick_compute_dtype((7, 5), _TF()), _TF().float32)
        self.assertIs(train_dpo.load_plan((7, 5))["dtype"], "float16")


class _TF:
    class _D:
        def __init__(self, name):
            self.name = name

    bfloat16 = _D("bfloat16")
    float16 = _D("float16")
    float32 = _D("float32")


if __name__ == "__main__":
    unittest.main()
