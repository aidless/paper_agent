# -*- coding: utf-8 -*-
"""DPO adapter 挂载的可训练性断言（不依赖 GPU / transformers / peft 真实包）.

为什么有这组测试（2026-11-19 云端实测）：
    Kaggle T4 上跑 `train_dpo.py` 130 秒后失败——
    `ValueError: You cannot perform fine-tuning on purely quantized models.
     Please attach trainable adapters on top of the quantized model ...`
    根因是「SFT adapter 缺失」时代码只 `print("[warn] ...")` 就继续，4bit 量化基座
    没有任何可训练参数，DPOTrainer 一定拒跑。修法是把「挂 adapter」变成一条必然
    发生的事：有 SFT adapter 就续训，没有就新建一个 LoRA。

这组测试钉住三件事：
  1. 没有 adapter 时必须新建 LoRA（不是只打一行 warn）；
  2. 有 adapter 时必须走续训并解冻 lora 参数；
  3. 调用点必须真的用这个函数（结构性断言，防止有人改回 warn-and-continue）。
"""
import inspect
import json
import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "train"))

import train_dpo  # noqa: E402  模块级只 import argparse/json/os/random/numpy


class _FakeParam:
    def __init__(self, name):
        self.name = name
        self.requires_grad = False


class _FakeModel:
    """最小可观测模型：只记参数名，够判「lora 参数是否被解冻」。"""

    def __init__(self):
        self._params = [("base_layer.weight", _FakeParam("base_layer.weight")),
                        ("lora_A.default.weight", _FakeParam("lora_A.default.weight")),
                        ("lora_B.default.weight", _FakeParam("lora_B.default.weight"))]

    def named_parameters(self):
        return list(self._params)


class _FakePeftModel:
    def __init__(self, model):
        self.model = model

    def named_parameters(self):
        return self.model.named_parameters()

    def save_pretrained(self, out_dir):
        pass


class _FakePeft:
    """假 peft 模块：记录调用了哪条路径。"""

    calls = []

    class PeftModel:
        @staticmethod
        def from_pretrained(model, path, is_trainable=False):
            _FakePeft.calls.append(("from_pretrained", path, is_trainable))
            return _FakePeftModel(model)

    @staticmethod
    def get_peft_model(model, cfg):
        # peft 传进来的是 LoraConfig 对象（不是 dict），假件也要按对象记
        _FakePeft.calls.append(("get_peft_model", dict(getattr(cfg, "kw", {}))))
        return _FakePeftModel(model)


class _FakeLoraConfig:
    def __init__(self, **kw):
        self.kw = dict(kw)


class TestAttachTrainableAdapter(unittest.TestCase):
    def setUp(self):
        _FakePeft.calls = []

    def test_无adapter时必须新建LoRA而不是只warn(self):
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, "no-such-checkpoint")
            model, mode = train_dpo.attach_trainable_adapter(
                _FakeModel(), missing, peft_mod=_FakePeft, lora_config_cls=_FakeLoraConfig)
        self.assertEqual(mode, "fresh_lora", "adapter 缺失时必须新建 LoRA，否则量化基座无可训练参数")
        kinds = [c[0] for c in _FakePeft.calls]
        self.assertEqual(kinds, ["get_peft_model"])

    def test_新建LoRA的target_modules覆盖七个投影层(self):
        with tempfile.TemporaryDirectory() as td:
            train_dpo.attach_trainable_adapter(
                _FakeModel(), os.path.join(td, "nope"),
                peft_mod=_FakePeft, lora_config_cls=_FakeLoraConfig)
        cfg = [c[1] for c in _FakePeft.calls if c[0] == "get_peft_model"][0]
        self.assertEqual(cfg["target_modules"],
                         ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.assertEqual(cfg["task_type"], "CAUSAL_LM")
        self.assertEqual(cfg["r"], 16)
        self.assertEqual(cfg["lora_alpha"], 32)

    def test_有adapter时续训并解冻lora参数(self):
        with tempfile.TemporaryDirectory() as td:
            ad = os.path.join(td, "ckpt")
            os.makedirs(ad)
            with open(os.path.join(ad, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump({"r": 16}, f)
            model, mode = train_dpo.attach_trainable_adapter(
                _FakeModel(), ad, peft_mod=_FakePeft, lora_config_cls=_FakeLoraConfig)
        self.assertEqual(mode, "resume_sft")
        self.assertEqual([c[0] for c in _FakePeft.calls], ["from_pretrained"])
        frozen = {n: p.requires_grad for n, p in model.named_parameters()}
        self.assertTrue(frozen["lora_A.default.weight"], "lora 参数必须解冻，否则 DPOTrainer 报无可训练参数")
        self.assertFalse(frozen["base_layer.weight"], "基座参数必须保持冻结（4bit 量化基座不可训练）")

    def test_空路径也不崩(self):
        model, mode = train_dpo.attach_trainable_adapter(
            _FakeModel(), "", peft_mod=_FakePeft, lora_config_cls=_FakeLoraConfig)
        self.assertEqual(mode, "fresh_lora")


class TestCallSiteWiring(unittest.TestCase):
    def test_调用点必须用attach函数而非warn后继续(self):
        src = inspect.getsource(train_dpo.main)
        self.assertIn("attach_trainable_adapter(", src)
        # 旧实现的那句 warn 是「知道有问题但照跑」的标志，必须消失
        self.assertNotIn("从基座直做DPO", src)
        self.assertIn("[adapter] mode=%s", src)
        # 「挂完 adapter 仍无参数」必须当场抛，不许带着 0 个可训练张量进训练循环
        self.assertIn("if n_train == 0:", src)
        self.assertIn("raise RuntimeError", src)


if __name__ == "__main__":
    unittest.main()
