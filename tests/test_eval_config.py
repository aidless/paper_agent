# -*- coding: utf-8 -*-
"""eval_config 的契约测试：BASE 路径与 4bit 开关必须可从环境覆盖。

为什么需要这两个开关：AMD/ROCm 实例上基座是本地 fp16 目录（HF 仓库名离线加载会失败），
且没有 bitsandbytes（4bit 加载起不来）。评测要能与历史 4bit 报告区分口径地跑完。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from eval_config import DEFAULT_BASE, resolve_base_model, use_4bit  # noqa: E402


def test_默认回落_HF_仓库名():
    assert resolve_base_model({}) == "Qwen/Qwen3-4B"
    assert DEFAULT_BASE == "Qwen/Qwen3-4B"


def test_env_非空时用本地路径_并去空白():
    assert resolve_base_model({"EVAL_BASE_MODEL": "/workspace/models/Qwen3-4B"}) == "/workspace/models/Qwen3-4B"
    assert resolve_base_model({"EVAL_BASE_MODEL": "  /workspace/models/Qwen3-4B \n"}) == "/workspace/models/Qwen3-4B"


def test_env_是空串或纯空白时仍回落默认值():
    assert resolve_base_model({"EVAL_BASE_MODEL": ""}) == DEFAULT_BASE
    assert resolve_base_model({"EVAL_BASE_MODEL": "   "}) == DEFAULT_BASE


def test_未设置时默认走_4bit_保持历史口径():
    assert use_4bit({}) is True


@pytest.mark.parametrize("raw", ["0", "false", "False", "no", "off"])
def test_显式关掉_4bit(raw):
    assert use_4bit({"EVAL_LOAD_4BIT": raw}) is False


@pytest.mark.parametrize("raw", ["1", "true", "yes"])
def test_显式开着_4bit(raw):
    assert use_4bit({"EVAL_LOAD_4BIT": raw}) is True


def test_空串当作未设_保持默认_4bit():
    """空值最可能来自「变量没传」而不是「我想关」——默认行为必须保持历史口径。

    （本条原与我第一版实现在 _FALSEY 里同时含 "" 相冲突："" 既算 falsey 又被
    `or "1"` 覆盖成默认值。契约就此钉死：只有明确 false 词才算关。）
    """
    assert use_4bit({"EVAL_LOAD_4BIT": ""}) is True
    assert use_4bit({"EVAL_LOAD_4BIT": "   "}) is True


def test_真实环境读取不炸():
    """模拟脚本侧用法：从 os.environ 直接算。"""
    os.environ["EVAL_BASE_MODEL"] = "/tmp/whatever"
    try:
        assert resolve_base_model(os.environ) == "/tmp/whatever"
    finally:
        os.environ.pop("EVAL_BASE_MODEL", None)
