# -*- coding: utf-8 -*-
"""eval_3way 的加载配置解析（纯函数，可单测）。

历史约束：eval_3way.py 在 import 时就会 from_pretrained 建 tokenizer，无法被测试直接 import。
所以把「BASE 路径从哪来」「要不要 4bit」这两件必须能验的事抽到这里，
脚本只调用、不再内联判断。
"""
from __future__ import annotations

from typing import Mapping

DEFAULT_BASE = "Qwen/Qwen3-4B"
_FALSEY = frozenset({"0", "false", "no", "off"})


def resolve_base_model(env: Mapping[str, str]) -> str:
    """EVAL_BASE_MODEL 非空则用它（strip），否则回落 HF 仓库名。

    存在理由：AMD 实例上基座是本地 fp16 目录 /workspace/models/Qwen3-4B，
    不能走 HF 仓库名（离线加载会失败），但默认行为必须保持历史口径不变。
    """
    value = (env.get("EVAL_BASE_MODEL") or "").strip()
    return value or DEFAULT_BASE


def use_4bit(env: Mapping[str, str]) -> bool:
    """EVAL_LOAD_4BIT 决定是否 bnb 4bit 加载；默认 True 保持与历史报告同口径。

    AMD/ROCm 上没有 bitsandbytes，必须能显式关掉改走 bf16 全量加载，
    否则脚本在那边根本起不来。
    """
    raw = (env.get("EVAL_LOAD_4BIT") or "").strip().lower()
    if not raw:  # 未设置 / 空串 / 纯空白：都当「没传」，保持历史 4bit 口径
        return True
    return raw not in _FALSEY

DEFAULT_N_GEN = 8
DEFAULT_SEED = 42


def _positive_int(env, key, default):
    """读一个正整数配置：未设/空串回落 default，非正或非整数一律抛 ValueError。

    存在理由：扩样本是要能调的，但「0 条」或「-8 条」不是配置而是错误，
    静默返回 0 会让评测跑出一个分母为零的报告。
    """
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{key} 必须是整数，收到 {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{key} 必须是正整数，收到 {value}")
    return value


def resolve_n_gen(env: Mapping[str, str]) -> int:
    """生成样本条数（EVAL_N_GEN）；默认 8 与历史报告同口径。"""
    return _positive_int(env, "EVAL_N_GEN", DEFAULT_N_GEN)


def resolve_seed(env: Mapping[str, str]) -> int:
    """取样种子（EVAL_SEED）；默认 42 与历史报告同口径。"""
    return _positive_int(env, "EVAL_SEED", DEFAULT_SEED)
