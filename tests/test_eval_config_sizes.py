"""N_GEN / SEED 必须可配置的契约测试（先红后绿）。

存在理由：8 条生成样本下 evd=6/8、loop=3/8，一个样本翻转就是 12.5 个百分点，
三条判据只能当方向性信号。扩样本要在不改脚本结构的前提下能调，
且**默认值必须仍是 8 / 42**，否则历史报告与新报告口径不可比。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from eval_config import DEFAULT_N_GEN, DEFAULT_SEED, resolve_n_gen, resolve_seed  # noqa: E402


def test_默认值与历史口径一致():
    assert DEFAULT_N_GEN == 8
    assert DEFAULT_SEED == 42
    assert resolve_n_gen({}) == 8
    assert resolve_seed({}) == 42


def test_空串与纯空白当作未设():
    assert resolve_n_gen({"EVAL_N_GEN": ""}) == 8
    assert resolve_n_gen({"EVAL_N_GEN": "   "}) == 8
    assert resolve_seed({"EVAL_SEED": " "}) == 42


def test_可显式扩大():
    assert resolve_n_gen({"EVAL_N_GEN": "40"}) == 40
    assert resolve_n_gen({"EVAL_N_GEN": " 20 "}) == 20


def test_可显式换种子():
    assert resolve_seed({"EVAL_SEED": "123"}) == 123


@pytest.mark.parametrize("raw", ["0", "-1", "-8"])
def test_非正数必须拒绝而不是静默返回0(raw):
    with pytest.raises(ValueError, match="EVAL_N_GEN"):
        resolve_n_gen({"EVAL_N_GEN": raw})


def test_非整数必须拒绝():
    with pytest.raises(ValueError, match="EVAL_N_GEN"):
        resolve_n_gen({"EVAL_N_GEN": "8.5"})
    with pytest.raises(ValueError, match="EVAL_N_GEN"):
        resolve_n_gen({"EVAL_N_GEN": "abc"})


def test_种子非整数必须拒绝():
    with pytest.raises(ValueError, match="EVAL_SEED"):
        resolve_seed({"EVAL_SEED": "1.5"})
