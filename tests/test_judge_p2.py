"""judge_p2 的判据测试：先红后绿。

四条判据来自 `notes/2026-11-20-dpo-eval-on-amd.md` 的「判据先声明」表，
是看到读数之前写下的。这里把它们钉成断言。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from judge_p2 import JudgeError, judge  # noqa: E402

REPORT = json.loads(
    (Path(__file__).resolve().parents[1] / "reports" / "eval_3way_p2.json").read_text(encoding="utf-8")
)


def _report(p1=None, p2=None):
    """造一个最小报告；None 的字段用真实报告的读数填。"""
    base = {
        "ckpt136": dict(REPORT["ckpt136"]),
        "tmlr-qwen3-4b-p2": dict(REPORT["tmlr-qwen3-4b-p2"]),
    }
    if p1:
        base["ckpt136"].update(p1)
    if p2:
        base["tmlr-qwen3-4b-p2"].update(p2)
    return base


def test_真实报告四条全过():
    verdict = judge(REPORT)
    assert verdict["passed"] is True
    assert len(verdict["rows"]) == 4
    assert all(r["passed"] for r in verdict["rows"])


def test_nll退化必须判红():
    report = _report(p2={"val_nll_60": 2.9})  # 高于 2.8648
    verdict = judge(report)
    assert verdict["passed"] is False
    assert [r["passed"] for r in verdict["rows"]] == [False, True, True, True]


def test_证据绑定率下降必须判红():
    report = _report(p2={"gen": dict(REPORT["tmlr-qwen3-4b-p2"]["gen"], evd=0.5)})
    verdict = judge(report)
    assert verdict["passed"] is False
    assert [r["passed"] for r in verdict["rows"]] == [True, False, True, True]


def test_distinct2下降必须判红():
    report = _report(p2={"gen": dict(REPORT["tmlr-qwen3-4b-p2"]["gen"], d2=0.4)})
    verdict = judge(report)
    assert verdict["passed"] is False
    assert [r["passed"] for r in verdict["rows"]] == [True, True, False, True]


def test_循环复读增加必须判红():
    report = _report(p2={"gen": dict(REPORT["tmlr-qwen3-4b-p2"]["gen"], loop=8)})
    verdict = judge(report)
    assert verdict["passed"] is False
    assert [r["passed"] for r in verdict["rows"]] == [True, True, True, False]


def test_持平算过而不是退步():
    # evd 0.75 == 0.75、loop 3 == 3 都是持平，判据写的是 >= / <=
    report = _report(p1={"val_nll_60": 2.7079, "gen": {**REPORT["ckpt136"]["gen"], "evd": 0.75, "d2": 0.5592, "loop": 3}})
    assert judge(report)["passed"] is True


def test_缺字段要报错不是静默通过():
    report = _report(p2={"val_nll_60": None})
    with pytest.raises(JudgeError, match="val_nll_60"):
        judge(report)


def test_缺模型段要报错():
    with pytest.raises(JudgeError, match="没有模型段"):
        judge({"ckpt136": REPORT["ckpt136"]})


def test_缺_gen子字段要报错():
    p2 = dict(REPORT["tmlr-qwen3-4b-p2"])
    p2["gen"] = {"tag": 0.0}
    with pytest.raises(JudgeError, match="gen.evd"):
        judge(_report(p2=p2))


def test_布尔不是数值证据():
    with pytest.raises(JudgeError, match="布尔"):
        judge(_report(p2={"val_nll_60": True}))


def test_cli_全过返回0():
    out = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "eval" / "judge_p2.py"), "reports/eval_3way_p2.json"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "4/4" in out.stdout


def test_cli_有判据不过返回1(tmp_path):
    bad = _report(p2={"val_nll_60": 3.5})
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "eval" / "judge_p2.py"), str(p)],
        capture_output=True, text=True,
    )
    assert out.returncode == 1
    assert "FAIL" in out.stdout


def test_cli_坏报告返回2(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not json", encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "eval" / "judge_p2.py"), str(p)],
        capture_output=True, text=True,
    )
    assert out.returncode == 2


def test_cli_json输出可解析():
    out = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "eval" / "judge_p2.py"),
         "reports/eval_3way_p2.json", "--json"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0
    payload = json.loads(out.stdout)
    assert payload["passed"] is True
    assert len(payload["rows"]) == 4


def test_cli_缺字段返回2(tmp_path):
    """结构错（缺字段）必须 exit 2，不能当「四条全过」放行。"""
    # 注意：不能用 _report(p2=p2)，因为 dict.update 不会删键 —— 夹具要显式剔除
    p2 = {k: v for k, v in REPORT["tmlr-qwen3-4b-p2"].items() if k != "val_nll_60"}
    bad = {"ckpt136": REPORT["ckpt136"], "tmlr-qwen3-4b-p2": p2}
    p = tmp_path / "missing.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "eval" / "judge_p2.py"), str(p)],
        capture_output=True, text=True,
    )
    assert out.returncode == 2, out.stdout + out.stderr
