"""把「跑前锁定的四条判据」钉成机械判定，而不是人读表时挑好看的看。

先写测试（tests/test_judge_p2.py），再写这个实现。
"""

from __future__ import annotations

import argparse
import json
import sys

CRITERIA: tuple[tuple[str, tuple[str, ...], tuple[str, ...], str], ...] = (
    ("val NLL 不退化", ("val_nll_60",), ("val_nll_60",), "le"),
    ("证据绑定率不退化", ("gen", "evd"), ("gen", "evd"), "ge"),
    ("distinct-2 不下降", ("gen", "d2"), ("gen", "d2"), "ge"),
    ("循环复读不增加", ("gen", "loop"), ("gen", "loop"), "le"),
)


class JudgeError(Exception):
    """报告形状不满足判据所需（缺键、非数字、缺模型段）。"""


def _dig(obj, path: tuple[str, ...]):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise JudgeError(f"报告缺少字段: {'.'.join(path)}")
        cur = cur[key]
    return cur


def judge(report, baseline: str = "ckpt136", candidate: str = "tmlr-qwen3-4b-p2") -> dict:
    if not isinstance(report, dict):
        raise JudgeError("报告顶层必须是对象")
    for name in (baseline, candidate):
        if name not in report:
            raise JudgeError(f"报告里没有模型段: {name}")
    rows = []
    for label, cand_path, base_path, direction in CRITERIA:
        cand_value = _dig(report[candidate], cand_path)
        base_value = _dig(report[baseline], base_path)
        if isinstance(cand_value, bool) or isinstance(base_value, bool):
            raise JudgeError(f"{label}: 布尔不是数值证据")
        cand_path_s = ".".join(cand_path)
        if not isinstance(cand_value, (int, float)) or not isinstance(base_value, (int, float)):
            raise JudgeError(f"{label}: {cand_path_s} 取值不是数字（{cand_value!r} vs {base_value!r}）")
        if direction == "le":
            ok = cand_value <= base_value
        elif direction == "ge":
            ok = cand_value >= base_value
        else:  # pragma: no cover
            raise JudgeError(f"未知比较方向: {direction}")
        rows.append(
            {
                "criterion": label,
                "candidate": cand_value,
                "baseline": base_value,
                "direction": direction,
                "passed": ok,
            }
        )
    return {"rows": rows, "passed": all(r["passed"] for r in rows)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="按跑前锁定的四条判据机械判定 DPO 评测报告")
    ap.add_argument("report")
    ap.add_argument("--baseline", default="ckpt136")
    ap.add_argument("--candidate", default="tmlr-qwen3-4b-p2")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        with open(args.report, encoding="utf-8") as fh:
            report = json.load(fh)
        verdict = judge(report, baseline=args.baseline, candidate=args.candidate)
    except JudgeError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"[ERR] 报告不是合法 JSON: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"[ERR] 读不到报告: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
    else:
        print(f"基线={args.baseline}  候选={args.candidate}")
        for row in verdict["rows"]:
            mark = "PASS" if row["passed"] else "FAIL"
            print(f"  {mark}  {row['criterion']:18s} 候选={row['candidate']!r:>10} "
                  f"基线={row['baseline']!r:>10} ({row['direction']})")
        n = sum(1 for r in verdict["rows"] if r["passed"])
        tail = "四条全过，DPO 在 SFT 之上有正收益" if verdict["passed"] else "有判据未满足"
        print(f"判决: {n}/{len(verdict['rows'])} — {tail}")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
