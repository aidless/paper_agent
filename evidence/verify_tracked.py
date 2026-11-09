#!/usr/bin/env python3
"""Verify evidence that is expected to work from a fresh Git clone.

Also checks that committed eval JSON is internally consistent and that the
README / P1 verdict still quote those numbers. Token NLL is not recomputed:
only the stored scalar is locked, and generation aggregates are recomputed
from gen_detail.

Length uses the same rule as eval/eval_3way.py: int(round(mean, 4)), which
truncates. 418.625 stays 418. Do not "correct" that to 419.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_files():
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [ROOT / entry.decode() for entry in result.stdout.split(b"\0") if entry]


def _mean(rows, key):
    return sum(row[key] for row in rows) / float(len(rows))


def check_eval_json(path, failures):
    data = json.loads(path.read_text(encoding="utf-8"))
    for key, block in data.items():
        if not isinstance(block, dict) or key.startswith("_"):
            continue
        if "gen" not in block or "gen_detail" not in block:
            failures.append("%s key %s missing gen/gen_detail" % (path.name, key))
            continue
        detail = block["gen_detail"]
        if not detail:
            failures.append("%s %s empty gen_detail" % (path.name, key))
            continue
        gen = block["gen"]
        expected = {
            "tag": round(_mean(detail, "tag"), 4),
            "evd": round(_mean(detail, "evd"), 4),
            "d2": round(_mean(detail, "d2"), 4),
            "loop": int(sum(bool(row["loop"]) for row in detail)),
            "len": int(round(_mean(detail, "len"), 4)),
        }
        for name, exp in expected.items():
            got = gen[name]
            if isinstance(exp, float):
                if abs(float(got) - exp) > 5e-5:
                    failures.append(
                        "%s %s gen.%s %s != recomputed %s"
                        % (path.name, key, name, got, exp)
                    )
            elif int(got) != exp:
                failures.append(
                    "%s %s gen.%s %s != recomputed %s"
                    % (path.name, key, name, got, exp)
                )
        if "val_nll_60" not in block:
            failures.append("%s %s missing val_nll_60 scalar" % (path.name, key))
    return data


def _require(text, needle, failures, label):
    if needle not in text:
        failures.append("%s missing %r" % (label, needle))


def check_sibling_markdown(json_path, data, failures):
    md_path = json_path.with_suffix(".md")
    if not md_path.is_file():
        failures.append("missing markdown sibling %s" % md_path.relative_to(ROOT))
        return
    text = md_path.read_text(encoding="utf-8")
    for key, block in data.items():
        if not isinstance(block, dict) or key.startswith("_"):
            continue
        nll = "%.4f" % block["val_nll_60"]
        _require(text, nll, failures, md_path.name)
        evd = float(block["gen"]["evd"])
        # Two-decimal cells hide 0.125 as 0.12. Require the 4-decimal form
        # whenever rounding to 2 decimals is not the same value.
        if abs(evd - round(evd + 0.0, 2)) > 1e-6:
            token = ("%.4f" % evd).rstrip("0").rstrip(".")
            if token not in text and ("%.4f" % evd) not in text:
                failures.append(
                    "%s evidence rate %s truncated away from JSON %s"
                    % (md_path.name, token, evd)
                )


def check_readme(failures, p1_eval):
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"<!-- claims:eval_3way.json\n(.*?)-->", text, re.S)
    if not match:
        failures.append("README missing claims:eval_3way.json block")
        return
    claims = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        claims[key.strip()] = value.strip()
    expected = {
        "base.val_nll_60": "%.4f" % p1_eval["base"]["val_nll_60"],
        "ckpt270.val_nll_60": "%.4f" % p1_eval["ckpt270"]["val_nll_60"],
        "p1_gate": "FAIL",
    }
    for key, value in expected.items():
        if claims.get(key) != value:
            failures.append(
                "README claim %s=%r != %s" % (key, claims.get(key), value)
            )
    visible = text.split("<!--", 1)[0]
    for value in expected.values():
        if value not in visible:
            failures.append("README problem statement missing %s" % value)
    problem = re.search(r"^## Problem\n(.*?)(?=^## |\n<!--)", text, re.S | re.M)
    if not problem:
        failures.append("README missing ## Problem section")
    else:
        lines = [line for line in problem.group(1).splitlines() if line.strip()]
        if len(lines) > 3:
            failures.append("README problem is %d lines; limit is 3" % len(lines))
    if "STATUS: WORK-IN-PROGRESS" not in visible or "not A yet" not in visible:
        failures.append("README must keep STATUS: WORK-IN-PROGRESS — not A yet")
    if re.search(r"STATUS:\s*A\b", visible):
        failures.append("README claims STATUS A")
    if "does not reverse" not in text:
        failures.append("README must say the later v2 NLL does not reverse P1 FAIL")
    modes = (ROOT / "FAILURE_MODES.md").read_text(encoding="utf-8")
    for value in (expected["base.val_nll_60"], expected["ckpt270.val_nll_60"], "FAIL"):
        if value not in modes:
            failures.append("FAILURE_MODES.md missing %s" % value)
    if "PE_S2O_KB_FAISS" not in modes or "ARXIV_INCREMENTAL" not in modes:
        failures.append("FAILURE_MODES.md must name the external index variables")
    v2 = json.loads((ROOT / "reports" / "eval_3way_v2.json").read_text(encoding="utf-8"))
    allowed = {
        "%.4f" % p1_eval["base"]["val_nll_60"],
        "%.4f" % p1_eval["ckpt68"]["val_nll_60"],
        "%.4f" % p1_eval["ckpt270"]["val_nll_60"],
        "%.4f" % v2["base"]["val_nll_60"],
        "%.4f" % v2["ckpt136"]["val_nll_60"],
        "%.4f" % v2["ckpt_final"]["val_nll_60"],
    }
    found = set(re.findall(r"\d+\.\d+", text))
    extra = sorted(found - allowed)
    if extra:
        failures.append("README decimals absent from committed eval JSON: %s" % extra)


def check_p1_verdict(failures, p1_eval):
    text = (ROOT / "reports" / "P1_VERDICT.md").read_text(encoding="utf-8")
    _require(text, "门禁判定：FAIL", failures, "P1_VERDICT.md")
    _require(text, "不构成改判", failures, "P1_VERDICT.md")
    if "门禁判定：PASS" in text or "门禁判定：通过" in text:
        failures.append("P1_VERDICT.md reverses the FAIL gate")
    for key in ("base", "ckpt68", "ckpt270"):
        rendered = "%.4f" % p1_eval[key]["val_nll_60"]
        _require(text, rendered, failures, "P1_VERDICT.md")
    if text.count("%.4f" % p1_eval["ckpt270"]["val_nll_60"]) < 2:
        failures.append("P1_VERDICT.md must quote both checkpoint NLL cells")
    if "17263" in text:
        failures.append(
            "P1_VERDICT.md must keep its 2026-09-10 arXiv count distinct from the later fingerprint"
        )
    _require(text, "16902", failures, "P1_VERDICT.md")


def check_dated_counts(failures):
    fingerprint = json.loads(
        (ROOT / "evidence" / "env_fingerprint.json").read_text(encoding="utf-8")
    )
    counts = fingerprint["corpus_row_counts"]
    if counts.get("arxiv_incremental") != 17263:
        failures.append("env_fingerprint arXiv row count is no longer the 2026-10-08 snapshot")
    if counts.get("peS2o_base") != 541745:
        failures.append("env_fingerprint peS2o row count is no longer the 2026-10-08 snapshot")
    upgrade = (ROOT / "reports" / "RETRIEVAL_UPGRADE.md").read_text(encoding="utf-8")
    if "541744" not in upgrade:
        failures.append(
            "RETRIEVAL_UPGRADE.md lost the historical 541744 vector count (do not collapse it into the row count)"
        )


def check_fixture_success(failures):
    rows = []
    with (ROOT / "data" / "eval_fixed.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rate = sum(bool(row.get("pred_ok")) for row in rows) / float(len(rows))
    reported = json.loads(
        (ROOT / "reports" / "eval_results.json").read_text(encoding="utf-8")
    )
    if abs(float(reported["task_success"]) - rate) > 1e-9:
        failures.append(
            "eval_results.json task_success %s != pred_ok mean %s"
            % (reported["task_success"], rate)
        )
    md = (ROOT / "reports" / "eval.md").read_text(encoding="utf-8")
    if "0.625" not in md:
        failures.append("eval.md lost the fixture task_success 0.625")
    if "not a model score" not in md:
        failures.append("eval.md must say task_success is not a model score")
    if "not recomputed offline" not in md:
        failures.append("eval.md must say grounded_rate is not recomputed offline")


def check_tracked_and_manifest(failures):
    tracked = tracked_files()
    for path in tracked:
        if not path.is_file():
            failures.append("missing tracked file: %s" % path.relative_to(ROOT))
            continue
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                failures.append("invalid JSON %s: %s" % (path.relative_to(ROOT), exc))

    manifest = ROOT / "evidence" / "MANIFEST.sha256"
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            failures.append("invalid manifest line %d" % number)
            continue
        target = manifest.parent / relative
        if not target.is_file():
            failures.append("manifest target missing: evidence/%s" % relative)
        elif sha256(target) != expected:
            failures.append("manifest hash mismatch: evidence/%s" % relative)
    return tracked


def main():
    failures = []
    tracked = check_tracked_and_manifest(failures)
    p1_eval = None
    reports = ROOT / "reports"
    for path in sorted(reports.glob("eval_3way*.json")):
        data = check_eval_json(path, failures)
        check_sibling_markdown(path, data, failures)
        if path.name == "eval_3way.json":
            p1_eval = data
    if p1_eval is None:
        failures.append("missing reports/eval_3way.json")
    else:
        check_readme(failures, p1_eval)
        check_p1_verdict(failures, p1_eval)
    check_dated_counts(failures)
    check_fixture_success(failures)

    if failures:
        for failure in failures:
            print("FAIL: %s" % failure)
        return 1

    json_count = sum(path.suffix == ".json" for path in tracked)
    print(
        "PASS: %d tracked files present; %d JSON files parse; evidence manifest matches; "
        "eval gen_detail aggregates match; README claims match eval_3way.json; P1 gate remains FAIL"
        % (len(tracked), json_count)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
