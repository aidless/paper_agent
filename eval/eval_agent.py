# -*- coding: utf-8 -*-
"""Score the frozen fixture. Does not invent rows or a grounded rate.

task_success in the committed report is the mean of pred_ok in
data/eval_fixed.jsonl. It is not a model score. grounded_rate needs an
external sqlite file:

  PE_S2O_KB_FAISS/papers.db     historical: /data1/peS2o_kb_faiss/papers.db
  ARXIV_INCREMENTAL/arxiv.db    historical: /data1/arxiv_incremental/arxiv.db

If neither file is present this command exits 2 and does not rewrite
reports/eval_results.json.
"""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from repo_root import repo_path

EVAL_FILE = repo_path("data/eval_fixed.jsonl")
OUT_JSON = repo_path("reports/eval_results.json")
OUT_MD = repo_path("reports/eval.md")


def _db_paths():
    found = []
    base = os.environ.get("PE_S2O_KB_FAISS", "").strip()
    incr = os.environ.get("ARXIV_INCREMENTAL", "").strip()
    if base:
        path = os.path.join(base, "papers.db")
        if os.path.exists(path):
            found.append(path)
    if incr:
        path = os.path.join(incr, "arxiv.db")
        if os.path.exists(path):
            found.append(path)
    return found


def grounded(title, dbs):
    if not title:
        return False
    q = "%" + title[:20] + "%"
    for db in dbs:
        try:
            con = sqlite3.connect(db)
            hit = con.execute("SELECT 1 FROM papers WHERE title LIKE ? LIMIT 1", (q,)).fetchone()
            con.close()
            if hit:
                return True
        except Exception:
            pass
    return False


def main():
    dbs = _db_paths()
    if not dbs:
        print(
            "FAIL: grounded_rate needs PE_S2O_KB_FAISS/papers.db or ARXIV_INCREMENTAL/arxiv.db.",
            file=sys.stderr,
        )
        print(
            "Historical paths (not in this repo): "
            "/data1/peS2o_kb_faiss/papers.db and /data1/arxiv_incremental/arxiv.db",
            file=sys.stderr,
        )
        print("Refusing to rewrite reports/eval_results.json.", file=sys.stderr)
        return 2
    if not os.path.exists(EVAL_FILE):
        print("FAIL: missing data/eval_fixed.jsonl; not inventing placeholder rows.", file=sys.stderr)
        return 2
    items = []
    with open(EVAL_FILE, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    n = len(items)
    ok = sum(1 for x in items if x.get("pred_ok"))
    gr = sum(1 for x in items if grounded(x.get("title", ""), dbs))
    res = {
        "n": n,
        "task_success": ok / float(n or 1),
        "grounded_rate": gr / float(n or 1),
        "note": "task_success is the pred_ok mean of the fixture, not a model score. grounded_rate was computed from the external sqlite files configured in this run.",
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(res, handle, ensure_ascii=False, indent=2)
    with open(OUT_MD, "w", encoding="utf-8") as handle:
        handle.write(
            "# Eval报告\n\nn=%d, success=%.3f, grounded=%.3f\n\n%s\n"
            % (n, res["task_success"], res["grounded_rate"], res["note"])
        )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
