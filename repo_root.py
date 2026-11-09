"""Paths for a clone whose repository root is this package.

Older commands assumed a parent directory and a ``paper_agent/`` prefix.
External corpora lived on the original machine and are not in git. Those
absolute paths are named below so error text can point at the record.
They are not defaults.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

HISTORICAL_PES2O = "/data1/peS2o_kb_faiss"
HISTORICAL_ARXIV = "/data1/arxiv_incremental"
HISTORICAL_CORPUS = "/data3/Research/corpus_hf/all-papers.parquet"
HISTORICAL_VENV = "/data1/paper_agent_venv"


def repo_path(path):
    path = path.replace("\\", "/")
    if os.path.isabs(path):
        return path
    if path.startswith("paper_agent/"):
        path = path[len("paper_agent/"):]
    return os.path.join(ROOT, path)


def refuse_corpus_rebuild():
    """Exit 2 unless an external parquet is set and --write was passed.

    The builders overwrite committed jsonl. A fresh clone must not do that
    by falling through to a hardcoded machine path.
    """
    corpus = os.environ.get("PAPER_CORPUS_PARQUET", "").strip()
    if not corpus or "--write" not in sys.argv:
        print(
            "FAIL: refusing to overwrite committed jsonl. "
            "Set PAPER_CORPUS_PARQUET and pass --write.",
            file=sys.stderr,
        )
        print(
            "Historical parquet (not in this repo): %s" % HISTORICAL_CORPUS,
            file=sys.stderr,
        )
        sys.exit(2)
    return corpus
