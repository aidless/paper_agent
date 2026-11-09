# -*- coding: utf-8 -*-
"""Build the incremental arXiv FAISS index in the peS2o MiniLM space.

Requires ARXIV_INCREMENTAL (a directory that contains arxiv.db). The
historical original-machine path is /data1/arxiv_incremental and is not a
default. Missing input exits 2 before any embedding import.
"""
import os
import sys


def main():
    incr = os.environ.get("ARXIV_INCREMENTAL", "").strip()
    db = os.path.join(incr, "arxiv.db") if incr else ""
    if not incr or not os.path.isdir(incr) or not os.path.exists(db):
        print("FAIL: ARXIV_INCREMENTAL is unset or has no arxiv.db.", file=sys.stderr)
        print("Historical path (not in this repo): /data1/arxiv_incremental", file=sys.stderr)
        return 2

    import json
    import sqlite3

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    for _v in ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        os.environ.pop(_v, None)
    os.environ["HF_HUB_OFFLINE"] = "1"

    out_index = os.path.join(incr, "arxiv.index")
    out_ids = os.path.join(incr, "arxiv_ids.json")
    con = sqlite3.connect(db)
    rows = list(con.execute("SELECT arxiv_id, title, abstract FROM papers ORDER BY published DESC"))
    con.close()
    print("[rows] %d" % len(rows), flush=True)

    import glob as _glob
    snap = _glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"))
    model = SentenceTransformer(snap[0] if snap else "all-MiniLM-L6-v2")
    texts = [((r[1] or "") + " " + (r[2] or ""))[:2000] for r in rows]
    emb = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, out_index)
    with open(out_ids, "w", encoding="utf-8") as handle:
        json.dump([r[0] for r in rows], handle)
    print("[done] ntotal=%d dim=%d -> %s" % (index.ntotal, emb.shape[1], out_index), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
