# -*- coding: utf-8 -*-
"""增量库向量索引：与peS2o基座同一embedding空间(all-MiniLM-L6-v2, 384维归一化IP).
输入 /data1/arxiv_incremental/arxiv.db -> 输出 arxiv.index + arxiv_ids.json
"""
import json
import os
import sqlite3

import faiss
import numpy as np

for _v in ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
           "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
    __import__("os").environ.pop(_v, None)
__import__("os").environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

INCR_DIR = "/data1/arxiv_incremental"
INCR_DB = os.path.join(INCR_DIR, "arxiv.db")
OUT_INDEX = os.path.join(INCR_DIR, "arxiv.index")
OUT_IDS = os.path.join(INCR_DIR, "arxiv_ids.json")

con = sqlite3.connect(INCR_DB)
rows = list(con.execute("SELECT arxiv_id, title, abstract FROM papers ORDER BY published DESC"))
con.close()
print("[rows] %d" % len(rows), flush=True)

import glob as _glob
_snap = _glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"))
model = SentenceTransformer(_snap[0] if _snap else "all-MiniLM-L6-v2")
texts = [((r[1] or "") + " " + (r[2] or ""))[:2000] for r in rows]
emb = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
emb = np.asarray(emb, dtype=np.float32)

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, OUT_INDEX)
with open(OUT_IDS, "w", encoding="utf-8") as f:
    json.dump([r[0] for r in rows], f)
print("[done] ntotal=%d dim=%d -> %s" % (index.ntotal, emb.shape[1], OUT_INDEX), flush=True)
