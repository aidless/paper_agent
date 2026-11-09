# -*- coding: utf-8 -*-
"""Dual retrieval: read-only peS2o plus an incremental arXiv corpus.

Directories come from the environment and are not defaulted to the original
machine:

  PE_S2O_KB_FAISS     historical path, not in this repo: /data1/peS2o_kb_faiss
  ARXIV_INCREMENTAL   historical path, not in this repo: /data1/arxiv_incremental

The CLI exits 2 when either directory is missing. Empty hits are not a
successful search. A historical note recorded papers.index at 541744 vectors;
this file does not re-count the index.
"""
import json
import os
import sqlite3
import sys


def _base():
    return os.environ.get("PE_S2O_KB_FAISS", "").strip()


def _incr():
    return os.environ.get("ARXIV_INCREMENTAL", "").strip()


def _incr_db():
    root = _incr()
    return os.path.join(root, "arxiv.db") if root else ""


def _has_index(root, names):
    return bool(root) and any(os.path.exists(os.path.join(root, name)) for name in names)


def index_problem():
    """Configuration errors for a real fused search. Empty means both indexes exist."""
    problems = []
    base = _base()
    incr = _incr()
    if not base:
        problems.append(
            "PE_S2O_KB_FAISS unset (historical, not in repo: /data1/peS2o_kb_faiss)"
        )
    elif not os.path.isdir(base):
        problems.append("PE_S2O_KB_FAISS is not a directory: %s" % base)
    elif not _has_index(base, ("papers.db", "papers.index")):
        problems.append("PE_S2O_KB_FAISS has neither papers.db nor papers.index")
    if not incr:
        problems.append(
            "ARXIV_INCREMENTAL unset (historical, not in repo: /data1/arxiv_incremental)"
        )
    elif not os.path.isdir(incr):
        problems.append("ARXIV_INCREMENTAL is not a directory: %s" % incr)
    elif not _has_index(incr, ("arxiv.db", "arxiv.index")):
        problems.append("ARXIV_INCREMENTAL has neither arxiv.db nor arxiv.index")
    return problems


def _tokens(query):
    toks = [t.strip().lower() for t in query.replace(",", " ").split() if len(t.strip()) >= 3]
    return toks[:4] if toks else [query]


def search_pes2o(query, top_k=5):
    root = _base()
    db = os.path.join(root, "papers.db") if root else ""
    if not db or not os.path.exists(db):
        return []
    con = sqlite3.connect(db)
    out = []
    try:
        # papers表: 至少有title字段; 用LIKE做兜底, 向量索引由kb_search提供
        # 长查询先分词OR检索, 避免整句LIKE零命中
        for tok in _tokens(query):
            for r in con.execute(
                "SELECT rowid, title FROM papers WHERE title LIKE ? LIMIT ?",
                ("%" + tok + "%", top_k),
            ):
                out.append({"id": "pes2o:%s" % r[0], "title": r[1], "source": "pes2o", "score": 0.5})
                if len(out) >= top_k:
                    break
            if len(out) >= top_k:
                break
    except Exception:
        pass
    finally:
        con.close()
    return out


def search_incr(query, top_k=5):
    db = _incr_db()
    if not db or not os.path.exists(db):
        return []
    con = sqlite3.connect(db)
    out = []
    try:
        rows = list(con.execute(
            "SELECT arxiv_id,title,authors,published,url FROM papers WHERE title LIKE ? OR abstract LIKE ? ORDER BY published DESC LIMIT ?",
            ("%" + query + "%", "%" + query + "%", top_k)))
        for r in rows:
            out.append({"id": r[0], "title": r[1], "authors": r[2], "published": r[3], "url": r[4], "source": "arxiv", "score": 0.8})
    except Exception:
        pass
    finally:
        con.close()
    return out


def _st_model():
    """共享embedding模型(基座本地副本优先, 与两路索引同一空间)."""
    if not hasattr(_st_model, "cache"):
        _st_model.cache = "init"
    if _st_model.cache == "init":
        try:
            import glob
            for _v in ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
                       "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
                os.environ.pop(_v, None)
            os.environ["HF_HUB_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            root = _base()
            local = os.path.join(root, "models", "all-MiniLM-L6-v2") if root else ""
            if not local or not os.path.isdir(local):
                snap = glob.glob(os.path.expanduser(
                    "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"))
                local = snap[0] if snap else "all-MiniLM-L6-v2"
            _st_model.cache = SentenceTransformer(local, device="cpu")
        except Exception:
            _st_model.cache = None
    return _st_model.cache


def search_pes2o_vector(query, top_k=5):
    """基座向量检索. 只读, CPU. 缺索引时返回空, 不把空当成命中."""
    if not hasattr(search_pes2o_vector, "cache"):
        search_pes2o_vector.cache = "init"
    if search_pes2o_vector.cache == "init":
        try:
            import faiss
            import numpy as np
            root = _base()
            idx_path = os.path.join(root, "papers.index") if root else ""
            ids_path = os.path.join(root, "paper_ids.txt") if root else ""
            if not (idx_path and os.path.exists(idx_path) and os.path.exists(ids_path)):
                search_pes2o_vector.cache = None
            else:
                with open(ids_path, encoding="utf-8", errors="ignore") as f:
                    ids = [l.strip() for l in f]
                search_pes2o_vector.cache = {
                    "index": faiss.read_index(idx_path), "ids": ids, "np": np}
        except Exception:
            search_pes2o_vector.cache = None
    res = search_pes2o_vector.cache
    model = _st_model()
    if res is None or model is None:
        return []
    try:
        q = model.encode([query], normalize_embeddings=True)
        q = res["np"].asarray(q, dtype=res["np"].float32)
        scores, idxs = res["index"].search(q, top_k)
        root = _base()
        con = sqlite3.connect(os.path.join(root, "papers.db"))
        out = []
        try:
            for s, i in zip(scores[0].tolist(), idxs[0].tolist()):
                if i < 0 or i >= len(res["ids"]):
                    continue
                key = res["ids"][i]
                r = con.execute(
                    "SELECT paper_id,title,authors,year FROM papers WHERE id=?", (key,)).fetchone()
                if r is None:
                    r = con.execute(
                        "SELECT paper_id,title,authors,year FROM papers WHERE paper_id=?", (key,)).fetchone()
                if r:
                    out.append({"id": "pes2o:%s" % r[0], "title": r[1],
                                "authors": r[2], "published": r[3],
                                "source": "pes2o-vector", "score": round(float(s), 4)})
        finally:
            con.close()
        return out
    except Exception:
        return []


def _vec_resources():
    """惰性加载增量向量索引(与peS2o基座同一MiniLM-384空间). 缺文件时返回None."""
    if not hasattr(_vec_resources, "cache"):
        _vec_resources.cache = "init"
    if _vec_resources.cache == "init":
        try:
            import faiss
            import numpy as np
            for _v in ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
                       "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
                os.environ.pop(_v, None)
            os.environ["HF_HUB_OFFLINE"] = "1"
            root = _incr()
            idx_path = os.path.join(root, "arxiv.index") if root else ""
            ids_path = os.path.join(root, "arxiv_ids.json") if root else ""
            if not (idx_path and os.path.exists(idx_path) and os.path.exists(ids_path)):
                _vec_resources.cache = None
            else:
                _vec_resources.cache = {
                    "index": faiss.read_index(idx_path),
                    "ids": json.load(open(ids_path, encoding="utf-8")),
                    "np": np,
                }
        except Exception:
            _vec_resources.cache = None
    return _vec_resources.cache


def search_incr_vector(query, top_k=5):
    """增量库向量检索: cosine(归一化IP), 与基座同一空间, 分数可比."""
    res = _vec_resources()
    model = _st_model()
    if res is None or model is None:
        return []
    db = _incr_db()
    if not db:
        return []
    con = sqlite3.connect(db)
    try:
        q = model.encode([query], normalize_embeddings=True)
        q = res["np"].asarray(q, dtype=res["np"].float32)
        scores, idxs = res["index"].search(q, top_k)
        out = []
        for s, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i < 0:
                continue
            aid = res["ids"][i]
            r = con.execute(
                "SELECT arxiv_id,title,authors,published,url FROM papers WHERE arxiv_id=?",
                (aid,)).fetchone()
            if r:
                out.append({"id": r[0], "title": r[1], "authors": r[2],
                            "published": r[3], "url": r[4],
                            "source": "arxiv-vector", "score": round(float(s), 4)})
        return out
    except Exception:
        return []
    finally:
        con.close()


def fused_search(query, top_k=10, recency_boost=0.0, fresh_year="2026"):
    """四路融合. recency_boost: 给fresh_year及以后增量命中的加分(可调启发式, 默认0即纯相关性).
    诚实声明: boost是拍脑袋常数, 非学得权重; 调参需在标注相关性集上验证, 目前仅报告测量值。
    调用方在索引未配置时应该先看 index_problem()，不要把 [] 读成零命中。"""
    seen, out = set(), []
    cands = (search_incr_vector(query, top_k) + search_incr(query, top_k)
             + search_pes2o_vector(query, top_k) + search_pes2o(query, top_k))
    for r in cands:
        key = (r.get("title") or "").strip().lower()
        if key and key in seen:
            continue
        seen.add(key)
        if recency_boost and r.get("source", "").startswith("arxiv") \
                and (r.get("published") or "") >= fresh_year:
            r = dict(r, score=round(r.get("score", 0) + recency_boost, 4))
        out.append(r)
    out.sort(key=lambda x: x.get("score", 0), reverse=True)
    return out[:top_k]


if __name__ == "__main__":
    problems = index_problem()
    if problems:
        for item in problems:
            print("FAIL: %s" % item, file=sys.stderr)
        sys.exit(2)
    q = sys.argv[1] if len(sys.argv) > 1 else "LLM agent"
    print(json.dumps(fused_search(q), ensure_ascii=False, indent=2))
