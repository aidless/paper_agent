# -*- coding: utf-8 -*-
"""双路融合检索: peS2o只读基座 + arXiv增量库."""
import os, sqlite3, json

BASE_DIR = "/data1/peS2o_kb_faiss"
INCR_DIR = "/data1/arxiv_incremental"
INCR_DB = os.path.join(INCR_DIR, "arxiv.db")

def _tokens(query):
    toks = [t.strip().lower() for t in query.replace(",", " ").split() if len(t.strip()) >= 3]
    return toks[:4] if toks else [query]

def search_pes2o(query, top_k=5):
    db = os.path.join(BASE_DIR, "papers.db")
    if not os.path.exists(db):
        return []
    con = sqlite3.connect(db)
    out = []
    try:
        # papers表: 至少有title字段; 用LIKE做兜底, 向量索引由kb_search提供
        # 长查询先分词OR检索, 避免整句LIKE零命中
        for tok in _tokens(query):
            for r in con.execute("SELECT rowid, title FROM papers WHERE title LIKE ? LIMIT ?", ("%" + tok + "%", top_k)):
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
    if not os.path.exists(INCR_DB):
        return []
    con = sqlite3.connect(INCR_DB)
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
            local = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")
            if not os.path.isdir(local):
                snap = glob.glob(os.path.expanduser(
                    "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"))
                local = snap[0] if snap else "all-MiniLM-L6-v2"
            _st_model.cache = SentenceTransformer(local, device="cpu")
        except Exception:
            _st_model.cache = None
    return _st_model.cache


def search_pes2o_vector(query, top_k=5):
    """基座向量检索: papers.index(541744, MiniLM-384) + paper_ids.txt映射. 只读, CPU."""
    if not hasattr(search_pes2o_vector, "cache"):
        search_pes2o_vector.cache = "init"
    if search_pes2o_vector.cache == "init":
        try:
            import faiss
            import numpy as np
            idx_path = os.path.join(BASE_DIR, "papers.index")
            ids_path = os.path.join(BASE_DIR, "paper_ids.txt")
            if not (os.path.exists(idx_path) and os.path.exists(ids_path)):
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
        con = sqlite3.connect(os.path.join(BASE_DIR, "papers.db"))
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
            import glob
            import faiss
            import numpy as np
            for _v in ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
                       "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
                os.environ.pop(_v, None)
            os.environ["HF_HUB_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            idx_path = os.path.join(INCR_DIR, "arxiv.index")
            ids_path = os.path.join(INCR_DIR, "arxiv_ids.json")
            if not (os.path.exists(idx_path) and os.path.exists(ids_path)):
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
    con = sqlite3.connect(INCR_DB)
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
    诚实声明: boost是拍脑袋常数, 非学得权重; 调参需在标注相关性集上验证, 目前仅报告测量值。"""
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
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "LLM agent"
    print(json.dumps(fused_search(q), ensure_ascii=False, indent=2))
