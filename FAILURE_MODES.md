# FAILURE_MODES

Two ways this package can look healthier than the evidence.

## Template-shortcut false greens

`reports/P1_VERDICT.md` is the gate, and the gate is **FAIL**. On the recorded 60-row mask, val NLL falls from 4.4904 (base) to 1.1030 (ckpt-68 and ckpt-270) while stage-tag rate stays 0. Several greedy samples are about 15 characters (early EOS). distinct-2 can rise at the same time. That NLL drop is the false green: the P0 data is template-regular, and the model follows the template.

`eval/eval_3way.py` scores "evidence binding" with a regex (`证据|arxiv|year|venue|绑定|无证据` and similar). A template can match it. `reports/eval_3way_v2.json` later records val NLL 0.3879 and evidence-binding 1.0. Those stored scalars do not reverse the P1 FAIL, and this clone cannot recompute them from logits. ckpt-68 and ckpt-270 (and, in the v2 file, ckpt136 and ckpt_final) share generation rows. The P1 note treats that as greedy decoding meeting the same text, not as a second independent success.

`reports/eval_results.json` is a different trap. `task_success` 0.625 is the mean of the `pred_ok` flags already stored in `data/eval_fixed.jsonl` (5 of 8). It is not a model score. `grounded_rate` 0.875 was a lookup against external SQLite. It is not recomputed offline. `eval/eval_agent.py` used to invent eight placeholder rows when the jsonl was missing, then write a new report. It now exits 2 instead.

## Missing external index

Retrieval and grounded eval read two directories that are not in git:

- `PE_S2O_KB_FAISS` — historical original-machine path `/data1/peS2o_kb_faiss`
- `ARXIV_INCREMENTAL` — historical original-machine path `/data1/arxiv_incremental`

`retrieval/fuse.py`, `retrieval/build_incr_index.py`, `eval/eval_agent.py`, and `infer/infer_agent.py` exit 2 when those variables are unset or the index files are absent. They used to return an empty hit list or a draft that said zero hits, which looks like a query that ran. An empty list is not a successful search.

Data rebuilds (`data/build_p0.py`, `data/build_p0v2.py`, `data/build_dpo_v2.py`) overwrite committed jsonl. They require `PAPER_CORPUS_PARQUET` and `--write`. The historical parquet path is `/data3/Research/corpus_hf/all-papers.parquet`. It is not a default.

`evidence/make_manifest.py` and `evidence/env_fingerprint.json` still contain the original machine's absolute paths. They are a hashed snapshot. `evidence/verify_tracked.py` does not open `/data1` and does not rewrite that snapshot. On a fresh clone, `python3 evidence/make_manifest.py --check` should report the external files missing. That is drift of the historical inventory, not a reason to regenerate it.

## Counts that must not be collapsed

These are different snapshots and different fields. The offline verifier checks that the files still disagree, and does not pick a winner:

| Where | Figure | What it is |
|---|---|---|
| `reports/P1_VERDICT.md` (2026-09-10) | arXiv 16902 | count written in that dated note |
| `evidence/env_fingerprint.json` (2026-10-08) | arXiv 17263, peS2o rows 541745 | sqlite row counts on the fingerprint machine |
| `reports/RETRIEVAL_UPGRADE.md` | 541744 vectors | historical `papers.index` size, not the row count |

## What the offline verifier does not prove

`python3 evidence/verify_tracked.py` checks the git checkout, `evidence/MANIFEST.sha256`, JSON parse, generation aggregates recomputed from `gen_detail`, the README claim block, and that the P1 file still says FAIL. It does not run Qwen, does not hash checkpoints (they are gitignored), and does not recompute token NLL. A lower loss is not a writing result.
