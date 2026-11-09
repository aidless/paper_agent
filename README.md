# paper_agent

[![CI](https://github.com/aidless/paper_agent/actions/workflows/static-checks.yml/badge.svg)](https://github.com/aidless/paper_agent/actions/workflows/static-checks.yml)

STATUS: WORK-IN-PROGRESS — not A yet.

## Problem

Offline, `python3 evidence/verify_tracked.py` checks tracked files, `evidence/MANIFEST.sha256`, and generation aggregates recomputed from `gen_detail` in `reports/eval_3way.json`.
That JSON is the P1 gate **FAIL**: 60-row val NLL 4.4904 (base) and 1.1030 (ckpt-270). The drop is template-shortcut learning, not writing ability.
Training, inference, and retrieval need a GPU plus external `PE_S2O_KB_FAISS` and `ARXIV_INCREMENTAL` (historical `/data1/...` layout, not in this clone).

<!-- claims:eval_3way.json
base.val_nll_60=4.4904
ckpt270.val_nll_60=1.1030
p1_gate=FAIL
-->

## Offline path

A git clone, no GPU and no pip install. Expect seconds, not a training run. The command needs `git ls-files` (a GitHub zip of the tree without `.git` cannot run it).

```bash
python3 evidence/verify_tracked.py
```

`.github/workflows/static-checks.yml` runs that command after a Python syntax compile and a JSON parse. `evidence/data_manifest.sha256` is a historical inventory of checkpoints and external files. The verifier does not treat a missing external tree as success. On the original machine, `python3 evidence/make_manifest.py --check` audits that inventory; missing `/data1` files there are drift, not a cue to regenerate the snapshot.

The Chinese gate write-up is `reports/P1_VERDICT.md`. It stays FAIL.

## Not the offline path

Set `PE_S2O_KB_FAISS` and `ARXIV_INCREMENTAL` to indexes you provide. Historical directories were `/data1/peS2o_kb_faiss` and `/data1/arxiv_incremental`. Unset or missing indexes exit 2. See `FAILURE_MODES.md`.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# GPU and the external indexes. Not run for the numbers in this README.
python3 retrieval/fuse.py "LLM agent evaluation"
python3 eval/eval_agent.py
python3 infer/infer_agent.py --prompt "LLM agent evaluation GAP与基线" --no-model
```

Training entry points are `train/train_agent.py` and `train/train_dpo.py`. Rebuilding data (`data/build_p0.py` and the v2 builders) needs `PAPER_CORPUS_PARQUET` and `--write`, and overwrites committed jsonl. The historical parquet was `/data3/Research/corpus_hf/all-papers.parquet`. This tree does not contain it, and this change did not rebuild data or train.

## Gaps (why this is not A)

1. `val_nll_60` is a stored scalar. Token losses are not in the JSON, so 4.4904 and 1.1030 are locked to the file, not recomputed from logits.
2. Checkpoints and both FAISS corpora are outside the clone. Train, infer, and retrieval were not re-run here.
3. `reports/eval_3way_v2.json` records a later val NLL 0.3879. That number does not reverse the P1 FAIL. It is not recomputed from logits either.
4. Dated corpus counts disagree across files. They are listed in `FAILURE_MODES.md` and were not re-counted.
