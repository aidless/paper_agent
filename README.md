# 论文全流程Agent (Qwen3-4B QLoRA + peS2o/arXiv双库)

[![CI](https://github.com/aidless/paper_agent/actions/workflows/static-checks.yml/badge.svg)](https://github.com/aidless/paper_agent/actions/workflows/static-checks.yml)

> **English TL;DR**: A minimal engineering package for a paper-workflow agent on Qwen3-4B QLoRA: SFT/DPO data construction, retrieval fusion over an external peS2o FAISS index plus an incremental arXiv corpus, inference, and three-way evaluation; the evidence chain is verifiable without a GPU via `python3 evidence/verify_tracked.py`. Its defining stance is honest negative reporting — the recorded P1 verdict flags template-shortcut learning over genuine writing ability, with numbers recomputable from committed JSON.

论文全流程 Agent 的最小工程包：SFT/DPO 数据构建、检索融合、推理与三方对比评估，全部代码与报告都在本仓内。

**验证过什么**：`reports/P1_VERDICT.md` 记录了 60 条 val（同一掩码）NLL 从 base 4.4904 降到 ckpt-270 1.1030，8 条贪心生成；同时坦诚判定 P1 门禁 FAIL（学到模板捷径而非写作能力）。数字可由 `reports/eval_3way.json` 逐样本复算。2026-11-10 干净 checkout 实测：`python3 evidence/verify_tracked.py` → `PASS: 53 tracked files present; 9 JSON files parse; evidence manifest matches`。

**怎么跑**（干净 clone，不需要 GPU 即可验证证据）：

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3 evidence/verify_tracked.py
```

## 一键运行（需 GPU）

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
python3 data/build_sft.py
python3 retrieval/build_incr_index.py
python3 retrieval/fuse.py "LLM agent evaluation"
python3 eval/eval_agent.py
python3 infer/infer_agent.py --prompt "LLM agent evaluation GAP与基线"
.venv/bin/python train/train_agent.py --smoke
.venv/bin/python train/train_agent.py
```

## 约定
- 基座语料 `$PE_S2O_KB_FAISS`（外部只读 FAISS 索引，541745 篇；不随仓库分发，需自行准备）
- 增量语料目录 `$ARXIV_INCREMENTAL`，独立于基座，每日 `--fetch` 增量更新
- 训练输出 `outputs/tmlr-qwen3-4b-agent` + `cfg_snapshot.json`
- TMLR: claim ledger必填, Abstract最后写, G1/G3/G4/G9/G10硬线

## 仓库结构

```
paper_agent/
├── data/       # SFT/DPO 数据构建与产物
│   │             build_sft.py / build_p0v2.py / build_dpo_v2.py → sft_*.jsonl, dpo_*.jsonl
├── retrieval/  # 双库检索融合：peS2o 基座（外部只读 FAISS）+ arXiv 增量
│   │             build_incr_index.py（建/更新增量索引）、fuse.py（融合查询）
├── train/      # QLoRA 训练：train_agent.py（SFT，含 --smoke）、train_dpo.py（DPO）
├── infer/      # 推理入口：infer_agent.py
├── eval/       # 三方对比评估：eval_3way.py / eval_agent.py / gen_probe.py
├── reports/    # 结论与判定：P1_VERDICT.md / P2_VERDICT.md / eval_3way*.json+md
└── evidence/   # 证据链：verify_tracked.py（免 GPU 离线校验）、MANIFEST.sha256、
                  env_fingerprint.json、make_manifest.py / make_inventory.py
```

数据流向：`data/`（构建语料）→ `retrieval/`（检索融合）→ `train/`（SFT→DPO）→
`infer/` + `eval/`（推理与三方评估）→ `reports/`（判定落盘）→ `evidence/`（全链哈希校验）。

## Reproduce offline

```bash
# Fresh-clone check: tracked files, committed JSON, and evidence hashes
python3 evidence/verify_tracked.py
```

The fresh-clone verifier deliberately does not validate
`evidence/data_manifest.sha256`. That dated snapshot includes local model
checkpoints and mutable external `/data1` corpora that are not redistributed.
On the original machine, `python3 evidence/make_manifest.py --check` can audit
that historical inventory, but missing or changed external files should be
reported as drift rather than silently replacing the snapshot.
