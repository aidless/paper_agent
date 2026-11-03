# 论文全流程Agent (Qwen3-4B QLoRA + peS2o/arXiv双库)

论文全流程 Agent 的最小工程包：SFT/DPO 数据构建、检索融合、推理与三方对比评估，全部代码与报告都在本仓内。

**验证过什么**：`reports/P1_VERDICT.md` 记录了 60 条 val（同一掩码）NLL 从 base 4.4904 降到 ckpt-270 1.1030，8 条贪心生成；同时坦诚判定 P1 门禁 FAIL（学到模板捷径而非写作能力）。数字可由 `reports/eval_3way.json` 逐样本复算。

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
