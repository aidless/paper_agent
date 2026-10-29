# 论文全流程Agent (Qwen3-4B QLoRA + peS2o/arXiv双库)

## 一键运行
```
/data1/paper_agent_venv/bin/pip install -r paper_agent/requirements.txt
/data1/paper_agent_venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
python3 paper_agent/data/build_sft.py
python3 arxiv_incremental.py --stats
python3 arxiv_incremental.py --fetch --days 7 --max-results 100 --query "LLM agent evaluation"
python3 paper_agent/retrieval/fuse.py "LLM agent evaluation"
python3 paper_agent/eval/eval_agent.py
python3 paper_agent/infer/infer_agent.py --prompt "LLM agent evaluation GAP与基线"
/data1/paper_agent_venv/bin/python paper_agent/train/train_agent.py --smoke
/data1/paper_agent_venv/bin/python paper_agent/train/train_agent.py
```

## 约定
- 基座 /data1/peS2o_kb_faiss 只读 (541745篇)
- 增量 /data1/arxiv_incremental 独立, 每日 --fetch
- venv在 /data1/paper_agent_venv (工作目录含冒号无法直接venv)
- 训练输出 paper_agent/outputs/tmlr-qwen3-4b-agent + cfg_snapshot.json
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
