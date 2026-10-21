"""P0 data construction (bootstrap): synthesize full-process SFT trajectories
from REAL corpus metadata (title/abstract/tldr/paper_id).
Each sampled paper yields 6 stage examples (D0..D5). Split by paper into
train/val. Also emits DPO preference pairs (grounded vs citation-dropped).
All evidence_refs point to real paper_ids. Honest label: template-synthesized.
"""
import json, random, sys
import pandas as pd

CORPUS = "/data3/Research/corpus_hf/all-papers.parquet"
OUT_TRAIN = "paper_agent/data/sft_p0_train.jsonl"
OUT_VAL = "paper_agent/data/sft_p0_val.jsonl"
OUT_DPO = "paper_agent/data/dpo_p0.jsonl"
N_PAPERS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SEED = 42

STAGES = ["D0", "D1", "D2", "D3", "D4", "D5"]
STAGE_TASK = {
 "D0": "为该方向做文献综述, 给出GAP与基线池",
 "D1": "把GAP转成可证伪假设, 设计消融实验",
 "D2": "写出可复现实验配置, 含种子管理",
 "D3": "起草方法段落, 每个声明绑定证据",
 "D4": "扮演审稿人攻击上一步草稿, 逐条回应并修改",
 "D5": "做TMLR合规自查与数字一致性核对",
}
STAGE_TOOL = {"D0": "fused_search", "D1": "ledger_check", "D2": "code",
              "D3": "write", "D4": "critic", "D5": "check"}

def S(x):
    try:
        if x is None or pd.isna(x): return ""
    except Exception:
        pass
    return str(x)

def build_examples(row):
    pid, title, abs_, tldr = S(row["paper_id"]), S(row["title"]), S(row["abstract"])[:1200], S(row["tldr"])
    venue = row["conference"] or "TMLR"
    exs = []
    for st in STAGES:
        obs = "召回: [%s] %s | 摘要: %s | TLDR: %s" % (pid, title, abs_[:600], tldr[:300])
        ex = {
          "messages": [
            {"role": "user", "content": "方向: %s. %s" % (title, STAGE_TASK[st])},
            {"role": "assistant", "content": "[%s] 基于证据(%s)处理: %s…; 结论已绑定证据, 无证据不做强声明。" % (st, pid, STAGE_TASK[st][:20])}],
          "task_stage": st, "tools": [STAGE_TOOL[st]],
          "thought": "分步处理, 声明绑定证据%s" % pid,
          "action": "%s(step=%s)" % (STAGE_TOOL[st], st),
          "observation": obs,
          "evidence_refs": [str(pid)],
          "arxiv_grounding": [str(row["arxiv_id"])] if row["arxiv_id"] else [],
          "target_venue": str(venue), "synthetic": "p0-template",
        }
        exs.append(ex)
    # DPO pair: grounded revision (chosen) vs citation-dropped (rejected)
    dpo = {"prompt": "修改以下段落, 保留证据绑定: [%s] %s" % (pid, title),
           "chosen": "修改后: 结论A(证据%s); 结论B(证据%s)。无证据的推测已删除。" % (pid, pid),
           "rejected": "修改后: 结论A显然成立; 结论B得到广泛认同。 (无证据绑定)",
           "paper_id": str(pid)}
    return exs, dpo

def main():
    random.seed(SEED)
    df = pd.read_parquet(CORPUS)
    df = df[df["title"].notna() & df["abstract"].notna()]
    df = df[df["abstract"].str.len() > 200]
    print("[p0] eligible papers:", len(df))
    idx = random.sample(range(len(df)), min(N_PAPERS, len(df)))
    samp = df.iloc[idx].reset_index(drop=True)
    n_val = max(1, int(len(samp) * 0.1))
    val_ids = set(random.sample(range(len(samp)), n_val))
    n_tr = n_va = n_dpo = 0
    with open(OUT_TRAIN, "w") as ft, open(OUT_VAL, "w") as fv, open(OUT_DPO, "w") as fd:
        for i, row in samp.iterrows():
            exs, dpo = build_examples(row)
            f = fv if i in val_ids else ft
            for e in exs:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
                if i in val_ids: n_va += 1
                else: n_tr += 1
            fd.write(json.dumps(dpo, ensure_ascii=False) + "\n"); n_dpo += 1
    print("[p0] train=%d val=%d dpo=%d papers=%d" % (n_tr, n_va, n_dpo, len(samp)))

if __name__ == "__main__":
    main()
