# -*- coding: utf-8 -*-
"""SFT样本生成器: D0-D5全流程轨迹, 强制evidence_refs+arxiv_grounding字段."""
import json, os
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "sft_fullprocess.jsonl")

SEEDS = [
 {"task_stage": "D0", "instruction": "为LLM agent评测方向做文献综述, 给出GAP与基线池",
  "tools": ["fused_search"], "thought": "先双路检索peS2o+arXiv增量, 按方法族归纳",
  "action": "fused_search('LLM agent evaluation')", "observation": "召回peS2o经典+arXiv近7天增量",
  "final": "GAP1: 推理轨迹评估缺失细粒度指标; 基线: AgentBench, WebArena, SciAgentArena",
  "evidence_refs": ["papers.db:title"], "arxiv_grounding": ["arXiv:2407.xxxxx"]},
 {"task_stage": "D1", "instruction": "把GAP转成可证伪假设, 设计消融实验",
  "tools": ["ledger_check"], "thought": "假设必须可证伪, 绑定基线与指标",
  "action": "ledger_check(claim='轨迹长度与成功率负相关')", "observation": "需补充统计功效与样本量",
  "final": "H1: 轨迹压缩10%成功率变化<2%; 消融: 移除反思模块对照",
  "evidence_refs": ["protocol.md:H1"], "arxiv_grounding": []},
 {"task_stage": "D2", "instruction": "写出可复现实验配置, 种子管理",
  "tools": ["code"], "thought": "配置即证据, 种子固定",
  "action": "run(seed=42, cfg_snapshot)", "observation": "cfg_snapshot.json已落盘",
  "final": "seed=42, n=30, alpha=0.05, power=0.80",
  "evidence_refs": ["cfg_snapshot.json"], "arxiv_grounding": []},
 {"task_stage": "D4", "instruction": "从证据值直写Methods+Results, Abstract最后写",
  "tools": ["verify_claim_ledger", "scan_number"], "thought": "数字必须重算, 不得跨节复制",
  "action": "verify_claim_ledger()", "observation": "PASS, 无uncovered数值",
  "final": "Methods报告N=30, McNemar exact p=0.042; Results与ledger一致",
  "evidence_refs": ["results.json:McNemar"], "arxiv_grounding": []},
 {"task_stage": "D5", "instruction": "TMLR G1-G11合规打包",
  "tools": ["check_tmlr", "build_manifest"], "thought": "模板匿名+双盲隔离+哈希对账",
  "action": "check_tmlr()", "observation": "G1/G3/G4 PASS",
  "final": "submission.zip + evidence_manifest.json SHA-256一致",
  "evidence_refs": ["evidence_manifest.json"], "arxiv_grounding": []},
]

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for s in SEEDS:
            f.write(json.dumps({"messages": [
                {"role": "user", "content": s["instruction"]},
                {"role": "assistant", "content": s["final"]}],
                "task_stage": s["task_stage"], "tools": s["tools"],
                "thought": s["thought"], "action": s["action"],
                "observation": s["observation"],
                "evidence_refs": s["evidence_refs"],
                "arxiv_grounding": s["arxiv_grounding"],
                "target_venue": "TMLR"}, ensure_ascii=False) + "\n")
    print("wrote %s (%d)" % (OUT, len(SEEDS)))

if __name__ == "__main__":
    main()
