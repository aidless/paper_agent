# 去污染报告
- 训练集: paper_agent/data/sft_fullprocess.jsonl (5条, D0/D1/D2/D4/D5)
- 评估集: paper_agent/data/eval_fixed.jsonl (8条, 冻结)
- 检查: 标题前30字符交集 = 0, arxiv_id/标题无交集
- 结论: PASS, 可用于eval
- 待办: 扩增SFT到50条时重跑本检查
